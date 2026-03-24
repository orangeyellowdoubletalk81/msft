import torch
import torch.distributed as dist
import torch.optim as optim
from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_scheduler

import os
import json
import functools
from typing import ClassVar
from dataclasses import dataclass, field, asdict

from utils.utils import setup_model

@dataclass
class TrainBatchSettings:
    dict_excluded: ClassVar[list[str]] = []
    batch_size: int
    grad_accumulation: int
    
    @property
    def effective_batch_size(self):
        return self.batch_size * self.grad_accumulation * torch.cuda.device_count()


@dataclass
class TestBatchSettings:
    dict_excluded: ClassVar[list[str]] = []
    batch_size: int

    @property
    def effective_batch_size(self):
        return self.batch_size * torch.cuda.device_count()


@dataclass
class TrainSettings:
    dict_excluded: ClassVar[list[str]] = []
    epochs: int
    lr: float
    weight_decay: float
    warm_up_ratio: float
    scheduler_name: str
    optimizer_name: str
    batch_settings: TrainBatchSettings

    def __post_init__(self):
        if 0 > self.warm_up_ratio or 1 < self.warm_up_ratio:
            raise ValueError(f"{self.warm_up_ratio} must be between 0 and 1.")

    def get_optimizer(self, model):
        match self.optimizer_name:
            case "AdamW":
                return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            case "Adam":
                return optim.Adam(model.parameters(), lr=self.lr)
            case _:
                raise ValueError(f"{self.optimizer_name} is not valid optimizer name.")

    def get_scheduler(self, optimizer, num_training_steps):
        num_warmup_steps = max(0, int(num_training_steps * self.warm_up_ratio))
        return get_scheduler(name=self.scheduler_name, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    @property
    def grad_accumulation(self):
        return self.batch_settings.grad_accumulation

    @property
    def batch_size(self):
        return self.batch_settings.batch_size

    @property
    def effective_batch_size(self):
        return self.batch_settings.effective_batch_size


@dataclass
class TestSettings:
    dict_excluded: ClassVar[list[str]] = []
    eval_divisor: int
    gen_length: int
    is_zeroshot: bool
    batch_settings: TestBatchSettings

    @property
    def batch_size(self):
        return self.batch_settings.batch_size

    @property
    def effective_batch_size(self):
        return self.batch_settings.effective_batch_size


@dataclass
class ModelSettings:
    dict_excluded: ClassVar[list[str]] = ["precision", "mixed_precision"]
    model_name: str
    precision_name: str
    precision: torch.dtype = field(init=False)
    mixed_precision: MixedPrecision = field(init=False)

    def __post_init__(self):
        match self.precision_name:
            case "bf16":
                self.precision = torch.bfloat16
            case "fp16":
                self.precision = torch.float16
            case _:
                raise ValueError(f"{self.precision_name}은 옳지 않은 precision입니다.")
        self.mixed_precision = MixedPrecision(
            param_dtype=self.precision,
            reduce_dtype=self.precision,
            buffer_dtype=self.precision
        )

    def get_fsdp_wrapped_model(self, rank):
        model = setup_model(self.model_name, self.precision)
        return self._fsdp_wrap(model, rank)

    def get_model_with_hf_checkpoint(self, ckpt_dir, rank):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, trust_remote_code=True, torch_dtype=self.precision
        )
        if rank == 0:
            print(f"Loaded HF checkpoint from {ckpt_dir}")
        return self._fsdp_wrap(model, rank)

    def get_ddp_copy_model(self, model, rank):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        full_state_dict = _get_full_state_dict(model, cfg)
        copied_model = setup_model(self.model_name, self.precision)
        return _set_full_state_dict(copied_model, full_state_dict, rank)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _fsdp_wrap(self, model, rank):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100
        )
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=self.mixed_precision,
            device_id=torch.device(rank)
        )


@dataclass
class LogAndSaveSettings:
    dict_excluded: ClassVar[list[str]] = []
    exp_name: str
    log_dir: str
    ckpt_dir: str
    log_divisor: int
    save_divisor: int
    resume_ckpt_dir: str

    adaptive_checkpointing: bool = False

    log_directory: str = field(init=False)
    ckpt_directory: str = field(init=False)


    def __post_init__(self):
        while os.path.exists(self.exp_name):
            self.exp_name = self.exp_name + "_"
        os.makedirs(self.exp_name, exist_ok=True)
        self.log_directory = os.path.join(self.exp_name, self.log_dir)
        self.ckpt_directory = os.path.join(self.exp_name, self.ckpt_dir)
        
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(self.ckpt_directory, exist_ok=True)

    @property
    def sanity_check_log_path(self):
        return os.path.join(self.exp_name, "sanity_check.txt")

    @property
    def config_log_path(self):
        return os.path.join(self.log_directory, "config.json")

    @property
    def gradient_log_path(self):
        return os.path.join(self.log_directory, "grad_log.json")
    
    @property
    def train_loss_log_path(self):
        return os.path.join(self.log_directory, "train_log.json")

    @property
    def accuracy_log_path(self):
        return os.path.join(self.log_directory, "accuracy_log.json")
      
    @property      
    def debug_log_path(self):
        return os.path.join(self.log_directory, "debug.json")

    def get_ckpt_save_path(self, save_epoch):
        return os.path.join(self.ckpt_directory, f"model-{save_epoch:.2f}.pth")

    def get_hf_ckpt_save_dir(self, save_epoch):
        return os.path.join(self.ckpt_directory, f"hf-model-{save_epoch:.2f}")

    @property
    def eval_tmp_ckpt_dir(self):
        return os.path.join(self.ckpt_directory, "_vllm_eval_tmp")

    def get_prediction_log_path(self, epoch: float):
        return os.path.join(self.log_directory, f"prediction_{epoch:.2f}.txt")

@dataclass
class DatasetSettings:
    dict_excluded: ClassVar[list[str]] = ["dataset_path", "fiveshot_prompt_path"]
    ROOT: ClassVar[str] = ".."
    dataset_file: str
    fiveshot_prompt_file: str
    max_train_length: int
    max_test_length: int

    dataset_path: str = field(init=False)
    fiveshot_prompt_path: str = field(init=False)

    def __post_init__(self):
        self.dataset_path = os.path.join(DatasetSettings.ROOT, self.dataset_file)
        self.fiveshot_prompt_path = os.path.join(DatasetSettings.ROOT, self.fiveshot_prompt_file)


def _get_full_state_dict(model, cfg):
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        full_state_dict = model.state_dict()
    dist.barrier()
    return full_state_dict


def _set_full_state_dict(model, full_state_dict, rank):
    model.load_state_dict(full_state_dict)
    model = model.to(rank)
    model = DDP(model, device_ids = [rank], output_device=rank)
    return model


def log_config(args, dataset_settings, train_settings, test_settings, model_settings, log_and_save_settings):
    configfile = log_and_save_settings.config_log_path

    configs = vars(args)
    for (name, setting) in [("dataset", dataset_settings), ("train", train_settings), ("test", test_settings), ("model", model_settings), ("log_and_save", log_and_save_settings)]:
        configs[name] = asdict(
                setting,
                dict_factory=lambda data: {key: value for key, value in data if key not in setting.dict_excluded}
            )
    
    with open(configfile, "w") as json_file:
        json.dump(configs, json_file, indent=4)