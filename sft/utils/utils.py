import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig, StateDictType, FullStateDictConfig, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import logging
import random
import glob
from datetime import timedelta

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    if os.environ.get('MASTER_PORT') is None:
        os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(hours=2),
                            device_id=torch.device("cuda", rank))


def is_qwen(model_name):
    return "Qwen" in model_name


def setup_model(model_name, model_dtype=torch.bfloat16):
    if is_qwen(model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=model_dtype)
    else :
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype)
    return model


def log_args(output_file, **kwargs):
    if os.path.exists(output_file):
        with open(output_file, "r") as json_file:
            logs = json.load(json_file)
    else:
        logs = []
    
    logs.append(kwargs)
    with open(output_file, "w") as json_file:
        json.dump(logs, json_file, indent=4)


def save_model(log_and_save_settings, model, rank, eval_epoch):
    dist.barrier()
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
        if rank==0:
            torch.save(state_dict, log_and_save_settings.get_ckpt_save_path(eval_epoch))
            print("saved model in:", log_and_save_settings.ckpt_dir)
    dist.barrier()


def load_model_epoch_of(log_and_save_settings, model, rank, float_epoch):
    dist.barrier()
    state_dict = torch.load(log_and_save_settings.get_ckpt_save_path(float_epoch))
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        model.load_state_dict(state_dict)
    if rank == 0:
        print(f"Loaded checkpoint from {log_and_save_settings.resume_ckpt_dir}")
    dist.barrier()


def load_model(log_and_save_settings, model, rank):
    if not os.path.exists(log_and_save_settings.resume_ckpt_dir):
        if rank == 0:
            print(f"\nWARNING: Checkpoint path {log_and_save_settings.resume_ckpt_dir} does not exist.")
        dist.barrier()
        return

    dist.barrier()
    state_dict = torch.load(log_and_save_settings.resume_ckpt_dir, map_location="cpu")
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        model.load_state_dict(state_dict)
    if rank == 0:
        print(f"\nLoaded checkpoint from {log_and_save_settings.resume_ckpt_dir}")
    dist.barrier()        

def _round_metrics(obj, ndigits=4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_metrics(v, ndigits) for k, v in obj.items()}
    return obj


def update_accuracy_log(accuracy_log_path, new_metrics, log_epoch):
    try:
        with open(accuracy_log_path, "r", encoding='utf-8') as f:
            accuracy_log = json.load(f)
    except FileNotFoundError:
        accuracy_log = {}

    accuracy_log[f"{log_epoch:.2f}"] = _round_metrics(new_metrics)
    with open(accuracy_log_path, "w", encoding='utf-8') as f:
        json.dump(accuracy_log, f, indent=4)

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event= torch.cuda.Event(enable_timing=True)
    
    def record_start(self):
        self.start_event.record()
    
    def record_end(self):
        self.end_event.record()
    
    def print_elapsed_time(self, rank):
        if rank != 0:
            return

        self.end_event.synchronize()

        print(f"CUDA event elapsed time: {self.start_event.elapsed_time(self.end_event) / 1000}sec")
