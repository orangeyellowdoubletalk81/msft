import gc
import socket

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

import json
import os
import shutil
import subprocess
import sys
import tempfile

from utils.utils import setup_model
from utils.utils_test import _parse_final_answer, _calculate_metrics, output_logs


def save_hf_checkpoint(model, tokenizer, save_dir, rank, model_settings):
    dist.barrier()
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = setup_model(model_settings.model_name, model_settings.precision)
        unwrapped.load_state_dict(state_dict)
        unwrapped.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        del unwrapped

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def persist_checkpoint(tmp_dir, dest_dir, rank):
    if rank == 0:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(tmp_dir, dest_dir)
        os.utime(dest_dir, None)
    dist.barrier()


def delete_checkpoint_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Pruned checkpoint: {os.path.basename(path)}")
    elif os.path.isfile(path):
        os.remove(path)
        print(f"Pruned checkpoint: {os.path.basename(path)}")


def vllm_evaluate(model, tokenizer, rank, world_size, model_settings, test_settings,
                  log_and_save_settings, dataloader, epoch, float_epoch,
                  dataset_settings=None):
    tmp_ckpt_dir = log_and_save_settings.eval_tmp_ckpt_dir
    save_hf_checkpoint(model, tokenizer, tmp_ckpt_dir, rank, model_settings)

    all_test_data = _collect_test_data(dataloader)

    shard_size = (len(all_test_data) + world_size - 1) // world_size
    my_shard = all_test_data[rank * shard_size : (rank + 1) * shard_size]

    local_metrics, local_logs = None, []
    if len(my_shard) > 0:
        local_metrics, local_logs = _run_vllm_inference(
            tmp_ckpt_dir, model_settings, test_settings,
            my_shard, rank=rank, dataset_settings=dataset_settings
        )

    all_logs_list = [None] * world_size
    dist.all_gather_object(all_logs_list, local_logs)

    final_metrics = None
    if rank == 0:
        merged_logs = []
        for logs in all_logs_list:
            merged_logs.extend(logs)

        pred_file = log_and_save_settings.get_prediction_log_path(float_epoch)

        eval_results = []
        for item in merged_logs:
            parsed_prediction = _parse_final_answer(item["prediction"])
            parsed_ground_truth = _parse_final_answer(item["ground_truth"])
            verdict = (parsed_prediction is not None) and (parsed_prediction == parsed_ground_truth)
            eval_results.append({"category": item["category"], "verdict": verdict})
        final_metrics = _calculate_metrics(eval_results)
        output_logs(pred_file, merged_logs, float_epoch)

    metrics_list = [final_metrics]
    dist.broadcast_object_list(metrics_list, src=0)
    final_metrics = metrics_list[0]

    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    return final_metrics


def _collect_test_data(dataloader):
    all_data = []
    dataset = dataloader.dataset_test
    for i in range(len(dataset)):
        item = dataset[i]
        all_data.append({
            "prompt_text": item["prompt_text"],
            "ground_truth": item["ground_truth"],
            "golden_example": item["golden_example"],
            "category": item["category"],
        })
    return all_data


# Strip training distributed env vars to avoid conflicts with vLLM subprocess
_DISTRIBUTED_ENV_VARS = [
    "MASTER_ADDR", "MASTER_PORT",
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "VLLM_HOST_IP", "VLLM_PORT",
]

_SUBPROCESS_SCRIPT = os.path.join(os.path.dirname(__file__), "vllm_eval_subprocess.py")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_vllm_inference(model_path, model_settings, test_settings, test_data,
                        rank=0, dataset_settings=None):
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in _DISTRIBUTED_ENV_VARS}

    clean_env["MASTER_ADDR"] = "127.0.0.1"
    clean_env["MASTER_PORT"] = str(_find_free_port())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        gpu_ids = visible.split(",")
        clean_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
    else:
        clean_env["CUDA_VISIBLE_DEVICES"] = str(rank)
    clean_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_path = os.path.join(tmpdir, "test_data.json")
        output_path = os.path.join(tmpdir, "results.json")

        with open(test_data_path, "w") as f:
            json.dump(test_data, f)

        _free, _total = torch.cuda.mem_get_info(rank)
        gpu_mem_util = round(max(0.15, min(0.85, (_free / _total) * 0.85)), 2)

        cfg = {
            "model_path": model_path,
            "dtype": "bfloat16" if model_settings.precision_name == "bf16" else "float16",
            "gpu_mem_util": gpu_mem_util,
            "max_model_len": (
                dataset_settings.max_test_length + test_settings.gen_length
                if dataset_settings is not None else 2048
            ),
            "gen_length": test_settings.gen_length,
            "test_data_path": test_data_path,
            "output_path": output_path,
        }

        proc = subprocess.Popen(
            [sys.executable, _SUBPROCESS_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=clean_env,
            start_new_session=True,
        )
        proc.communicate(input=json.dumps(cfg).encode())

        if proc.returncode != 0:
            raise RuntimeError(
                f"vLLM eval subprocess failed with exit code {proc.returncode}"
            )

        with open(output_path) as f:
            results = json.load(f)

    logs = results["logs"]
    return None, logs
