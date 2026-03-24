import gc
import json
import os
import sys
import tempfile

import torch

# Pre-init torch.distributed with FileStore before vLLM import
import torch.distributed as dist

if not dist.is_initialized():
    _store_file = tempfile.mktemp(prefix="vllm_dist_", suffix=".store")
    _store = dist.FileStore(_store_file, 1)   # world_size=1, TP=1
    dist.init_process_group(
        backend="gloo",
        store=_store,
        rank=0,
        world_size=1,
    )

from vllm import LLM, SamplingParams


def _clean_text(text: str) -> str:
    idx = text.find("<|endoftext|>")
    return text[:idx].strip() if idx != -1 else text


def main():
    cfg = json.load(sys.stdin)

    llm = LLM(
        model=cfg["model_path"],
        dtype=cfg["dtype"],
        trust_remote_code=True,
        gpu_memory_utilization=cfg["gpu_mem_util"],
        max_model_len=cfg["max_model_len"],
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        max_num_seqs=cfg.get("max_num_seqs", 32),
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=cfg["gen_length"],
    )

    with open(cfg["test_data_path"]) as f:
        test_data = json.load(f)

    prompts = [item["prompt_text"] for item in test_data]
    outputs = llm.generate(prompts, sampling_params)

    logs = []
    for i, (output, item) in enumerate(zip(outputs, test_data)):
        prediction = _clean_text(output.outputs[0].text)
        logs.append({
            "rank": 0,
            "local_counter": i,
            "input": item["prompt_text"],
            "prediction": prediction,
            "ground_truth": item["ground_truth"],
            "golden_example": item["golden_example"],
            "category": item["category"],
        })

    with open(cfg["output_path"], "w") as f:
        json.dump({"logs": logs}, f)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
