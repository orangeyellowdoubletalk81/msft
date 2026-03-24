# mSFT: Addressing Dataset Mixtures OverfitingHeterogeneously in Multi-task SFT

## Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPUs
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

---

## Quick Start

### 1. Data

Multi-task dataset and few-shot examples are under `data/`. The repository includes example mixtures:

```
data/
├── mix-5cat-9k.json      # 5 categories, ~9k samples
├── mix-10cat-18k.json     # 10 categories, ~18k samples
├── mix-15cat-27k.json     # 15 categories, ~27k samples
└── fiveshot.json          # Few-shot evaluation prompts
```

### 2. Configure

Edit the config file at `sft/configs/msft-config.json`. \
The default configuration assumes 4× RTX 3090 GPUs with an effective batch size of 64 (`batch_size` 4 × `grad_accumulation` 4 × 4 GPUs).

```jsonc
{
    "model_name": "Qwen/Qwen2.5-3B",
    "dataset_file": "data/mix-10cat-18k.json",
    "epochs": 10,
    "batch_size": 4,
    "grad_accumulation": 4,
    "lr": 1e-5,
    "eval_divisor": 4          // evaluations per epoch
}
```

### 3. Run mSFT

```bash
cd sft
bash run_mSFT.sh
```

Key parameters in `run_mSFT.sh`:

| Parameter | Default | Description |
|---|---|---|
| `CONFIG_FILE` | `configs/msft-config.json` | Training config path |
| `MAX_STAGES` | `10` | Maximum mSFT iterations |
| `GPU_IDS` | `""` (all) | Comma-separated GPU IDs |
| `DROP_COUNT` | `1` | Sub-datasets to drop per stage |
| `SEED` | `20` | Random seed |

---

## File Structure

```
├── requirements.txt
├── data/                        # Dataset files
│   ├── mix-*.json               # Multi-task mixtures
│   └── fiveshot.json            # Few-shot prompts
└── sft/
    ├── run_mSFT.sh              # Main pipeline script
    ├── train_eval.py            # Training and evaluation loop
    ├── analyze_history.py       # Overfitting detection & stage analysis
    ├── configs/
    │   └── msft-config.json     # Training configuration
    ├── settings/
    │   └── configs.py           # Dataclass definitions for settings
    ├── modules/                 # Training modules (event system, trackers, updaters)
    └── utils/                   # Utilities (data loading, evaluation, vLLM inference)
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{koh2026msftaddressingdatasetmixtures,
      title={mSFT: Addressing Dataset Mixtures Overfiting Heterogeneously in Multi-task SFT}, 
      author={Woosung Koh and Jeyoung Jeon and Youngjin Song and Yujin Cheon and Soowon Oh and Jaehyeong Choi and Se-Young Yun},
      year={2026},
      eprint={2603.21606},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.21606}, 
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
