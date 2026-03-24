import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

import json
import logging
from dataclasses import dataclass
from tqdm import tqdm

OUTPUT_EXIST = [
    "deepmind/aqua_rat/raw",
    "openai/gsm8k/main"
]

ANSWER_EXIST = [
    "allenai/ai2_arc/ARC-Easy",
    "allenai/openbookqa/main",
    "allenai/sciq",
    "allenai/winogrande/winogrande_debiased",
    "deepmind/aqua_rat/raw",
    "google/boolq",
    "openai/gsm8k/main",
    "openlifescienceai/medmcqa",
    "Rowan/hellaswag",
    "tau/commonsense_qa"
]

QUESTION_HEADER = "Q:"
ANSEWR_HEADER = "A:"
OUTPUT_PROMPT_TEMPLATE = "####"


def _extract_prompts(inputs, num_examples):
    return [inputs[i][0]["content"].strip() for i in range(num_examples)]


def _integrate_answers(categories, outputs, answers, use_output: bool = True) -> list:
    global OUTPUT_EXIST
    integrated_answers = []
    for cat, output, answer in zip(categories, outputs, answers):
        if use_output and cat in OUTPUT_EXIST:
            integrated_answers.append(output)
            continue

        integrated_answers.append(answer)
    return integrated_answers


def _get_data_fields(examples, use_output: bool = True):
    inputs, categories = examples["input"], examples["category"]
    num_examples = len(inputs)
    prompts = _extract_prompts(inputs, num_examples)
    outputs, ground_truths = examples.get("output", [None] * num_examples), examples["answer"]
    golden_examples = _integrate_answers(categories, outputs, ground_truths, use_output)
    return num_examples, prompts, categories, ground_truths, golden_examples


def _get_train_prompt_text(prompt, category, golden_example, eos_token, use_output: bool):
    global QUESTION_HEADER, ANSEWR_HEADER, OUTPUT_PROMPT_TEMPLATE, OUTPUT_EXIST
    if use_output and category in OUTPUT_EXIST:
        return f"{QUESTION_HEADER} {prompt}\n{ANSEWR_HEADER} {golden_example}{eos_token}"
    
    return f"{QUESTION_HEADER} {prompt}\n{ANSEWR_HEADER} {OUTPUT_PROMPT_TEMPLATE} {golden_example}{eos_token}"


def _get_test_with_five_shot_prompt_text(fiveshot_prompt_template, prompt):
    global QUESTION_HEADER, ANSEWR_HEADER, OUTPUT_PROMPT_TEMPLATE, OUTPUT_EXIST
    return fiveshot_prompt_template.replace("{question}", prompt)


def _mask_input_prompt(prompt, tokenizer, input_ids, attention_mask):
    global QUESTION_HEADER, ANSEWR_HEADER
    prompt_only_text = f"{QUESTION_HEADER} {prompt}\n{ANSEWR_HEADER} "
            
    prompt_ids = tokenizer(prompt_only_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    
    mask_region = attention_mask.index(1) + prompt_len - 1
    labels = input_ids.copy()
    for j in range(min(mask_region, len(labels))):
        labels[j] = -100
    return prompt_len, mask_region, labels


def _log_input_ids_after_masking(log_and_save_settings, prompt_text, tokenizer, prompt_len, input_ids, attention_mask, labels, mask_region):
    logger = logging.getLogger("train_data_logger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_and_save_settings.sanity_check_log_path, mode='w')
        logger.addHandler(handler)
    
    logger.info(f"--- Example ---")
    logger.info(f"Full Text:\n{prompt_text}\n")
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    log_lines = [f"{'Index':>5} | {'Token':<25} | {'Token ID':>8} | {'Label':>8}"]
    log_lines.append("-" * 55)
    for j in range(len(input_ids)):
        log_lines.append(f"{j:>5} | {tokens[j]:<25} | {input_ids[j]:>8} | {labels[j]:>8}")
    log_lines.append("-" * 55)
    log_lines.append(f"Prompt lines: {prompt_len}")
    log_lines.append("-" * 55)
    log_lines.append(f"Attention mask 1: {attention_mask.index(1)} Question mask: {prompt_len} Total: {mask_region}")
    
    logger.info("\n".join(log_lines) + "\n")



def preprocess_function_default(dataset_settings, test_settings, log_and_save_settings, rank, examples, tokenizer, split, fiveshot_prompt_dict: dict, use_output: bool = True):
    eos_token, pad_token_id = tokenizer.eos_token, tokenizer.pad_token_id

    num_examples, prompts, categories, ground_truths, golden_examples = _get_data_fields(examples, use_output)

    batch_input_ids, batch_attention_mask, batch_labels, prompt_texts = [], [], [], []

    for i in range(num_examples):
        if split == 'train':
            current_max_length = dataset_settings.max_train_length
            prompt_text = _get_train_prompt_text(prompts[i], categories[i], golden_examples[i], eos_token, use_output)
        else:
            current_max_length = dataset_settings.max_test_length
            if test_settings.is_zeroshot:
                prompt_text = f"{QUESTION_HEADER} {prompt}\n{ANSEWR_HEADER}"
            else:
                if use_output and categories[i] in OUTPUT_EXIST:
                    template = fiveshot_prompt_dict[categories[i]]["output_format"]
                else:
                    template = fiveshot_prompt_dict[categories[i]]["answer_format"]
                prompt_text = _get_test_with_five_shot_prompt_text(template, prompts[i])

        tokenized_full = tokenizer(
            prompt_text, 
            padding="max_length", 
            truncation=True,
            max_length=current_max_length,
            return_tensors=None
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        if split == "train":
            prompt_len, mask_region, labels = _mask_input_prompt(prompts[i], tokenizer, input_ids, attention_mask)
        else:
            labels = [-100] * len(input_ids)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        prompt_texts.append(prompt_text)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
        "prompt_text": prompt_texts,
        "ground_truth": ground_truths,
        "golden_example": golden_examples,
        "category": categories,
    }


def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    prompt_texts = [ex["prompt_text"] for ex in batch]
    ground_truths = [ex["ground_truth"] for ex in batch]
    golden_examples = [ex["golden_example"] for ex in batch]
    categories = [ex["category"] for ex in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_text": prompt_texts,
        "ground_truth": ground_truths,
        "golden_example": golden_examples,
        "category": categories
    }


def _check_data_integrity(dataset):
    global OUTPUT_EXIST, ANSWER_EXIST
    assert "input" in dataset
    for inp in dataset["input"]:
        assert isinstance(inp, list)
        assert len(inp) == 1
        
        inp_element = inp[0]
        assert isinstance(inp_element, dict)
        assert "content" in inp_element.keys()
        assert isinstance(inp_element["content"], str)
    
    assert "category" in dataset
    for i, category in enumerate(dataset["category"]):
        assert isinstance(category, str)

        if category in OUTPUT_EXIST:
            assert isinstance(dataset["output"][i], str)
    
    assert "answer" in dataset
    for answer in dataset["answer"]:
        assert isinstance(answer, str)


class DataLoader:
    def __init__(self, dataset_settings, train_settings, test_settings, log_and_save_settings, tokenizer, rank, world_size, preprocess_function=preprocess_function_default, collate_fn=collate_fn):
        self.dataset_train = load_dataset("json", data_files=dataset_settings.dataset_path, field="train")["train"]
        self.dataset_test = load_dataset("json", data_files=dataset_settings.dataset_path, field="validation")["train"] # << This is not a logical error

        with open(dataset_settings.fiveshot_prompt_path, mode="r", encoding="utf-8") as f:
            self.fiveshot_prompt_dict = json.load(f)

        self.preprocess_datasets(dataset_settings, train_settings, test_settings, log_and_save_settings, rank, preprocess_function, tokenizer)
        self._setup_samplers(rank, world_size)
        self.setup_keyword_arguments(train_settings, test_settings, collate_fn)
        self._setup_dataloaders()

    def preprocess_datasets(self, dataset_settings, train_settings, test_settings, log_and_save_settings, rank, preprocess_function, tokenizer):
        def _check_integrity_and_preprocess(examples, preprocess_function):
            _check_data_integrity(examples)
            return preprocess_function(examples)

        def _preprocess_train(examples):
            return preprocess_function(
                dataset_settings, test_settings, log_and_save_settings, 
                rank, examples, tokenizer, "train", self.fiveshot_prompt_dict
            )

        def _preprocess_test(examples):
            return preprocess_function(
                dataset_settings, test_settings, log_and_save_settings, 
                rank, examples, tokenizer, "test", self.fiveshot_prompt_dict
            )
        
        self.dataset_train = self.dataset_train.map(lambda examples: _check_integrity_and_preprocess(examples, _preprocess_train), batched=True)
        self.dataset_test = self.dataset_test.map(lambda examples: _check_integrity_and_preprocess(examples, _preprocess_test), batched=True)

    def _setup_samplers(self, rank, world_size):
        self.sampler_train = DistributedSampler(self.dataset_train, rank=rank, num_replicas=world_size, shuffle=True)
        self.sampler_test = DistributedSampler(self.dataset_test, rank=rank, num_replicas=world_size)

    def setup_keyword_arguments(self, train_settings, test_settings, collate_fn):
        self.train_kwargs = {'batch_size': train_settings.batch_settings.batch_size, 'sampler': self.sampler_train, 'collate_fn': collate_fn}
        self.test_kwargs = {'batch_size': test_settings.batch_settings.batch_size, 'sampler': self.sampler_test, 'collate_fn': collate_fn}
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': False}

        self.train_kwargs.update(cuda_kwargs)
        self.test_kwargs.update(cuda_kwargs)     

    def _setup_dataloaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, **self.train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, **self.test_kwargs)

    def filter_dataset(self, rank, world_size, filter_function):
        self.dataset_train = self.dataset_train.filter(filter_function, batched=True)

        self.sampler_train = DistributedSampler(self.dataset_train, rank=rank, num_replicas=world_size, shuffle=True)

        self.train_kwargs.update({"sampler": self.sampler_train})

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, **self.train_kwargs)

    @property
    def train_size(self):
        return len(self.train_loader)
    
    @property
    def test_size(self):
        return len(self.test_loader)

    def get_train_progress_bar(self, rank, epoch):
        return tqdm(
            enumerate(self.train_loader),
            total=self.train_size,
            desc=f"Epoch {epoch}",
            position=rank,
            leave=False,
            disable=(rank != 0),  # Only show progress bar on rank 0
        )
    
    def get_test_progress_bar(self, rank, epoch):
        return tqdm(
            enumerate(self.test_loader),
            total=self.test_size,
            desc=f"Eval {epoch:.2f}",
            position=rank,
            leave=False,
            disable=(rank != 0),
        )
