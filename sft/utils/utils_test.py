import torch
import torch.distributed as dist

import gc
import re

from utils.utils import is_qwen


def _sanity_check(rank, tokenizer, do_print: bool):
    if rank != 0:
        return
    
    if do_print:
        print("--- DEBUG: Tokenization Check ---")
    text_to_check = "The answer is C.<|endoftext|>"
    token_ids = tokenizer.encode(text_to_check)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    if do_print:
        print(f"--- Tokenization Check ---")
        print(f"Original Text: {text_to_check}")
        print(f"Tokens: {tokens}")
        print(f"Is it one token? -> {len(tokens) > 0 and tokens[-1] == tokenizer.eos_token}")
        print("--------------------------")

    assert len(tokens) > 0 and tokens[-1] == tokenizer.eos_token


def _init_test_model(model_settings, model, rank):
    test_model = model_settings.get_ddp_copy_model(model, rank)
    test_model.eval()
    return test_model


def generate_greedy(model_name, tokenizer, generate_fn, input_ids, attention_mask, gen_len):
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    if is_qwen(model_name):
        gen_kwargs.update(
            dict(
                max_new_tokens=gen_len, do_sample=False,
                temperature=None,
                top_k=None, top_p=None
            )
        )
    else:
        gen_kwargs.update(
            dict(
                max_length=input_ids.size(1) + gen_len, do_sample=False,
                temperature=None,
                top_k=None, top_p=None
            )
        )
    out = generate_fn(**gen_kwargs)
    return out


def _parse_data(rank, data):
    assert "input_ids" in data
    assert "attention_mask" in data
    assert "prompt_text" in data
    assert "ground_truth" in data
    assert "golden_example" in data
    assert "category" in data

    input_ids = data["input_ids"].to(rank)
    attention_mask = data["attention_mask"].to(rank)
    prompt_text = data["prompt_text"]
    ground_truth = data["ground_truth"]
    golden_example = data["golden_example"]
    category = data["category"]

    return input_ids, attention_mask, prompt_text, ground_truth, golden_example, category

def _gather_local_logs(world_size, local_logs):
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_logs)

    merged = []
    for part in gathered:
        if part:
            merged.extend(part)
    
    merged.sort(key=lambda x: (x["rank"], x["local_counter"]))
    return merged

def _parse_final_answer(text: str):
    match = re.search(r"####\s*(.*)", str(text)) 
    raw_answer = match.group(1).strip() if match else str(text).strip()
    cleaned_answer = raw_answer.rstrip('.')
    try:
        return float(cleaned_answer)
    except (ValueError, TypeError):
        return cleaned_answer.lower()

def _run_evaluation_loop(rank, test_model, dataloader, model_settings, test_settings, tokenizer, log_epoch):
    generate_fn = test_model.module.generate if hasattr(test_model, 'module') else test_model.generate

    local_logs = []
    local_eval_results = []
    local_counter = 0

    eval_progress_bar = dataloader.get_test_progress_bar(rank, log_epoch)

    with torch.no_grad():
        for batch_i, data in eval_progress_bar:
            input_ids, attention_mask, prompt_text, ground_truth, golden_example, category = _parse_data(rank, data)

            sequences = generate_greedy(model_settings.model_name, tokenizer, generate_fn, input_ids, attention_mask, test_settings.gen_length)
            gen_tokens = sequences[:, input_ids.shape[-1]:]
            prediction = tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)
                
            for i, (inp, pred, gro, gold, cat) in enumerate(zip(prompt_text, prediction, ground_truth, golden_example, category)):
                end_token_index = pred.find("<|endoftext|>")
                if end_token_index != -1:
                    pred = pred[:end_token_index].strip()

                parsed_prediction = _parse_final_answer(pred)
                parsed_ground_truth = _parse_final_answer(gro)
                verdict = (parsed_prediction is not None) and (parsed_prediction == parsed_ground_truth)
                
                local_eval_results.append({"category": cat, "verdict": verdict, "rank": rank, "local_counter": local_counter})
                local_logs.append({
                    "rank": rank, "local_counter": local_counter, "input": inp,
                    "prediction": pred, "ground_truth": gro,
                    "golden_example": gold, "category": cat
                })
                local_counter += 1

    return local_logs, local_eval_results
    
def _calculate_metrics(gathered_eval_results):
    category_counts = {}
    for item in gathered_eval_results:
        cat = item["category"]
        if cat not in category_counts:
            category_counts[cat] = {"total": 0, "correct": 0}
        
        category_counts[cat]["total"] += 1
        if item["verdict"]:
            category_counts[cat]["correct"] += 1
    
    total_count = sum(v['total'] for v in category_counts.values())
    correct_count = sum(v['correct'] for v in category_counts.values())
    
    per_category_accuracy = {
        cat: (counts['correct'] / counts['total']) if counts['total'] > 0 else 0
        for cat, counts in category_counts.items()
    }

    final_metrics = {
        'overall_accuracy': (correct_count / total_count) if total_count > 0 else 0,
        'per_category': per_category_accuracy
    }
    return final_metrics

def output_logs(pred_file, logs, log_epoch):
    with open(pred_file, 'w', encoding='utf-8') as f:
        f.write(f"Epoch: {log_epoch:.2f}\n\n")

        for n, item in enumerate(logs, start=1):
            f.write(f"Example {n}\n")
            f.write(f"Category: {item['category']}\n")
            f.write(f"Input:\n{item['input']}\n")
            f.write(f"Prediction:\n{item['prediction']}\n")
            f.write(f"Ground Truth:\n{item['ground_truth']}\n")
            f.write(f"Golden Example:\n{item['golden_example']}")
            f.write("\n" + "-" * 50 + "\n\n")


def test(test_settings, model_settings, log_and_save_settings, model, rank, world_size, dataloader, tokenizer, cur_epoch, log_epoch):
    _sanity_check(rank, tokenizer, True)

    test_model = _init_test_model(model_settings, model, rank)

    local_logs, local_eval_results = _run_evaluation_loop(
         rank, test_model, dataloader, model_settings, test_settings, tokenizer, log_epoch
    )

    gathered_logs = _gather_local_logs(world_size, local_logs)
    gathered_eval_results = _gather_local_logs(world_size, local_eval_results)

    final_metrics = None
    if rank == 0:
        pred_file = log_and_save_settings.get_prediction_log_path(log_epoch)
        output_logs(pred_file, gathered_logs, log_epoch)
        
        final_metrics = _calculate_metrics(gathered_eval_results)

    del test_model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return final_metrics
