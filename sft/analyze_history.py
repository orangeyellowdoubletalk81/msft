import os
import json
import argparse
from typing import Dict, List, Tuple, Optional


def _sorted_epoch_keys(history: Dict):
    return sorted(history.keys(), key=lambda k: float(k))


def _detect_metric_key(history: Dict) -> str:
    first_data = history[next(iter(history.keys()))]

    if 'overall_accuracy' in first_data:
        print(f"INFO: Detected greedy evaluation mode (metric: overall_accuracy)")
        return 'overall_accuracy'

    if 'mixed' in first_data:
        print(f"INFO: Using legacy 'mixed' key as overall metric")
        return 'mixed'

    raise ValueError(f"Cannot detect metric key. Available keys: {list(first_data.keys())}")


def _write_stage_summary(
    log_dir: str,
    stage: int,
    dropped_categories: List[str],
    load_epoch: float,
    excluded_categories: List[str],
    peak_overall_epoch: float,
    peak_overall_acc: float,
    global_offset: float,
    metric_key: str = 'overall_accuracy',
):
    out_path = os.path.join(log_dir, "stage_summary.json")
    global_epoch_pointer = float(global_offset) + float(load_epoch)

    payload = {
        "stage": int(stage),
        "dropped_categories": dropped_categories,
        "load_epoch": round(float(load_epoch), 2),
        "global_offset": round(float(global_offset), 2),
        "global_epoch_pointer": round(float(global_epoch_pointer), 2),
        "excluded_categories": ",".join(excluded_categories),
        "peak_overall_epoch": round(float(peak_overall_epoch), 2),
        "peak_overall_acc": float(peak_overall_acc),
        "metric_key": metric_key,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"INFO: Wrote stage summary to {out_path}")


def analyze_and_prune(log_dir: str, excluded_categories: list, stage: int,
                      use_overall_cutoff: bool = False, global_offset: float = 0.0,
                      drop_count: int = 1):

    history_file = os.path.join(log_dir, "accuracy_log.json")
    ckpt_dir = os.path.abspath(os.path.join(log_dir, "../checkpoints"))

    if not os.path.exists(history_file):
        print(f"ERROR: History file not found at {history_file}")
        exit(1)

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    if not history:
        print("ERROR: History file is empty.")
        exit(1)

    try:
        metric_key = _detect_metric_key(history)
    except ValueError as e:
        print(f"ERROR: {e}")
        exit(1)

    epoch_keys = _sorted_epoch_keys(history)

    # Find epoch with peak overall metric
    peak_overall_epoch = float(epoch_keys[0])
    max_overall_acc = float("-inf")

    for ek in epoch_keys:
        data = history[ek]
        current_metric = float(data.get(metric_key, 0.0))
        if current_metric > max_overall_acc:
            max_overall_acc = current_metric
            peak_overall_epoch = float(ek)

    print(f"\nINFO: Peak {metric_key} ({max_overall_acc:.4f}) found at epoch {peak_overall_epoch:.2f}")

    if use_overall_cutoff:
        epoch_keys_used = [ek for ek in epoch_keys if float(ek) <= peak_overall_epoch]
        if not epoch_keys_used:
            epoch_keys_used = epoch_keys[:]
        print(f"INFO: Applying overall cutoff. analyzing epochs <= {peak_overall_epoch}")
    else:
        epoch_keys_used = epoch_keys[:]

    all_categories = set()
    for ek in epoch_keys_used:
        all_categories |= set(history[ek].get('per_category', {}).keys())
    all_categories = sorted(all_categories)

    category_peaks = []

    for category in all_categories:
        if category in excluded_categories:
            continue

        series: List[Tuple[float, float]] = []
        for ek in epoch_keys_used:
            per_cat = history[ek].get('per_category', {})
            if category in per_cat:
                series.append((float(ek), float(per_cat[category])))

        if not series:
            continue

        best_epoch, best_score = max(series, key=lambda x: x[1])
        category_peaks.append({
            'category': category,
            'epoch': float(best_epoch),
            'peak_acc': float(best_score),
        })

    if not category_peaks:
        dropped_categories = []
        load_epoch = peak_overall_epoch
        print("OUTPUT:ANALYSIS_COMPLETE")
    else:
        last_epoch = float(epoch_keys_used[-1])

        min_peak_epoch = min(cp['epoch'] for cp in category_peaks)

            if min_peak_epoch >= last_epoch:
                dropped_categories = []
                load_epoch = last_epoch
                print(f"INFO: All categories peaked at the final epoch ({last_epoch:.2f}). No overfitting detected.")
                print(f"OUTPUT:NO_DROP,LOAD_CHECKPOINT_EPOCH={load_epoch:.2f}")
                _write_stage_summary(
                    log_dir=log_dir, stage=stage, dropped_categories=dropped_categories,
                    load_epoch=load_epoch, excluded_categories=excluded_categories,
                    peak_overall_epoch=peak_overall_epoch, peak_overall_acc=max_overall_acc,
                    global_offset=global_offset, metric_key=metric_key,
                )
                return

            earliest_candidates = sorted(
                [cp for cp in category_peaks if cp['epoch'] == min_peak_epoch],
                key=lambda x: x['category']
            )
            candidates_to_drop = earliest_candidates[:drop_count]

        dropped_categories = [cp['category'] for cp in candidates_to_drop]
        target_epoch = candidates_to_drop[0]['epoch']

        for cp in candidates_to_drop:
            print(f"INFO: Candidate to drop: '{cp['category']}' (Peak at Epoch {cp['epoch']:.2f}, Score {cp['peak_acc']:.4f})")

        ckpt_filename = f"hf-model-{target_epoch:.2f}"
        candidate_ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

        if os.path.exists(candidate_ckpt_path):
            load_epoch = target_epoch
            print(f"INFO: Found checkpoint '{ckpt_filename}'.")
        else:
            load_epoch = peak_overall_epoch
            print(f"WARN: Checkpoint '{ckpt_filename}' not found. Using fallback '{peak_overall_epoch:.2f}'.")

        drop_str = "|".join(dropped_categories)
        print(f"OUTPUT:DROP_CATEGORY={drop_str},LOAD_CHECKPOINT_EPOCH={load_epoch:.2f}")

    _write_stage_summary(
        log_dir=log_dir,
        stage=stage,
        dropped_categories=dropped_categories,
        load_epoch=load_epoch,
        excluded_categories=excluded_categories,
        peak_overall_epoch=peak_overall_epoch,
        peak_overall_acc=max_overall_acc,
        global_offset=global_offset,
        metric_key=metric_key,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment history and determine the next step.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to experiment log directory")
    parser.add_argument("--excluded-categories", type=str, default="", help="Already excluded categories")
    parser.add_argument("--stage", type=int, required=True, help="current stage number")
    parser.add_argument("--global-offset", type=float, required=True, help="cumulative global offset")
    parser.add_argument("--use-overall-cutoff", action="store_true", help="limit analysis to epochs <= overall peak epoch")
    parser.add_argument("--drop-count", type=int, default=1, help="maximum number of categories to drop per stage (default: 1)")

    args = parser.parse_args()
    excluded_list = [cat.strip() for cat in args.excluded_categories.split(',') if cat.strip()]

    analyze_and_prune(
        log_dir=args.log_dir,
        excluded_categories=excluded_list,
        stage=args.stage,
        use_overall_cutoff=args.use_overall_cutoff,
        global_offset=args.global_offset,
        drop_count=args.drop_count,
    )
