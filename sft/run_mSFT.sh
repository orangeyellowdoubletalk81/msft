#!/bin/bash

set -e

BASE_EXP_NAME="msft"
CONFIG_FILE="configs/msft-config.json"
SEED=20
MAX_STAGES=10
GPU_IDS=""
MASTER_PORT=""

TRAIN_SCRIPT="train_eval.py"
ANALYZE_SCRIPT="analyze_history.py"

DROP_COUNT=1

START_STAGE=1
RESUME_EXCLUDED_CATS=""
RESUME_CKPT_TO_LOAD=""
RESUME_GLOBAL_OFFSET=0.0
RESUME_GLOBAL_BEST_ACC=-1
RESUME_GLOBAL_BEST_CKPT=""

find_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# Remove all hf-model-* checkpoints in CKPT_DIR except load ckpt / overall peak
#   $1 = CKPT_DIR, $2 = keep_path_1 (load ckpt), $3 = keep_path_2 (overall peak)
prune_stage_checkpoints() {
    local ckpt_dir="$1"
    local keep1="$(realpath "$2" 2>/dev/null || echo "$2")"
    local keep2="$(realpath "$3" 2>/dev/null || echo "$3")"

    echo "[Prune] Cleaning up stage checkpoints in $ckpt_dir ..."
    for d in "$ckpt_dir"/hf-model-*/; do
        [ -d "$d" ] || continue
        local abs="$(realpath "$d")"
        if [ "$abs" = "$keep1" ] || [ "$abs" = "$keep2" ]; then
            echo "[Prune] Keeping: $(basename "$d")"
        else
            echo "[Prune] Removing: $(basename "$d")"
            rm -rf "$d"
        fi
    done
}

if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=$(find_free_port)
    echo "Auto-assigned MASTER_PORT: $MASTER_PORT"
fi

if [ "$START_STAGE" -gt 1 ]; then
    echo "Resuming pipeline from Stage $START_STAGE..."
    EXCLUDED_CATS="$RESUME_EXCLUDED_CATS"
    CKPT_TO_LOAD="$RESUME_CKPT_TO_LOAD"
    CURRENT_GLOBAL_OFFSET="$RESUME_GLOBAL_OFFSET"
    GLOBAL_BEST_ACC="$RESUME_GLOBAL_BEST_ACC"
    GLOBAL_BEST_CKPT="$RESUME_GLOBAL_BEST_CKPT"
else
    echo "Starting new pipeline from Stage 1..."
    EXCLUDED_CATS=""
    CKPT_TO_LOAD=""
    CURRENT_GLOBAL_OFFSET=0.0
    GLOBAL_BEST_ACC=-1
    GLOBAL_BEST_CKPT=""
fi

for STAGE in $(seq $START_STAGE $MAX_STAGES)
do
    echo ""
    echo "=================================================="
    echo "========= STAGE $STAGE (Global Offset: $CURRENT_GLOBAL_OFFSET) =========="
    echo "=================================================="

    CURRENT_EXP_NAME="${BASE_EXP_NAME}_stage${STAGE}"

    CMD="uv run $TRAIN_SCRIPT \
        --config $CONFIG_FILE \
        --seed $SEED \
        --exp-name $CURRENT_EXP_NAME \
        --exclude-cats \"$EXCLUDED_CATS\" \
        --global-best-acc $GLOBAL_BEST_ACC"

    if [ ! -z "$CKPT_TO_LOAD" ]; then
        CMD="$CMD --resume-from $CKPT_TO_LOAD"
    fi

    if [ ! -z "$GLOBAL_BEST_CKPT" ]; then
        CMD="$CMD --global-best-ckpt $GLOBAL_BEST_CKPT"
    fi

    if [ ! -z "$GPU_IDS" ]; then
        CMD="$CMD --gpus $GPU_IDS"
    fi

    if [ ! -z "$MASTER_PORT" ]; then
        CMD="$CMD --master-port $MASTER_PORT"
    fi

    echo "[Train] Executing Training for Stage $STAGE..."
    echo "   Command: $CMD"

    eval $CMD


    LOG_DIR="${CURRENT_EXP_NAME}_${SEED}/logs"
    CKPT_DIR="${CURRENT_EXP_NAME}_${SEED}/checkpoints"

    echo "[Analyze] Analyzing results from Stage $STAGE..."

    ANALYSIS_FLAGS=""
    if [ "$USE_OVERALL_CUTOFF" = "true" ]; then ANALYSIS_FLAGS="$ANALYSIS_FLAGS --use-overall-cutoff"; fi
    ANALYSIS_FLAGS="$ANALYSIS_FLAGS --drop-count $DROP_COUNT"

    analysis_raw_output=$(uv run $ANALYZE_SCRIPT \
        --log-dir "$LOG_DIR" \
        --excluded-categories "$EXCLUDED_CATS" \
        --stage "$STAGE" \
        --global-offset "$CURRENT_GLOBAL_OFFSET" \
        $ANALYSIS_FLAGS)

    echo "$analysis_raw_output"

    analysis_output=$(echo "$analysis_raw_output" | grep "OUTPUT:")
    echo "   Parsed Output: $analysis_output"

    if [[ $analysis_output == *"ANALYSIS_COMPLETE"* ]]; then
        echo "Analysis complete. No more categories to drop. Pipeline finished."
        break
    fi

    if [[ $analysis_output == *"NO_DROP"* ]]; then
        LOAD_EPOCH=$(echo $analysis_output | grep -o 'LOAD_CHECKPOINT_EPOCH=[^,]*' | cut -d'=' -f2)
        CURRENT_GLOBAL_OFFSET=$(python -c "print(f'{float($CURRENT_GLOBAL_OFFSET) + float($LOAD_EPOCH):.2f}')")

        CKPT_TO_LOAD="${CKPT_DIR}/hf-model-${LOAD_EPOCH}"

        if [ ! -d "$CKPT_TO_LOAD" ] && [ ! -d "$(pwd)/$CKPT_TO_LOAD" ]; then
            echo "WARNING: Checkpoint file $CKPT_TO_LOAD not found!"
        fi

        echo "No category peaked. Continuing with same categories."
        echo "  - Load Checkpoint: $CKPT_TO_LOAD"
        echo "  - New Global Offset: $CURRENT_GLOBAL_OFFSET"

        read PEAK_OVERALL_EPOCH PEAK_OVERALL_ACC < <(python3 -c "
import json
d = json.load(open('${LOG_DIR}/stage_summary.json'))
print(f\"{d['peak_overall_epoch']:.2f} {d['peak_overall_acc']}\")
")
        STAGE_OVERALL_CKPT="${CKPT_DIR}/hf-model-${PEAK_OVERALL_EPOCH}"
        if python3 -c "exit(0 if float('${PEAK_OVERALL_ACC}') > float('${GLOBAL_BEST_ACC}') else 1)" 2>/dev/null; then
            echo "[GlobalBest] Updated: acc=${PEAK_OVERALL_ACC} → ${STAGE_OVERALL_CKPT}"
            GLOBAL_BEST_ACC="${PEAK_OVERALL_ACC}"
            GLOBAL_BEST_CKPT="${STAGE_OVERALL_CKPT}"
            KEEP_OVERALL="${STAGE_OVERALL_CKPT}"
        else
            echo "[GlobalBest] No improvement (stage=${PEAK_OVERALL_ACC} <= best=${GLOBAL_BEST_ACC}). Stage overall peak pruned."
            KEEP_OVERALL="${CKPT_TO_LOAD}"
        fi
        prune_stage_checkpoints "$CKPT_DIR" "$CKPT_TO_LOAD" "$KEEP_OVERALL"
        continue
    fi

    DROP_CATEGORY=$(echo $analysis_output | grep -o 'DROP_CATEGORY=[^,]*' | cut -d'=' -f2)
    LOAD_EPOCH=$(echo $analysis_output | grep -o 'LOAD_CHECKPOINT_EPOCH=[^,]*' | cut -d'=' -f2)

    if [ -z "$DROP_CATEGORY" ] || [ -z "$LOAD_EPOCH" ]; then
        echo "ERROR: Failed to parse analysis output. Stopping pipeline."
        exit 1
    fi

    CURRENT_GLOBAL_OFFSET=$(python -c "print(f'{float($CURRENT_GLOBAL_OFFSET) + float($LOAD_EPOCH):.2f}')")

    CKPT_TO_LOAD="${CKPT_DIR}/hf-model-${LOAD_EPOCH}"

    if [ ! -d "$CKPT_TO_LOAD" ] && [ ! -d "$(pwd)/$CKPT_TO_LOAD" ]; then
         echo "WARNING: Checkpoint file $CKPT_TO_LOAD not found! Pipeline might fail next stage."
    fi

    DROP_CATS_COMMA=$(echo "$DROP_CATEGORY" | tr '|' ',')
    if [ -z "$EXCLUDED_CATS" ]; then
        EXCLUDED_CATS="${DROP_CATS_COMMA}"
    else
        EXCLUDED_CATS="${EXCLUDED_CATS},${DROP_CATS_COMMA}"
    fi

    read PEAK_OVERALL_EPOCH PEAK_OVERALL_ACC < <(python3 -c "
import json
d = json.load(open('${LOG_DIR}/stage_summary.json'))
print(f\"{d['peak_overall_epoch']:.2f} {d['peak_overall_acc']}\")
")
    STAGE_OVERALL_CKPT="${CKPT_DIR}/hf-model-${PEAK_OVERALL_EPOCH}"
    if python3 -c "exit(0 if float('${PEAK_OVERALL_ACC}') > float('${GLOBAL_BEST_ACC}') else 1)" 2>/dev/null; then
        echo "[GlobalBest] Updated: acc=${PEAK_OVERALL_ACC} → ${STAGE_OVERALL_CKPT}"
        GLOBAL_BEST_ACC="${PEAK_OVERALL_ACC}"
        GLOBAL_BEST_CKPT="${STAGE_OVERALL_CKPT}"
        KEEP_OVERALL="${STAGE_OVERALL_CKPT}"
    else
        echo "[GlobalBest] No improvement (stage=${PEAK_OVERALL_ACC} <= best=${GLOBAL_BEST_ACC}). Stage overall peak pruned."
        KEEP_OVERALL="${CKPT_TO_LOAD}"
    fi
    prune_stage_checkpoints "$CKPT_DIR" "$CKPT_TO_LOAD" "$KEEP_OVERALL"

    echo "Next Stage Plan:"
    echo "  - Drop Category: $DROP_CATS_COMMA"
    echo "  - Load Checkpoint: $CKPT_TO_LOAD"
    echo "  - New Global Offset: $CURRENT_GLOBAL_OFFSET"

    if [ "$STAGE" -eq "$MAX_STAGES" ]; then
        echo "Reached max stages ($MAX_STAGES). Stopping pipeline."
    fi
done

echo "Pipeline execution finished at: $(date)"
