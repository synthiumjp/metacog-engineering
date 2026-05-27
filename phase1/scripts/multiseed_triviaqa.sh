#!/bin/bash
# ===========================================================================
# Multi-Seed TriviaQA Replication
# ---------------------------------------------------------------------------
# Trains N seeds for Llama 8B (gentle-lr), Mistral 7B (gentlest), Qwen 7B
# (primary). Resume-safe: skips existing adapters.
#
# Prerequisites:
#   - .venv_metacog activated
#   - Models available on local/network storage
#   - Source configs exist (from the original single-seed training)
#
# Usage:
#   cd ~/jpwork/metacog-engineering/phase1/scripts
#   bash multiseed_triviaqa.sh
#
# Runtime: ~1 hour per seed per model on M3 Ultra (~15 hours total)
# ===========================================================================
set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: activate .venv_metacog first"
    exit 1
fi

SCRIPTS_DIR=~/jpwork/metacog-engineering/phase1/scripts
RESULTS_BASE=~/jpwork/metacog-engineering/phase1/results_raw
SEEDS="123 456 789 1234 5678"

cd "$SCRIPTS_DIR"

# ---------------------------------------------------------------------------
# Model configs (bash 3.2 compatible — no associative arrays)
# ---------------------------------------------------------------------------
MODEL_KEYS="llama8b mistral7b qwen7b"

get_model_config() {
    case "$1" in
        llama8b)
            MODEL_PATH="$HOME/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16"
            FINETUNE_BASE="$RESULTS_BASE/finetune/Meta-Llama-3.1-8B-Instruct-bf16"
            SOURCE_CONFIG_NAME="ablation_gentle_lr"
            LABEL_PREFIX="llama8b_gentle_lr"
            MODEL_NAME="Meta-Llama-3.1-8B-Instruct-bf16"
            ;;
        mistral7b)
            MODEL_PATH="$HOME/mnt/models-lan/foresight/synthesis-archive/Mistral-7B-Instruct-v0.3"
            FINETUNE_BASE="$RESULTS_BASE/finetune/Mistral-7B-Instruct-v0.3"
            SOURCE_CONFIG_NAME="ablation_gentlest"
            LABEL_PREFIX="mistral7b_gentlest"
            MODEL_NAME="Mistral-7B-Instruct-v0.3"
            ;;
        qwen7b)
            MODEL_PATH="$HOME/mnt/models-lan/foresight/synthesis-archive/Qwen2.5-7B-Instruct-bf16"
            FINETUNE_BASE="$RESULTS_BASE/finetune/Qwen2.5-7B-Instruct-bf16"
            SOURCE_CONFIG_NAME="probe_target"
            LABEL_PREFIX="qwen7b_primary"
            MODEL_NAME="Qwen2.5-7B-Instruct-bf16"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for model_key in $MODEL_KEYS; do
    get_model_config "$model_key"
    SOURCE_CONFIG="$FINETUNE_BASE/$SOURCE_CONFIG_NAME/config.yaml"

    echo ""
    echo "================================================================"
    echo "MODEL: $model_key ($MODEL_NAME)"
    echo "================================================================"

    if [ ! -f "$SOURCE_CONFIG" ]; then
        echo "ERROR: Config not found: $SOURCE_CONFIG"
        continue
    fi

    for seed in $SEEDS; do
        SEED_DIR="$FINETUNE_BASE/multiseed_${SOURCE_CONFIG_NAME}_seed${seed}"
        ADAPTER_DIR="$SEED_DIR/adapters"

        echo ""
        echo "--- Seed $seed ---"

        # Create seed-specific config
        mkdir -p "$SEED_DIR"
        if [ ! -f "$SEED_DIR/config.yaml" ]; then
            cp "$SOURCE_CONFIG" "$SEED_DIR/config.yaml"
            # Fix data path to use RESULTS_BASE (handles symlink/move issues)
            python3 -c "
import yaml
with open('$SEED_DIR/config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = $seed
cfg['adapter_path'] = '$ADAPTER_DIR'
# Ensure data path points to the correct location
import os
data_path = cfg.get('data', '')
if data_path and not os.path.exists(data_path):
    # Try the RESULTS_BASE version
    alt = data_path.replace('/jpwork/results/', '/jpwork/metacog-engineering/phase1/results_raw/')
    if os.path.exists(alt):
        cfg['data'] = alt
        print(f'  Fixed data path: {alt}')
with open('$SEED_DIR/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f'  Created config (seed=$seed)')
"
        else
            echo "  Config exists, skipping"
        fi

        # Train (skip if adapter exists)
        if [ -f "$ADAPTER_DIR/adapters.safetensors" ]; then
            echo "  Adapter exists, skipping training"
        else
            echo "  Training..."
            python3 -m mlx_lm lora --config "$SEED_DIR/config.yaml"
        fi

        # Eval (skip if responses exist)
        EVAL_JSON="$RESULTS_BASE/step4/ablation_multiseed_${LABEL_PREFIX}_seed${seed}_responses.json"
        if [ -f "$EVAL_JSON" ]; then
            echo "  Eval exists, skipping"
        else
            echo "  Evaluating..."
            python3 eval_ablation.py \
                --model-path "$MODEL_PATH" \
                --model-name "$MODEL_NAME" \
                --adapter-path "$ADAPTER_DIR" \
                --label "multiseed_${LABEL_PREFIX}_seed${seed}"
        fi
    done
done

echo ""
echo "================================================================"
echo "TRAINING + EVAL COMPLETE"
echo "Run rescore_multiseed.py for flex-matched AUROC2 values."
echo "================================================================"
