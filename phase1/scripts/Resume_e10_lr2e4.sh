#!/usr/bin/env bash
# resume_e10_lr2e4.sh — Resume 3 missing seeds from lr=2e-4 E10
# Seeds 3456, 7890, 2468 crashed/didn't start (PC restart)
# Also checks seed 5678 (completed but not inspected)
#
# Usage: bash resume_e10_lr2e4.sh
# Run from: ~/jpwork/metacog-engineering/phase1/

set -e

MODEL_PATH="/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
BASE_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/finetune/gemma-3-12b-it/brier_e2e_gsm8k"
RESULTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/domain_gen/e10"
SCRIPTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/scripts"

SEEDS=(3456 7890 2468)

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Resume E10 lr=2e-4: seeds ${SEEDS[*]}"
echo "=========================================="
echo "Started: $(date)"
echo ""

# ── Check seed 5678 first ──
if [ -f "${RESULTS_DIR}/e2e_ce_binary_seed5678.json" ]; then
    echo "Seed 5678 result exists:"
    python3 -c "
import json
with open('${RESULTS_DIR}/e2e_ce_binary_seed5678.json') as f:
    d = json.load(f)
print(f\"  AUROC2={d.get('auroc2',0):.3f}  Acc={d.get('accuracy',0):.3f}  Conf={d.get('conf_mean',0):.1f}\")
" 2>/dev/null || echo "  (parse error)"
elif [ -f "${BASE_DIR}/adapter_e2e_ce_seed5678/adapters.safetensors" ]; then
    echo "Seed 5678: adapter exists but no result — needs eval"
    echo "  Adding to queue..."
    SEEDS=(5678 "${SEEDS[@]}")
else
    echo "Seed 5678: no adapter found"
fi
echo ""

for SEED in "${SEEDS[@]}"; do
    ADAPTER_DIR="${BASE_DIR}/adapter_e2e_ce_seed${SEED}"
    CONFIG="${BASE_DIR}/config_e2e_ce_seed${SEED}.yaml"
    OUTPUT="${RESULTS_DIR}/e2e_ce_binary_seed${SEED}.json"

    if [ -f "$OUTPUT" ]; then
        echo "  Seed ${SEED}: result exists, skipping"
        continue
    fi

    echo "━━━ Seed ${SEED} — $(date '+%H:%M') ━━━"

    # Train if needed
    if [ -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        echo "  Adapter exists, skipping training"
    else
        echo "  Creating config and training..."
        cat > "$CONFIG" << EOF
model: ${MODEL_PATH}
data: ${BASE_DIR}
train: true
fine_tune_type: lora
lora_parameters:
  rank: 16
  scale: 2.0
  dropout: 0.05
  keys: ["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"]
iters: 846
learning_rate: 2e-4
lr_schedule:
  name: cosine_decay
  arguments: [2e-4, 846]
batch_size: 1
grad_accumulation_steps: 16
max_seq_length: 1024
seed: ${SEED}
mask_prompt: true
adapter_path: ${ADAPTER_DIR}
EOF

        python3 -m mlx_lm lora --config "$CONFIG" 2>&1 | tail -3
        echo "  Training complete"
    fi

    # Eval
    echo "  Evaluating... ($(date '+%H:%M'))"
    python3 "${SCRIPTS_DIR}/eval_e2e.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$ADAPTER_DIR" \
        --benchmark gsm8k \
        --coarse \
        --output "$OUTPUT"
    echo "  Eval complete"
    echo ""
done

echo "=========================================="
echo "Resume complete. $(date)"
echo "=========================================="
