#!/usr/bin/env bash
# e10_e2e_ce_gentle.sh — 10-seed E10 at lr=5e-5, 2538 iters (9 epochs)
# Trains all 10 seeds, then evals all (text), then evals all (logit)
#
# Usage: bash e10_e2e_ce_gentle.sh
# Expected: ~100 min training total, ~15 hrs text eval, ~15 hrs logit eval
# Run from: ~/jpwork/metacog-engineering/phase1/

set -e

MODEL_PATH="/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
BASE_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/finetune/gemma-3-12b-it/brier_e2e_gsm8k"
RESULTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/domain_gen/e10_gentle"
SCRIPTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/scripts"

SEEDS=(42 123 456 789 1234 5678 9012 3456 7890 2468)

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "E10 Gentle LR: End-to-End CE Binary GSM8K"
echo "lr=5e-5, iters=2538, rank=16"
echo "=========================================="
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo ""

# ── Phase 1: Training (10 × ~10 min) ──
echo "Phase 1: Training all 10 seeds..."
for SEED in "${SEEDS[@]}"; do
    ADAPTER_DIR="${BASE_DIR}/adapter_e2e_ce_gentle_seed${SEED}"
    CONFIG="${BASE_DIR}/config_e2e_ce_gentle_seed${SEED}.yaml"

    if [ -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        echo "  Seed ${SEED}: adapter exists, skipping training"
        continue
    fi

    echo "  Seed ${SEED}: creating config..."
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
iters: 2538
learning_rate: 5e-5
lr_schedule:
  name: cosine_decay
  arguments: [5e-5, 2538]
batch_size: 1
grad_accumulation_steps: 16
max_seq_length: 1024
seed: ${SEED}
mask_prompt: true
adapter_path: ${ADAPTER_DIR}
EOF

    echo "  Seed ${SEED}: training... ($(date '+%H:%M'))"
    python3 -m mlx_lm lora --config "$CONFIG" 2>&1 | tail -3
    echo "  Seed ${SEED}: training complete"
    echo ""
done

echo "Phase 1 complete."
echo ""

# ── Phase 2: Text eval (10 × ~90 min) ──
echo "Phase 2: Text eval all 10 seeds..."
for SEED in "${SEEDS[@]}"; do
    ADAPTER_DIR="${BASE_DIR}/adapter_e2e_ce_gentle_seed${SEED}"
    OUTPUT="${RESULTS_DIR}/e2e_ce_gentle_seed${SEED}.json"

    if [ -f "$OUTPUT" ]; then
        echo "  Seed ${SEED}: text result exists, skipping"
        continue
    fi

    if [ ! -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        echo "  Seed ${SEED}: no adapter found, skipping"
        continue
    fi

    echo "  Seed ${SEED}: text eval... ($(date '+%H:%M'))"
    python3 "${SCRIPTS_DIR}/eval_e2e.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$ADAPTER_DIR" \
        --benchmark gsm8k \
        --coarse \
        --output "$OUTPUT"
    echo "  Seed ${SEED}: text eval complete"
    echo ""
done

echo "Phase 2 complete."
echo ""

# ── Phase 3: Logit eval (10 × ~90 min) ──
echo "Phase 3: Logit eval all 10 seeds..."
for SEED in "${SEEDS[@]}"; do
    ADAPTER_DIR="${BASE_DIR}/adapter_e2e_ce_gentle_seed${SEED}"
    OUTPUT="${RESULTS_DIR}/e2e_ce_gentle_logit_seed${SEED}.json"

    if [ -f "$OUTPUT" ]; then
        echo "  Seed ${SEED}: logit result exists, skipping"
        continue
    fi

    if [ ! -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        echo "  Seed ${SEED}: no adapter found, skipping"
        continue
    fi

    echo "  Seed ${SEED}: logit eval... ($(date '+%H:%M'))"
    python3 "${SCRIPTS_DIR}/eval_e2e_logits.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$ADAPTER_DIR" \
        --benchmark gsm8k \
        --output "$OUTPUT"
    echo "  Seed ${SEED}: logit eval complete"
    echo ""
done

echo "Phase 3 complete."
echo ""

# ── Summary ──
echo "=========================================="
echo "E10 Gentle Results Summary"
echo "=========================================="
python3 -c "
import json, glob, numpy as np

text_files = sorted(glob.glob('${RESULTS_DIR}/e2e_ce_gentle_seed*.json'))
text_files = [f for f in text_files if 'logit' not in f]
logit_files = sorted(glob.glob('${RESULTS_DIR}/e2e_ce_gentle_logit_seed*.json'))

print('Seed  | Text AUROC2 | Logit AUROC2 | Acc   | Conf mean')
print('------|-------------|--------------|-------|----------')

text_aurocs = []
logit_aurocs = []
for f in text_files:
    seed = f.split('seed')[1].split('.')[0]
    with open(f) as fh:
        d = json.load(fh)
    ta = d.get('auroc2', 0)
    acc = d.get('accuracy', 0)
    cm = d.get('conf_mean', 0)
    text_aurocs.append(ta)

    # Find matching logit file
    lf = f.replace('gentle_seed', 'gentle_logit_seed')
    la = 'N/A'
    if lf in logit_files or True:
        try:
            with open(lf) as fh2:
                ld = json.load(fh2)
            la = f'{ld.get(\"auroc2\", 0):.3f}'
            logit_aurocs.append(ld.get('auroc2', 0))
        except:
            pass

    print(f'{seed:5s} | {ta:.3f}       | {la:12s} | {acc:.3f} | {cm:.1f}')

text_aurocs = np.array(text_aurocs)
print(f'')
print(f'Text  AUROC2: mean={text_aurocs.mean():.3f} +/- {text_aurocs.std():.3f}  (n={len(text_aurocs)})')
if logit_aurocs:
    logit_aurocs = np.array(logit_aurocs)
    print(f'Logit AUROC2: mean={logit_aurocs.mean():.3f} +/- {logit_aurocs.std():.3f}  (n={len(logit_aurocs)})')
print(f'Accuracy: mean={text_aurocs.mean():.3f}' if len(text_aurocs) else '')
"
echo ""
echo "Done. $(date)"
