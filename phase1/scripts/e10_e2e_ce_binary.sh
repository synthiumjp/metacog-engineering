#!/usr/bin/env bash
# e10_e2e_ce_binary.sh — Run 10-seed validation of end-to-end CE binary PT-CSFT on GSM8K
#
# Usage:
#   bash e10_e2e_ce_binary.sh
#
# Runs training (10 seeds × ~10 min = ~100 min) then eval (10 × ~90 min = ~15 hrs).
# Run overnight. Results written to results_raw/domain_gen/e10/

set -e

MODEL_PATH="/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
BASE_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/finetune/gemma-3-12b-it/brier_e2e_gsm8k"
RESULTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/domain_gen/e10"
SCRIPTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/scripts"

SEEDS=(42 123 456 789 1234 5678 9012 3456 7890 2468)

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "E10 Validation: End-to-End CE Binary GSM8K"
echo "=========================================="
echo "Seeds: ${SEEDS[*]}"
echo ""

# ── Phase 1: Training (10 × ~10 min) ──
echo "Phase 1: Training all 10 seeds..."
for SEED in "${SEEDS[@]}"; do
    ADAPTER_DIR="${BASE_DIR}/adapter_e2e_ce_seed${SEED}"
    CONFIG="${BASE_DIR}/config_e2e_ce_seed${SEED}.yaml"

    # Skip if adapter already exists
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

    echo "  Seed ${SEED}: training..."
    python3 -m mlx_lm lora --config "$CONFIG" 2>&1 | tail -3
    echo "  Seed ${SEED}: training complete"
    echo ""
done

echo "Phase 1 complete."
echo ""

# ── Phase 2: Evaluation (10 × ~90 min) ──
echo "Phase 2: Evaluating all 10 seeds..."
for SEED in "${SEEDS[@]}"; do
    ADAPTER_DIR="${BASE_DIR}/adapter_e2e_ce_seed${SEED}"
    OUTPUT="${RESULTS_DIR}/e2e_ce_binary_seed${SEED}.json"

    # Skip if result already exists
    if [ -f "$OUTPUT" ]; then
        echo "  Seed ${SEED}: result exists, skipping eval"
        continue
    fi

    if [ ! -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        echo "  Seed ${SEED}: no adapter found, skipping"
        continue
    fi

    echo "  Seed ${SEED}: evaluating... ($(date '+%H:%M'))"
    python3 "${SCRIPTS_DIR}/eval_e2e.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$ADAPTER_DIR" \
        --benchmark gsm8k \
        --coarse \
        --output "$OUTPUT"
    echo "  Seed ${SEED}: eval complete"
    echo ""
done

echo "Phase 2 complete."
echo ""

# ── Phase 3: Aggregate results ──
echo "=========================================="
echo "E10 Results Summary"
echo "=========================================="
python3 -c "
import json, glob, numpy as np

files = sorted(glob.glob('${RESULTS_DIR}/e2e_ce_binary_seed*.json'))
if not files:
    print('No results found yet.')
    exit()

aurocs = []
accs = []
conf_means = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    seed = f.split('seed')[1].split('.')[0]
    a2 = d.get('auroc2', 0)
    acc = d.get('accuracy', 0)
    cm = d.get('conf_mean', 0)
    aurocs.append(a2)
    accs.append(acc)
    conf_means.append(cm)
    print(f'  Seed {seed}: AUROC₂={a2:.3f}  Acc={acc:.3f}  Conf={cm:.1f}')

aurocs = np.array(aurocs)
accs = np.array(accs)
print(f'')
print(f'  AUROC₂: mean={aurocs.mean():.3f} ± {aurocs.std():.3f}  (range: {aurocs.min():.3f}-{aurocs.max():.3f})')
print(f'  Accuracy: mean={accs.mean():.3f} ± {accs.std():.3f}')
print(f'  N seeds: {len(aurocs)}')

# 95% CI via bootstrap
if len(aurocs) >= 3:
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(aurocs, size=len(aurocs), replace=True).mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f'  AUROC₂ 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')
"
echo ""
echo "Done."
