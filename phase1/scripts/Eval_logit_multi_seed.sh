#!/usr/bin/env bash
# eval_logit_multi_seed.sh — Logit eval on existing lr=2e-4 E10 adapters
# Tests whether logit signal is stable across seeds, including collapsed ones
#
# Usage: bash eval_logit_multi_seed.sh
# Expected: ~90 min per seed
# Run from: ~/jpwork/metacog-engineering/phase1/

set -e

MODEL_PATH="/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
ADAPTER_BASE="/Users/chrismarmo/jpwork/metacog-engineering/phase1/finetune/gemma-3-12b-it/brier_e2e_gsm8k"
RESULTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/domain_gen/e10"
SCRIPTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/scripts"

# Priority order: collapsed (123), moderate (456, 9012), good (42, 789), partial ceiling (1234)
SEEDS=(123 456 789 42 1234 9012)

echo "=========================================="
echo "Logit Eval on lr=2e-4 E10 Adapters"
echo "=========================================="
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo ""

for SEED in "${SEEDS[@]}"; do
    ADAPTER_DIR="${ADAPTER_BASE}/adapter_e2e_ce_seed${SEED}"
    OUTPUT="${RESULTS_DIR}/e2e_ce_binary_logit_seed${SEED}.json"

    if [ -f "$OUTPUT" ]; then
        echo "  Seed ${SEED}: logit result exists, skipping"
        continue
    fi

    if [ ! -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        echo "  Seed ${SEED}: no adapter at ${ADAPTER_DIR}, skipping"
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

echo ""
echo "=========================================="
echo "Logit vs Text Summary (lr=2e-4)"
echo "=========================================="
python3 -c "
import json, numpy as np

seeds = [123, 456, 789, 42, 1234, 9012]
results_dir = '${RESULTS_DIR}'

print('Seed  | Text AUROC2 | Logit AUROC2 | Delta')
print('------|-------------|--------------|------')

for seed in seeds:
    text_f = f'{results_dir}/e2e_ce_binary_seed{seed}.json'
    logit_f = f'{results_dir}/e2e_ce_binary_logit_seed{seed}.json'
    ta = la = None
    try:
        with open(text_f) as f:
            ta = json.load(f).get('auroc2', 0)
    except:
        pass
    try:
        with open(logit_f) as f:
            la = json.load(f).get('auroc2', 0)
    except:
        pass

    ts = f'{ta:.3f}' if ta is not None else 'N/A'
    ls = f'{la:.3f}' if la is not None else 'N/A'
    ds = f'{la - ta:+.3f}' if (ta is not None and la is not None) else 'N/A'
    print(f'{seed:5d} | {ts:11s} | {ls:12s} | {ds}')
"
echo ""
echo "Done. $(date)"
