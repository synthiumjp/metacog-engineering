#!/usr/bin/env bash
# eval_logit_triviaqa_adapters.sh — Logit eval on existing TriviaQA PT-CSFT adapters
# Tests whether the logit finding generalises beyond GSM8K to TriviaQA (fine mode 0-100)
#
# Usage: bash eval_logit_triviaqa_adapters.sh
# Run from: ~/jpwork/metacog-engineering/phase1/

set -e

MODELS_BASE="/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive"
FINETUNE_BASE="/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/finetune"
RESULTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/domain_gen/logit_triviaqa"
SCRIPTS_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1/scripts"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Logit Eval: TriviaQA PT-CSFT Adapters"
echo "=========================================="
echo "Started: $(date)"
echo ""

# Define model/adapter pairs — best configs from Phase 1
# Format: MODEL_DIR|ADAPTER_DIR|LABEL
CONFIGS=(
    "/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/Mistral-7B-Instruct-v0.3|/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/finetune/Mistral-7B-Instruct-v0.3/ablation_gentlest/adapters|mistral7b_gentlest"
    "/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it|/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/finetune/gemma-3-12b-it/probe_target/adapters|gemma12b_probe_target"
    "/Users/chrismarmo/jpwork/models/Qwen2.5-32B-Instruct-bf16|/Users/chrismarmo/jpwork/results/finetune/Qwen2.5-32B-Instruct-bf16/ablation_gentle_lr/adapters|qwen32b_gentle"
    "/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16|/Users/chrismarmo/jpwork/metacog-engineering/phase1/results_raw/finetune/Meta-Llama-3.1-8B-Instruct-bf16/ablation_gentle_lr/adapters|llama8b_gentle"
)
for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r MODEL_DIR ADAPTER_REL LABEL <<< "$CONFIG"

    MODEL_PATH="$MODEL_DIR"
    ADAPTER_PATH="$ADAPTER_REL"
    OUTPUT="$RESULTS_DIR/logit_triviaqa_${LABEL}.json"

    # Skip if done
    if [ -f "$OUTPUT" ]; then
        echo "  $LABEL: result exists, skipping"
        continue
    fi

    # Check adapter exists
    if [ ! -f "$ADAPTER_PATH/adapters.safetensors" ]; then
        echo "  $LABEL: adapter not found at $ADAPTER_PATH, skipping"
        continue
    fi

    echo "━━━ $LABEL — $(date '+%H:%M') ━━━"
    echo "  Model: $MODEL_DIR"
    echo "  Adapter: $ADAPTER_REL"

    python3 "$SCRIPTS_DIR/eval_logit_general.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$ADAPTER_PATH" \
        --benchmark triviaqa \
        --mode fine \
        --output "$OUTPUT"

    echo "  $LABEL complete"
    echo ""
done

# ── Summary ──
echo "=========================================="
echo "Logit TriviaQA Summary"
echo "=========================================="
python3 -c "
import json, glob
files = sorted(glob.glob('${RESULTS_DIR}/logit_triviaqa_*.json'))
files = [f for f in files if 'responses' not in f]
print(f'{'Label':30s} | Text AUROC2 | Logit AUROC2 | Delta')
print('-' * 75)
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    label = f.split('logit_triviaqa_')[1].split('.')[0]
    ta = d.get('text_auroc2', 0)
    la = d.get('logit_auroc2', 0)
    delta = la - ta
    print(f'{label:30s} | {ta:.3f}       | {la:.3f}        | {delta:+.3f}')
"
echo ""
echo "Done. $(date)"
