#!/usr/bin/env bash
# run_e2e_llama8b_gsm8k.sh — Llama 8B GSM8K End-to-End PT-CSFT
# Pipeline: generate responses → prep data → train → eval (text + logit)
#
# Usage: bash run_e2e_llama8b_gsm8k.sh
# Expected runtime: ~6 hrs generate + ~10 min train + ~90 min eval
# Run from: ~/jpwork/metacog-engineering/phase1/

set -e

MODEL_NAME="Meta-Llama-3.1-8B-Instruct-bf16"
MODEL_PATH="/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/$MODEL_NAME"
BASE_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1"
SCRIPTS_DIR="$BASE_DIR/scripts"
RESPONSES_FILE="$BASE_DIR/results_raw/domain_gen/responses_gsm8k_${MODEL_NAME}.json"
E2E_DIR="$BASE_DIR/finetune/$MODEL_NAME/brier_e2e_gsm8k"
RESULTS_DIR="$BASE_DIR/results_raw/domain_gen"

SEED=42
EPOCHS=9
LR="5e-5"

echo "=========================================="
echo "Llama 8B End-to-End PT-CSFT — GSM8K"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Started: $(date)"
echo ""

# ── Step 1: Generate greedy responses on GSM8K ──
echo "Step 1: Generate GSM8K responses..."
if [ -f "$RESPONSES_FILE" ]; then
    N_ITEMS=$(python3 -c "import json; print(len(json.load(open('$RESPONSES_FILE'))))")
    echo "  Responses exist ($N_ITEMS items), skipping"
else
    python3 "$SCRIPTS_DIR/gen_gsm8k_responses.py" \
        --model-path "$MODEL_PATH" \
        --output "$RESPONSES_FILE"
fi
echo ""

# ── Step 2: Prep training data ──
echo "Step 2: Prep end-to-end training data (coarse, binary)..."
mkdir -p "$E2E_DIR"
if [ -f "$E2E_DIR/train.jsonl" ]; then
    echo "  Training data exists, skipping"
    echo "  $(wc -l < "$E2E_DIR/train.jsonl") train items"
else
    python3 "$SCRIPTS_DIR/prep_brier_e2e.py" \
        --responses "$RESPONSES_FILE" \
        --domain gsm8k \
        --target-mode binary \
        --coarse \
        --output "$E2E_DIR/train.jsonl" \
        --seed $SEED
fi
echo ""

# ── Step 3: Config ──
N_TRAIN=$(wc -l < "$E2E_DIR/train.jsonl" | tr -d ' ')
GRAD_ACCUM=16
ITERS=$(( (N_TRAIN * EPOCHS) / GRAD_ACCUM ))
ADAPTER_DIR="$E2E_DIR/adapter_e2e_ce"
CONFIG_FILE="$E2E_DIR/config_e2e_ce.yaml"

echo "Step 3: Config..."
echo "  n_train=$N_TRAIN, epochs=$EPOCHS, grad_accum=$GRAD_ACCUM → iters=$ITERS"

cat > "$CONFIG_FILE" << EOF
model: $MODEL_PATH
data: $E2E_DIR
train: true
fine_tune_type: lora
lora_parameters:
  rank: 16
  scale: 2.0
  dropout: 0.05
  keys: ["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"]
iters: $ITERS
learning_rate: $LR
lr_schedule:
  name: cosine_decay
  arguments: [$LR, $ITERS]
batch_size: 1
grad_accumulation_steps: $GRAD_ACCUM
max_seq_length: 1024
seed: $SEED
mask_prompt: true
adapter_path: $ADAPTER_DIR
EOF

echo "  Config written: $CONFIG_FILE"
echo ""

# ── Step 4: Train ──
echo "Step 4: Training (iters=$ITERS, lr=$LR, r16)..."
if [ -f "$ADAPTER_DIR/adapters.safetensors" ]; then
    echo "  Adapter exists, skipping training"
else
    TRAIN_START=$(date +%s)
    python3 -m mlx_lm lora --config "$CONFIG_FILE" 2>&1 | tail -5
    TRAIN_END=$(date +%s)
    TRAIN_MINS=$(( (TRAIN_END - TRAIN_START) / 60 ))
    echo "  Training complete in ${TRAIN_MINS}m"
fi
echo ""

# ── Step 5: Text eval ──
echo "Step 5: Text eval (coarse)..."
OUTPUT_TEXT="$RESULTS_DIR/e2e_llama8b_gsm8k.json"
if [ -f "$OUTPUT_TEXT" ]; then
    echo "  Result exists, skipping"
else
    EVAL_START=$(date +%s)
    python3 "$SCRIPTS_DIR/eval_e2e.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$ADAPTER_DIR" \
        --benchmark gsm8k \
        --coarse \
        --output "$OUTPUT_TEXT"
    EVAL_END=$(date +%s)
    EVAL_MINS=$(( (EVAL_END - EVAL_START) / 60 ))
    echo "  Text eval complete in ${EVAL_MINS}m"
fi
echo ""

# ── Step 6: Logit eval ──
# NOTE: eval_e2e_logits.py has Gemma DIGIT_TOKEN_IDS hardcoded.
# Llama 8B digit tokens are 15-24 (not Gemma's IDs).
# Either:
#   (a) update eval_e2e_logits.py to auto-detect from tokenizer, or
#   (b) set DIGIT_TOKEN_IDS = [15,16,17,18,19,20,21,22,23,24] for Llama
# Skipping logit eval here until the script is adapted.
echo "Step 6: Logit eval..."
echo "  WARNING: eval_e2e_logits.py needs Llama digit token IDs (15-24)."
echo "  Gemma IDs are hardcoded. Adapt before running logit eval."
echo "  Skipping for now."
echo ""

# ── Summary ──
echo "=========================================="
echo "Llama 8B GSM8K End-to-End Complete"
echo "=========================================="
echo ""
echo "Comparison targets:"
echo "  Gemma 12B baseline verbal:   AUROC₂ = 0.546"
echo "  Gemma 12B E2E argmax (best): AUROC₂ = 0.803 (gentle seed 789)"
echo "  Gemma 12B E2E logit (best):  AUROC₂ = 0.775 (seed 42)"
echo "  Gemma 12B probe:             AUROC₂ = 0.769"
echo ""
echo "NOTE: Llama 8B baseline verbal and probe on GSM8K are not yet established."
echo "  Run step1_baseline_phase1.py and step1b_probe.py on Llama 8B GSM8K"
echo "  to get proper baselines for comparison."
echo ""
echo "TODO:"
echo "  1. Adapt eval_e2e_logits.py for Llama digit tokens (15-24)"
echo "  2. Run logit eval"
echo "  3. Establish Llama 8B GSM8K baseline + probe for comparison"
echo "  4. If positive, run E3 (seeds 42, 123, 456)"
echo ""
echo "Done. $(date)"
