#!/usr/bin/env bash
# run_e2e_70b_triviaqa.sh — Full pipeline: data prep → config → train → eval
#
# Usage: bash run_e2e_70b_triviaqa.sh
# Expected runtime: ~3-4 hrs (train ~2-3 hrs, eval ~1-2 hrs on 70B)
# Run from: ~/jpwork/metacog-engineering/phase1/

set -e

MODEL_NAME="Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED"
MODEL_PATH="/Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/$MODEL_NAME"
BASE_DIR="/Users/chrismarmo/jpwork/metacog-engineering/phase1"
TCAL_FILE="$BASE_DIR/results_raw/step1/tcal_greedy_responses_${MODEL_NAME}.json"
E2E_DIR="$BASE_DIR/finetune/$MODEL_NAME/e2e_triviaqa"
SCRIPTS_DIR="$BASE_DIR/scripts"
RESULTS_DIR="$BASE_DIR/results_raw/domain_gen"

SEED=42
EPOCHS=3

echo "=========================================="
echo "70B End-to-End PT-CSFT — TriviaQA"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Started: $(date)"
echo ""

# ── Step 1: Data prep ──
echo "Step 1: Data prep..."
if [ -f "$E2E_DIR/train.jsonl" ]; then
    echo "  Training data exists, skipping"
    echo "  $(wc -l < "$E2E_DIR/train.jsonl") train items"
else
    python3 "$SCRIPTS_DIR/prep_e2e_70b_triviaqa.py" \
        --tcal-responses "$TCAL_FILE" \
        --output-dir "$E2E_DIR" \
        --seed $SEED
fi
echo ""

# ── Step 2: Compute iters and create config ──
N_TRAIN=$(wc -l < "$E2E_DIR/train.jsonl" | tr -d ' ')
GRAD_ACCUM=16
ITERS=$(( (N_TRAIN * EPOCHS) / GRAD_ACCUM ))
LR="5e-5"

echo "Step 2: Config..."
echo "  n_train=$N_TRAIN, epochs=$EPOCHS, grad_accum=$GRAD_ACCUM → iters=$ITERS"

CONFIG_FILE="$E2E_DIR/config_e2e_ce.yaml"
ADAPTER_DIR="$E2E_DIR/adapter_e2e_ce"

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
max_seq_length: 512
seed: $SEED
mask_prompt: true
adapter_path: $ADAPTER_DIR
EOF

echo "  Config written: $CONFIG_FILE"
echo ""

# ── Step 3: Train ──
echo "Step 3: Training (iters=$ITERS, lr=$LR, r16)..."
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

# ── Step 4: Eval (text-based, using eval_ablation.py) ──
echo "Step 4: Text eval..."
OUTPUT_FILE="$RESULTS_DIR/e2e_70b_triviaqa.json"

if [ -f "$OUTPUT_FILE" ]; then
    echo "  Result exists, skipping"
else
    EVAL_START=$(date +%s)
    python3 "$SCRIPTS_DIR/eval_ablation.py" \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --adapter-path "$ADAPTER_DIR" \
        --label "e2e_70b_triviaqa"
    EVAL_END=$(date +%s)
    EVAL_MINS=$(( (EVAL_END - EVAL_START) / 60 ))
    echo "  Eval complete in ${EVAL_MINS}m"
fi
echo ""

# ── Summary ──
echo "=========================================="
echo "70B End-to-End Complete"
echo "=========================================="
echo ""
echo "Comparison targets:"
echo "  Baseline verbal:     AUROC₂ = 0.724"
echo "  Balanced confonly v1: AUROC₂ = 0.674 (VRS Valid)"
echo "  Probe:               AUROC₂ = 0.834"
echo ""
echo "Results saved by eval_ablation.py to:"
echo "  $BASE_DIR/results_raw/step4/"
echo ""
echo "TODO after text eval:"
echo "  1. Logit eval (needs eval_e2e_logits.py adapted for 0-100 fine mode)"
echo "  2. Bootstrap CIs"
echo "  3. If positive, run E3 (seeds 42, 123, 456)"
echo ""
echo "Done. $(date)"
