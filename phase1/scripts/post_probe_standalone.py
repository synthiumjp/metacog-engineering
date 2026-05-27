#!/usr/bin/env python3
"""
Post-Training Probe Analysis: Llama 8B gentle-lr
=================================================
Loads model+adapter, generates on T-eval to extract hidden states,
retrains probes on post-PT hidden states, compares to §9.4 baselines.

Tests whether the "preserved information with blocked verbalization"
pattern from the failed config is specific to that config or general
to LlamaForCausalLM.

Usage:
    python3 post_probe_standalone.py

Runtime: ~40 minutes on M3 Ultra
"""

import json, os, numpy as np, random, time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config — edit these paths for your setup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16"
)
ADAPTER_PATH = os.path.expanduser(
    "~/jpwork/metacog-engineering/phase1/results_raw/finetune/"
    "Meta-Llama-3.1-8B-Instruct-bf16/ablation_gentle_lr/adapters"
)
OUTPUT_DIR = os.path.expanduser(
    "~/jpwork/metacog-engineering/phase1/results_raw/step1_post_pt_gentle"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known baseline values from paper Table 15 / §9.4
BASELINE_PROBES = {"last": 0.806, "middle": 0.831}
FAILED_CONFIG = {"last_retrained": 0.749, "middle_retrained": 0.828}

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
import mlx.core as mx
from mlx_lm import load, generate

print("Loading model + gentle-lr adapter...")
model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)

if hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
    embed = model.model.embed_tokens
elif hasattr(model, "language_model"):
    layers = model.language_model.model.layers
    embed = model.language_model.model.embed_tokens
else:
    raise ValueError("Cannot find transformer layers")

n_layers = len(layers)
mid_layer = n_layers // 2
LAYER_INDICES = {"first": 0, "middle": mid_layer, "last": n_layers - 1}
print(f"Layers: {n_layers}, middle: {mid_layer}")

# ---------------------------------------------------------------------------
# Load T-eval (seed-42 shuffle, first 1000)
# ---------------------------------------------------------------------------
ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
indices = list(range(len(ds)))
random.seed(42)
random.shuffle(indices)
eval_items = [ds[int(i)] for i in indices[:1000]]
print(f"T-eval: {len(eval_items)} items")

PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

# ---------------------------------------------------------------------------
# Extract hidden states
# ---------------------------------------------------------------------------
print("\nExtracting hidden states...")
hidden_states = {name: [] for name in LAYER_INDICES}
answers_correct = []
t0 = time.time()

for idx, item in enumerate(eval_items):
    question = item["question"]
    aliases = item["answer"]["aliases"]

    messages = [{"role": "user", "content": PROMPT.format(question=question)}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, prompt=prompt_text, max_tokens=256)

    # Flex-match correctness
    resp_lower = response.lower().strip()
    is_correct = any(
        (alias.lower() in resp_lower or resp_lower in alias.lower())
        for alias in aliases
        if len(alias) >= 2 and len(resp_lower) >= 2
    )
    answers_correct.append(int(is_correct))

    # Tokenize full sequence for hidden state extraction
    full_text = prompt_text + response
    input_ids = mx.array(tokenizer.encode(full_text)).reshape(1, -1)
    last_pos = input_ids.shape[1] - 1

    # Manual forward pass — capture at target layers
    h = embed(input_ids)
    seq_len = h.shape[1]
    # CRITICAL: mask must be bfloat16 to match model dtype
    mask = mx.full((seq_len, seq_len), -1e9, dtype=mx.bfloat16)
    mask = mx.triu(mask, k=1)

    for layer_idx in range(n_layers):
        h = layers[layer_idx](h, mask=mask)
        for name, target_idx in LAYER_INDICES.items():
            if layer_idx == target_idx:
                state = np.array(h[0, last_pos, :].tolist(), dtype=np.float32)
                hidden_states[name].append(state)

    mx.eval(h)

    if idx % 50 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (1000 - idx - 1) / rate / 60 if rate > 0 else 0
        print(
            f"  [{idx:4d}/1000] acc={np.mean(answers_correct):.3f} "
            f"({elapsed/60:.0f}min, ~{eta:.0f}min remaining)"
        )

print(f"\nPost-PT accuracy: {np.mean(answers_correct):.3f}")

# ---------------------------------------------------------------------------
# Probe analysis
# ---------------------------------------------------------------------------
y = np.array(answers_correct)

print("\n" + "=" * 60)
print("PROBE ANALYSIS")
print("=" * 60)

results = {
    "model": "Llama-8B",
    "config": "gentle-lr (r16/lr5e-5)",
    "post_pt_accuracy": float(np.mean(answers_correct)),
    "n_items": len(answers_correct),
}

for name in ["last", "middle", "first"]:
    hs = np.array(hidden_states[name])
    n = min(len(hs), len(y))
    hs, y_use = hs[:n], y[:n]

    print(f"\n--- {name} layer (idx {LAYER_INDICES[name]}) ---")
    print(f"  Shape: {hs.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(hs)

    clf = LogisticRegressionCV(
        Cs=10, cv=5, solver="lbfgs", max_iter=2000,
        scoring="roc_auc", random_state=42,
    )
    cv_preds = cross_val_predict(clf, X, y_use, cv=5, method="predict_proba")[:, 1]
    retrained_auroc = roc_auc_score(y_use, cv_preds)
    results[f"{name}_retrained_cv"] = float(retrained_auroc)

    baseline_ref = BASELINE_PROBES.get(name, None)
    failed_ref = FAILED_CONFIG.get(f"{name}_retrained", None)

    print(f"  Retrained probe (5-fold CV): AUROC2 = {retrained_auroc:.3f}")
    if baseline_ref:
        print(f"  vs baseline on baseline states:   {baseline_ref:.3f} (delta = {retrained_auroc - baseline_ref:+.3f})")
    if failed_ref:
        print(f"  vs failed-config retrained:       {failed_ref:.3f}")

# Interpretation
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")

mid = results.get("middle_retrained_cv", 0)
mid_base = BASELINE_PROBES.get("middle", 0)

if mid > mid_base + 0.01:
    print("Pattern: INFORMATION AMPLIFIED (like Gemma 12B)")
elif mid > mid_base - 0.01:
    print("Pattern: INFORMATION PRESERVED (same as failed config)")
else:
    print(f"Pattern: PARTIAL LOSS (delta = {mid - mid_base:+.3f})")

print(f"\nComparison table for §9.4:")
print(f"{'Metric':<40} {'Failed(2e-4)':<14} {'Gentle(5e-5)':<14}")
print("-" * 68)
for layer in ["last", "middle"]:
    key = f"{layer}_retrained_cv"
    failed_key = f"{layer}_retrained"
    if key in results:
        fv = FAILED_CONFIG.get(failed_key, "N/A")
        fv_str = f"{fv:.3f}" if isinstance(fv, float) else fv
        print(f"  Retrained probe ({layer}):{' '*(18-len(layer))}{fv_str:<14} {results[key]:<14.3f}")

output_path = os.path.join(OUTPUT_DIR, "probe_analysis_gentle_lr.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {output_path}")
