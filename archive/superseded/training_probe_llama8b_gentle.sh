#!/bin/bash
# ===========================================================================
# Post-Training Probe Analysis: Llama 8B gentle-lr
# ---------------------------------------------------------------------------
# Runs the §9.4 analysis on the SUCCESSFUL gentle-lr config instead of
# the failed r16/lr2e-4 config. Tests whether the "preserved information
# with blocked verbalization" pattern is config-specific or general to
# LlamaForCausalLM.
#
# Steps:
#   1. Extract hidden states from Llama 8B with gentle-lr adapter loaded
#   2. Apply original baseline-trained probes to post-PT hidden states
#   3. Retrain fresh probes on post-PT hidden states (5-fold CV)
#   4. Compare: did information get amplified (like Gemma) or preserved?
#
# Usage: bash post_training_probe_llama8b_gentle.sh
# Runtime estimate: ~2-3 hours on M3 Ultra
# ===========================================================================

set -euo pipefail

SCRIPTS_DIR=~/jpwork/metacog-engineering/phase1/scripts
MODEL_PATH=~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16
MODEL_NAME="Meta-Llama-3.1-8B-Instruct-bf16"
ADAPTER_PATH=~/jpwork/metacog-engineering/phase1/results_raw/finetune/Meta-Llama-3.1-8B-Instruct-bf16/ablation_gentle_lr/adapters
OUTPUT_DIR=~/jpwork/metacog-engineering/phase1/results_raw/step1_post_pt_gentle

source ~/jpwork/metacog-engineering/.venv_metacog/bin/activate
cd "$SCRIPTS_DIR"

echo "================================================================"
echo "Post-Training Probe: Llama 8B gentle-lr"
echo "Adapter: $ADAPTER_PATH"
echo "Output: $OUTPUT_DIR"
echo "================================================================"

# Step 1: Extract hidden states with adapter loaded
# (Uses step1_baseline_phase1.py with --adapter-path flag)
echo ""
echo "Step 1: Extracting hidden states with gentle-lr adapter..."
if [[ -d "$OUTPUT_DIR" ]] && ls "$OUTPUT_DIR"/*hidden_states* 2>/dev/null | head -1 > /dev/null; then
    echo "  Hidden states exist, skipping extraction"
else
    python3 step1_baseline_phase1.py \
        --model_path "$MODEL_PATH" \
        --adapter-path "$ADAPTER_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --skip_meval --skip_tcal_hidden
    echo "  Extraction complete"
fi

# Step 2 & 3: Apply original probes + retrain fresh probes
echo ""
echo "Step 2-3: Probe analysis..."

python3 - << 'PYEOF'
"""
Post-training probe analysis for Llama 8B gentle-lr.
Compares original-probe and retrained-probe AUROC₂ on post-PT hidden states.
"""
import json, os, glob, pickle, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

# Paths
RESULTS_BASE = Path(os.path.expanduser("~/jpwork/metacog-engineering/phase1/results_raw"))
POST_PT_DIR = Path(os.path.expanduser("~/jpwork/metacog-engineering/phase1/results_raw/step1_post_pt_gentle"))
BASELINE_STEP1_DIR = RESULTS_BASE / "step1"
PROBE_DIR = RESULTS_BASE / "probe"

# Also check symlinked locations
PROBE_DIR_ALT = Path(os.path.expanduser("~/jpwork/results/probe"))
STEP1_DIR_ALT = Path(os.path.expanduser("~/jpwork/results/step1"))

MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-bf16"

# Layer positions to test
LAYER_POSITIONS = ["first", "middle", "last"]
TOKEN_POSITION = "last_answer_token"  # primary

def find_file(pattern, dirs):
    """Search for a file matching pattern in multiple directories."""
    for d in dirs:
        matches = glob.glob(str(d / pattern))
        if matches:
            return matches[0]
    return None

def load_hidden_states(step1_dir, model_name, layer_pos, token_pos="last_answer_token"):
    """Load hidden states from step1 output."""
    # Try various naming patterns
    patterns = [
        f"*{model_name}*{layer_pos}*{token_pos}*teval*.npy",
        f"*{model_name}*{layer_pos}_{token_pos}*teval*.npy",
        f"*{model_name}*teval*{layer_pos}*{token_pos}*.npy",
    ]
    for pat in patterns:
        matches = glob.glob(str(step1_dir / pat))
        if matches:
            return np.load(matches[0])

    # Broader search
    all_npy = glob.glob(str(step1_dir / f"*{model_name}*teval*.npy"))
    for f in all_npy:
        if layer_pos in f and token_pos in f:
            return np.load(f)

    return None

def load_correctness_labels(step1_dir, model_name):
    """Load correctness labels from step1 eval results."""
    patterns = [
        f"*{model_name}*teval*.json",
        f"*{model_name}*eval*.json",
    ]
    for pat in patterns:
        for d in [step1_dir, BASELINE_STEP1_DIR, STEP1_DIR_ALT]:
            matches = glob.glob(str(d / pat))
            for m in matches:
                try:
                    with open(m) as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 100:
                        labels = []
                        for item in data:
                            if "correct" in item:
                                labels.append(int(item["correct"]))
                        if len(labels) > 100:
                            return np.array(labels)
                except:
                    pass
    return None

def load_original_probe(probe_dir, model_name, layer_pos):
    """Load the original baseline-trained probe."""
    patterns = [
        f"*{model_name}*{layer_pos}*{TOKEN_POSITION}*probe*.pkl",
        f"*{model_name}*{layer_pos}*.pkl",
    ]
    for pat in patterns:
        for d in [probe_dir, PROBE_DIR_ALT]:
            matches = glob.glob(str(d / pat))
            if matches:
                with open(matches[0], "rb") as f:
                    return pickle.load(f)
    return None

print("=" * 60)
print("Post-Training Probe Analysis: Llama 8B gentle-lr")
print("=" * 60)

# Load correctness labels
labels = load_correctness_labels(POST_PT_DIR, MODEL_NAME)
if labels is None:
    # Try loading from baseline results
    labels = load_correctness_labels(BASELINE_STEP1_DIR, MODEL_NAME)

if labels is None:
    print("ERROR: Could not find correctness labels. Listing available files:")
    for d in [POST_PT_DIR, BASELINE_STEP1_DIR, STEP1_DIR_ALT]:
        print(f"\n  {d}:")
        for f in sorted(glob.glob(str(d / f"*{MODEL_NAME}*"))):
            print(f"    {os.path.basename(f)}")
    exit(1)

print(f"\nCorrectness labels: {len(labels)} items, {labels.sum()} correct ({labels.mean():.1%})")

results = {}

for layer_pos in LAYER_POSITIONS:
    print(f"\n{'─'*50}")
    print(f"Layer: {layer_pos}")
    print(f"{'─'*50}")

    # Load post-PT hidden states
    post_hs = load_hidden_states(POST_PT_DIR, MODEL_NAME, layer_pos)
    if post_hs is None:
        print(f"  Post-PT hidden states not found for {layer_pos}")
        # List what's available
        all_files = glob.glob(str(POST_PT_DIR / f"*{MODEL_NAME}*.npy"))
        if all_files:
            print(f"  Available .npy files:")
            for f in all_files:
                print(f"    {os.path.basename(f)}")
        continue

    n = min(len(post_hs), len(labels))
    post_hs = post_hs[:n]
    y = labels[:n]
    print(f"  Post-PT hidden states: {post_hs.shape}")

    # 1. Apply original baseline-trained probe
    orig_probe = load_original_probe(PROBE_DIR, MODEL_NAME, layer_pos)
    if orig_probe is not None:
        try:
            # The probe may expect scaled inputs
            scaler = StandardScaler()
            post_hs_scaled = scaler.fit_transform(post_hs)
            orig_preds = orig_probe.predict_proba(post_hs_scaled)[:, 1]
            orig_auroc = roc_auc_score(y, orig_preds)
            print(f"  Original probe on post-PT states: AUROC₂ = {orig_auroc:.3f}")
            results[f"{layer_pos}_original_on_postpt"] = orig_auroc
        except Exception as e:
            print(f"  Original probe failed: {e}")
            # Try without scaling
            try:
                orig_preds = orig_probe.predict_proba(post_hs)[:, 1]
                orig_auroc = roc_auc_score(y, orig_preds)
                print(f"  Original probe (unscaled) on post-PT states: AUROC₂ = {orig_auroc:.3f}")
                results[f"{layer_pos}_original_on_postpt"] = orig_auroc
            except Exception as e2:
                print(f"  Also failed unscaled: {e2}")
    else:
        print(f"  Original probe not found for {layer_pos}")

    # 2. Retrain fresh probe on post-PT hidden states (5-fold CV)
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(post_hs)

        clf = LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            scoring="roc_auc",
            random_state=42,
        )
        # Use cross_val_predict for unbiased estimate
        cv_preds = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        retrained_auroc = roc_auc_score(y, cv_preds)

        # Also fit on full data for reference
        clf.fit(X, y)
        full_preds = clf.predict_proba(X)[:, 1]
        full_auroc = roc_auc_score(y, full_preds)

        print(f"  Retrained probe (5-fold CV): AUROC₂ = {retrained_auroc:.3f}")
        print(f"  Retrained probe (full fit):  AUROC₂ = {full_auroc:.3f}")
        results[f"{layer_pos}_retrained_cv"] = retrained_auroc
        results[f"{layer_pos}_retrained_full"] = full_auroc
    except Exception as e:
        print(f"  Retrained probe failed: {e}")

# Summary comparison
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

# Known baseline values from paper (Table 15 / §9.4)
baseline_probes = {
    "last": 0.806,    # Original probe on baseline states (CV)
    "middle": 0.831,  # Original middle probe on baseline states
}

failed_config = {
    "last_original_delta": -0.192,
    "last_retrained_cv": 0.749,
    "middle_retrained_cv": 0.828,
}

print("\nComparison with failed config (r16/lr2e-4) from §9.4:")
print(f"{'Metric':<45} {'Failed (2e-4)':<15} {'Gentle (5e-5)':<15}")
print("─" * 75)

for layer in ["last", "middle"]:
    baseline_val = baseline_probes.get(layer, "—")

    # Original probe delta
    orig_key = f"{layer}_original_on_postpt"
    if orig_key in results and layer in baseline_probes:
        delta = results[orig_key] - baseline_probes[layer]
        failed_delta = failed_config.get(f"{layer}_original_delta", "—")
        print(f"  Original probe δ ({layer}):                {failed_delta:<15} {delta:+.3f}")

    # Retrained probe
    retrained_key = f"{layer}_retrained_cv"
    if retrained_key in results:
        failed_retrained = failed_config.get(f"{layer}_retrained_cv", "—")
        print(f"  Retrained probe CV ({layer}):               {failed_retrained:<15.3f} {results[retrained_key]:.3f}")

print(f"\nBaseline reference (probe on baseline states):")
for layer, val in baseline_probes.items():
    print(f"  {layer}: {val:.3f}")

# Save results
output_path = POST_PT_DIR / "probe_analysis_gentle_lr.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Interpretation
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")
if "middle_retrained_cv" in results:
    middle_retrained = results["middle_retrained_cv"]
    if middle_retrained > baseline_probes.get("middle", 0) + 0.01:
        print("Pattern: INFORMATION AMPLIFIED (like Gemma 12B)")
        print("  → Gentle-lr PT-CSFT creates additional linearly decodable structure")
        print("  → Different from failed config which preserved information but blocked verbalization")
    elif middle_retrained > baseline_probes.get("middle", 0) - 0.01:
        print("Pattern: INFORMATION PRESERVED (same as failed config)")
        print("  → Gentle-lr preserves correctness info but routes it better")
        print("  → The difference between success and failure is routing quality, not information content")
    else:
        delta = middle_retrained - baseline_probes.get("middle", 0)
        print(f"Pattern: PARTIAL LOSS (δ = {delta:+.3f})")
        print("  → Gentle-lr partially reorganises representations")
        print("  → Need to check if verbal AUROC₂ improved despite probe loss")

PYEOF

echo ""
echo "Post-training probe analysis complete."
