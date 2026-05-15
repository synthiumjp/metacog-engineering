"""Experiment 4: Post-Training Probe Analysis
=============================================

Applies pre-trained probes (from step1b, trained on baseline T-cal)
to hidden states extracted from the PT-CSFT-adapted model.

Scientific question: Does PT-CSFT preserve or disrupt the internal
correctness signal that probes read?

- If probe AUROC₂ is preserved after PT-CSFT → the internal signal
  is intact; PT-CSFT only changed the verbalization mapping
- If probe AUROC₂ drops → PT-CSFT disrupted the internal
  representations (expected for Llama failure case)

Usage:
    # First, extract post-PT hidden states:
    python3 step1_baseline_phase1.py \\
        --model_path <path> \\
        --adapter-path <adapter_dir> \\
        --output_dir ./results/step1_post_pt \\
        --skip_meval --skip_tcal_hidden

    # Then run this analysis:
    python3 step_exp4_post_pt_probes.py \\
        --model-name gemma-3-12b-it

    # Compare success vs failure:
    python3 step_exp4_post_pt_probes.py --model-name gemma-3-12b-it
    python3 step_exp4_post_pt_probes.py --model-name Qwen2.5-7B-Instruct-bf16
    python3 step_exp4_post_pt_probes.py --model-name Meta-Llama-3.1-8B-Instruct-bf16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


# Directories
PROBE_DIR = Path(os.path.expanduser("~/jpwork/results/probe"))
BASELINE_STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1"))
POST_PT_STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1_post_pt"))

# Probe configs to analyse (matching step1b)
PROBE_CONFIGS = [
    "first_pre_answer_token",
    "first_last_answer_token",
    "middle_pre_answer_token",
    "middle_last_answer_token",
    "last_pre_answer_token",
    "last_last_answer_token",
]


def load_probe_fits(model_name: str) -> dict:
    """Load saved probe fits (scaler + coef + intercept) from step1b."""
    fits_file = PROBE_DIR / f"probe_fits_{model_name}.npz"
    if not fits_file.exists():
        print(f"[fatal] Probe fits not found: {fits_file}")
        sys.exit(1)

    data = np.load(fits_file)
    probes = {}
    for config in PROBE_CONFIGS:
        coef_key = f"{config}__coef"
        intercept_key = f"{config}__intercept"
        scaler_mean_key = f"{config}__scaler_mean"
        scaler_scale_key = f"{config}__scaler_scale"
        degenerate_key = f"{config}__degenerate"

        if coef_key not in data:
            continue

        # Check if probe was degenerate (skip if so)
        if degenerate_key in data and bool(data[degenerate_key]):
            print(f"  [skip] {config}: degenerate probe")
            continue

        probe = {
            "coef": data[coef_key],
            "intercept": data[intercept_key],
        }
        if scaler_mean_key in data:
            probe["scaler_mean"] = data[scaler_mean_key]
            probe["scaler_scale"] = data[scaler_scale_key]
        probes[config] = probe

    print(f"[probe] Loaded {len(probes)} probe fits from {fits_file}")
    return probes


def load_hidden_states_and_labels(npz_file: Path, responses_file: Path,
                                  layer_label: str, position: str):
    """Load hidden states and correctness labels for a given config."""
    if not npz_file.exists():
        return None, None, None
    if not responses_file.exists():
        return None, None, None

    with open(responses_file) as f:
        records = json.load(f)

    # Build qid → correct mapping
    qid_correct = {}
    for r in records:
        qid_correct[r["question_id"]] = r["correct"]

    data = np.load(npz_file)

    X = []
    y = []
    qids = []
    for key in data.files:
        parts = key.split("__")
        if len(parts) != 3:
            continue
        qid, ll, pos = parts
        if ll == layer_label and pos == position:
            if qid in qid_correct:
                X.append(data[key])
                y.append(int(qid_correct[qid]))
                qids.append(qid)

    if not X:
        return None, None, None

    return np.array(X), np.array(y), qids


def apply_probe(X, probe):
    """Apply logistic regression probe with scaling: return P(correct)."""
    X_scaled = X.copy()
    if "scaler_mean" in probe:
        X_scaled = (X_scaled - probe["scaler_mean"]) / probe["scaler_scale"]
    logits = X_scaled @ probe["coef"].T + probe["intercept"]  # (n, 1)
    probs = 1.0 / (1.0 + np.exp(-logits.ravel()))
    return probs


def auroc2(probs, labels):
    """Compute AUROC₂."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def main():
    parser = argparse.ArgumentParser(description="Exp 4: Post-PT Probe Analysis")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--post-pt-dir", type=str, default=None,
                        help="Override post-PT step1 dir (default: results/step1_post_pt)")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Override baseline step1 dir (default: results/step1)")
    args = parser.parse_args()

    model_name = args.model_name
    post_pt_dir = Path(args.post_pt_dir) if args.post_pt_dir else POST_PT_STEP1_DIR
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else BASELINE_STEP1_DIR

    print(f"{'='*60}")
    print(f"Experiment 4: Post-Training Probe Analysis")
    print(f"Model: {model_name}")
    print(f"Baseline hidden states: {baseline_dir}")
    print(f"Post-PT hidden states: {post_pt_dir}")
    print(f"{'='*60}\n")

    # Load probe fits
    probes = load_probe_fits(model_name)
    if not probes:
        print("[fatal] No probes loaded")
        sys.exit(1)

    # Load pre-PT probe metrics for comparison
    pre_pt_metrics_file = PROBE_DIR / f"probe_metrics_{model_name}.json"
    pre_pt_auroc = {}
    if pre_pt_metrics_file.exists():
        with open(pre_pt_metrics_file) as f:
            pre_pt_data = json.load(f)
        for config, metrics in pre_pt_data.get("configs", {}).items():
            pre_pt_auroc[config] = metrics.get("auroc2_probe", float("nan"))
        print(f"[ref] Loaded pre-PT probe AUROC₂ from {pre_pt_metrics_file}")

    # File paths
    post_pt_npz = post_pt_dir / f"hidden_states_teval_{model_name}.npz"
    post_pt_responses = post_pt_dir / f"teval_responses_{model_name}.json"
    baseline_npz = baseline_dir / f"hidden_states_teval_{model_name}.npz"
    baseline_responses = baseline_dir / f"teval_responses_{model_name}.json"

    if not post_pt_npz.exists():
        print(f"[fatal] Post-PT hidden states not found: {post_pt_npz}")
        print(f"  Run step1_baseline_phase1.py with --adapter-path first")
        sys.exit(1)

    # Load post-PT responses for accuracy
    with open(post_pt_responses) as f:
        post_pt_records = json.load(f)
    post_pt_acc = sum(r["correct"] for r in post_pt_records) / len(post_pt_records)

    # Load baseline responses for comparison
    with open(baseline_responses) as f:
        baseline_records = json.load(f)
    baseline_acc = sum(r["correct"] for r in baseline_records) / len(baseline_records)

    print(f"\n  Baseline accuracy: {baseline_acc:.3f}")
    print(f"  Post-PT accuracy:  {post_pt_acc:.3f}")
    print(f"  Accuracy delta:    {post_pt_acc - baseline_acc:+.3f}\n")

    # Apply each probe to both baseline and post-PT hidden states
    results = {}

    print(f"{'Config':<30} {'Pre-PT AUROC₂':>14} {'Post-PT AUROC₂':>15} {'Delta':>8} {'n_pre':>6} {'n_post':>7}")
    print("-" * 85)

    for config in PROBE_CONFIGS:
        if config not in probes:
            continue

        probe = probes[config]
        parts = config.rsplit("_", 2)  # e.g., "middle_pre_answer_token"
        # Parse layer and position from config name
        if "first_pre" in config:
            layer_label, position = "first", "pre_answer_token"
        elif "first_last" in config:
            layer_label, position = "first", "last_answer_token"
        elif "middle_pre" in config:
            layer_label, position = "middle", "pre_answer_token"
        elif "middle_last" in config:
            layer_label, position = "middle", "last_answer_token"
        elif "last_pre" in config:
            layer_label, position = "last", "pre_answer_token"
        elif "last_last" in config:
            layer_label, position = "last", "last_answer_token"
        else:
            continue

        # Apply probe to baseline hidden states
        X_base, y_base, qids_base = load_hidden_states_and_labels(
            baseline_npz, baseline_responses, layer_label, position
        )
        if X_base is not None:
            probs_base = apply_probe(X_base, probe)
            auroc_base = auroc2(probs_base, y_base)
            n_base = len(y_base)
        else:
            auroc_base = float("nan")
            n_base = 0

        # Apply probe to post-PT hidden states
        X_post, y_post, qids_post = load_hidden_states_and_labels(
            post_pt_npz, post_pt_responses, layer_label, position
        )
        if X_post is not None:
            probs_post = apply_probe(X_post, probe)
            auroc_post = auroc2(probs_post, y_post)
            n_post = len(y_post)
        else:
            auroc_post = float("nan")
            n_post = 0

        delta = auroc_post - auroc_base if not (
            np.isnan(auroc_post) or np.isnan(auroc_base)
        ) else float("nan")

        # Use pre-PT from step1b as reference if available
        ref_auroc = pre_pt_auroc.get(config, auroc_base)

        results[config] = {
            "auroc_baseline_recomputed": auroc_base,
            "auroc_post_pt": auroc_post,
            "auroc_pre_pt_step1b": ref_auroc,
            "delta": delta,
            "n_baseline": n_base,
            "n_post_pt": n_post,
        }

        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "  nan"
        marker = " ★" if "last_last" in config else ""
        print(f"  {config:<28} {auroc_base:>12.3f}   {auroc_post:>13.3f}   {delta_str:>7}  {n_base:>5}  {n_post:>6}{marker}")

    # Summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4 SUMMARY: {model_name}")
    print(f"{'='*60}")

    primary = results.get("last_last_answer_token", {})
    peak_mid = results.get("middle_pre_answer_token", {})

    if primary:
        print(f"\n  Primary (last_last_answer_token):")
        print(f"    Pre-PT probe AUROC₂:  {primary.get('auroc_baseline_recomputed', float('nan')):.3f}")
        print(f"    Post-PT probe AUROC₂: {primary.get('auroc_post_pt', float('nan')):.3f}")
        d = primary.get('delta', float('nan'))
        if not np.isnan(d):
            print(f"    Delta:                {d:+.3f}")
            if d > 0.02:
                print(f"    → Probe signal STRENGTHENED after PT-CSFT")
            elif d < -0.02:
                print(f"    → Probe signal WEAKENED after PT-CSFT")
            else:
                print(f"    → Probe signal approximately PRESERVED")

    if peak_mid:
        print(f"\n  Peak middle (middle_pre_answer_token):")
        print(f"    Pre-PT probe AUROC₂:  {peak_mid.get('auroc_baseline_recomputed', float('nan')):.3f}")
        print(f"    Post-PT probe AUROC₂: {peak_mid.get('auroc_post_pt', float('nan')):.3f}")
        d = peak_mid.get('delta', float('nan'))
        if not np.isnan(d):
            print(f"    Delta:                {d:+.3f}")

    print(f"\n  Post-PT accuracy: {post_pt_acc:.3f} (baseline: {baseline_acc:.3f}, "
          f"drop: {post_pt_acc - baseline_acc:+.3f})")
    print(f"{'='*60}\n")

    # Save results
    output = {
        "model": model_name,
        "baseline_accuracy": baseline_acc,
        "post_pt_accuracy": post_pt_acc,
        "probe_results": results,
    }

    out_dir = Path(os.path.expanduser("~/jpwork/results/exp4"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"exp4_post_pt_probes_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"[save] {out_file}")


if __name__ == "__main__":
    main()
