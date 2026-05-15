"""Reviewer Response Experiments
================================

Experiment A: Retrained probes on post-PT-CSFT hidden states
  - 5-fold CV probing on post-PT T-eval hidden states
  - Answers: did information move or disappear?

Experiment B: ECE and Brier scores
  - Computed from saved step4 response files
  - Addresses: "discrimination vs calibration" concern

Usage:
    python3 step_reviewer_experiments.py \
        --model-name gemma-3-12b-it \
        --adapter-label probe_target
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1"))
POST_PT_DIR = Path(os.path.expanduser("~/jpwork/results/step1_post_pt"))
STEP4_DIR = Path(os.path.expanduser("~/jpwork/results/step4"))
OUTPUT_DIR = Path(os.path.expanduser("~/jpwork/results/reviewer_experiments"))


# ================================================================
# Experiment A: Retrained Probes
# ================================================================

def load_hidden_states_for_probing(npz_file, responses_file, layer_label, position):
    """Load hidden states and labels for a given layer/position config."""
    if not npz_file.exists() or not responses_file.exists():
        return None, None, None

    with open(responses_file) as f:
        records = json.load(f)

    qid_key = "question_id" if "question_id" in records[0] else "qid"
    correct_key = "correct"
    qid_correct = {r[qid_key]: r[correct_key] for r in records}

    data = np.load(npz_file)
    X, y, qids = [], [], []
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


def cv_probe_auroc(X, y, n_folds=5, seed=42):
    """5-fold CV probe: returns mean AUROC₂ and per-fold AUROCs."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegressionCV(
            Cs=5, cv=3, max_iter=1000, solver="lbfgs",
            random_state=seed
        )
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]
        auroc = roc_auc_score(y_test, probs)
        fold_aurocs.append(auroc)

    if not fold_aurocs:
        return float("nan"), []
    return float(np.mean(fold_aurocs)), fold_aurocs


def run_retrained_probes(model_name):
    """Compare baseline vs retrained probes on post-PT hidden states."""
    print(f"\n{'='*60}")
    print(f"Experiment A: Retrained Probes")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    post_pt_npz = POST_PT_DIR / f"hidden_states_teval_{model_name}.npz"
    post_pt_responses = POST_PT_DIR / f"teval_responses_{model_name}.json"
    baseline_npz = STEP1_DIR / f"hidden_states_teval_{model_name}.npz"
    baseline_responses = STEP1_DIR / f"teval_responses_{model_name}.json"

    if not post_pt_npz.exists():
        print(f"  [skip] No post-PT hidden states found: {post_pt_npz}")
        return None

    configs = [
        ("first", "pre_answer_token"),
        ("first", "last_answer_token"),
        ("middle", "pre_answer_token"),
        ("middle", "last_answer_token"),
        ("last", "pre_answer_token"),
        ("last", "last_answer_token"),
    ]

    results = {}

    print(f"  {'Config':<30} {'Baseline CV':>12} {'Post-PT CV':>12} {'Delta':>8} {'Interpretation'}")
    print(f"  {'-'*80}")

    for layer, position in configs:
        config_name = f"{layer}_{position}"

        # CV probe on baseline hidden states
        X_base, y_base, _ = load_hidden_states_for_probing(
            baseline_npz, baseline_responses, layer, position
        )
        if X_base is not None:
            auroc_base_cv, _ = cv_probe_auroc(X_base, y_base)
            n_base = len(y_base)
        else:
            auroc_base_cv = float("nan")
            n_base = 0

        # CV probe on post-PT hidden states
        X_post, y_post, _ = load_hidden_states_for_probing(
            post_pt_npz, post_pt_responses, layer, position
        )
        if X_post is not None:
            auroc_post_cv, _ = cv_probe_auroc(X_post, y_post)
            n_post = len(y_post)
        else:
            auroc_post_cv = float("nan")
            n_post = 0

        delta = auroc_post_cv - auroc_base_cv if not (
            np.isnan(auroc_post_cv) or np.isnan(auroc_base_cv)
        ) else float("nan")

        if not np.isnan(delta):
            if delta > -0.02:
                interp = "PRESERVED (info moved, not lost)"
            elif delta > -0.10:
                interp = "PARTIALLY PRESERVED"
            else:
                interp = "DEGRADED (info may be lost)"
        else:
            interp = "N/A"

        marker = " ★" if config_name == "last_last_answer_token" else ""
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "  nan"
        print(f"  {config_name:<30} {auroc_base_cv:>10.3f}   {auroc_post_cv:>10.3f}   {delta_str:>7}  {interp}{marker}")

        results[config_name] = {
            "auroc_baseline_cv": auroc_base_cv,
            "auroc_post_pt_cv": auroc_post_cv,
            "delta": delta,
            "n_baseline": n_base,
            "n_post_pt": n_post,
            "interpretation": interp,
        }

    # Summary
    primary = results.get("last_last_answer_token", {})
    peak = results.get("middle_pre_answer_token", {})

    print(f"\n  Summary:")
    if primary:
        d = primary.get("delta", float("nan"))
        print(f"    Primary (last_last): baseline CV={primary.get('auroc_baseline_cv',0):.3f} → "
              f"post-PT CV={primary.get('auroc_post_pt_cv',0):.3f} (δ={d:+.3f})")
    if peak:
        d = peak.get("delta", float("nan"))
        print(f"    Peak (middle_pre):   baseline CV={peak.get('auroc_baseline_cv',0):.3f} → "
              f"post-PT CV={peak.get('auroc_post_pt_cv',0):.3f} (δ={d:+.3f})")

    return results


# ================================================================
# Experiment B: ECE and Brier Scores
# ================================================================

def compute_ece(confidences, correct, n_bins=10):
    """Expected Calibration Error."""
    c = np.asarray(confidences, dtype=float) / 100.0  # normalize to [0,1]
    y = np.asarray(correct, dtype=float)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]

    if len(c) == 0:
        return float("nan")

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (c > bin_boundaries[i]) & (c <= bin_boundaries[i + 1])
        if bin_mask.sum() == 0:
            continue
        avg_conf = c[bin_mask].mean()
        avg_acc = y[bin_mask].mean()
        ece += bin_mask.sum() / len(c) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_brier(confidences, correct):
    """Brier score."""
    c = np.asarray(confidences, dtype=float) / 100.0
    y = np.asarray(correct, dtype=float)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    if len(c) == 0:
        return float("nan")
    return float(np.mean((c - y) ** 2))


def run_calibration_metrics(model_name, adapter_label):
    """Compute ECE and Brier for baseline and PT-CSFT."""
    print(f"\n{'='*60}")
    print(f"Experiment B: Calibration Metrics (ECE + Brier)")
    print(f"Model: {model_name}, Adapter: {adapter_label}")
    print(f"{'='*60}\n")

    # Load baseline
    baseline_file = STEP1_DIR / f"teval_responses_{model_name}.json"
    if not baseline_file.exists():
        print(f"  [skip] Baseline not found: {baseline_file}")
        return None

    with open(baseline_file) as f:
        baseline_records = json.load(f)

    qid_key = "question_id" if "question_id" in baseline_records[0] else "qid"
    conf_key = "parsed_confidence" if "parsed_confidence" in baseline_records[0] else "confidence"

    base_conf = [r.get(conf_key) for r in baseline_records]
    base_correct = [int(r["correct"]) for r in baseline_records]
    base_conf_clean = [c if c is not None else float("nan") for c in base_conf]

    # Load PT-CSFT
    candidates = []
    for pattern in [
        f"step4_{adapter_label}_teval_{model_name}.json",
        f"step4_{adapter_label}_{model_name}.json",
    ]:
        found = glob.glob(str(STEP4_DIR / pattern))
        candidates.extend([c for c in found if "_metrics" not in c])

    if not candidates:
        print(f"  [skip] PT-CSFT results not found for {adapter_label}")
        return None

    with open(candidates[0]) as f:
        ft_data = json.load(f)

    if isinstance(ft_data, dict):
        ft_records = ft_data.get("results", ft_data.get("records", []))
    else:
        ft_records = ft_data

    ft_conf_key = "confidence" if "confidence" in ft_records[0] else "parsed_confidence"
    ft_conf = [r.get(ft_conf_key) for r in ft_records]
    ft_correct = [int(r["correct"]) for r in ft_records]
    ft_conf_clean = [c if c is not None else float("nan") for c in ft_conf]

    # Compute metrics
    base_ece = compute_ece(base_conf_clean, base_correct)
    base_brier = compute_brier(base_conf_clean, base_correct)
    ft_ece = compute_ece(ft_conf_clean, ft_correct)
    ft_brier = compute_brier(ft_conf_clean, ft_correct)

    print(f"  {'Metric':<20} {'Baseline':>10} {'PT-CSFT':>10} {'Delta':>10} {'Better?':>8}")
    print(f"  {'-'*60}")
    ece_delta = ft_ece - base_ece
    brier_delta = ft_brier - base_brier
    print(f"  {'ECE':<20} {base_ece:>10.4f} {ft_ece:>10.4f} {ece_delta:>+10.4f} {'✓' if ece_delta < 0 else '✗':>8}")
    print(f"  {'Brier':<20} {base_brier:>10.4f} {ft_brier:>10.4f} {brier_delta:>+10.4f} {'✓' if brier_delta < 0 else '✗':>8}")

    # Also compute for perfect calibration reference
    base_acc = np.mean(base_correct)
    ft_acc = np.mean(ft_correct)
    print(f"\n  Accuracy: baseline={base_acc:.3f}, PT-CSFT={ft_acc:.3f}")
    print(f"  Note: ECE/Brier are affected by both discrimination AND accuracy.")
    print(f"  A model that always says 75% on items it gets 75% correct has ECE≈0 but AUROC₂=0.5.")

    results = {
        "baseline_ece": base_ece,
        "baseline_brier": base_brier,
        "ft_ece": ft_ece,
        "ft_brier": ft_brier,
        "ece_delta": ece_delta,
        "brier_delta": brier_delta,
        "baseline_accuracy": base_acc,
        "ft_accuracy": ft_acc,
    }
    return results


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Reviewer Response Experiments")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-label", default="probe_target")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Experiment A: Retrained probes
    probe_results = run_retrained_probes(args.model_name)

    # Experiment B: Calibration metrics
    cal_results = run_calibration_metrics(args.model_name, args.adapter_label)

    # Save all
    output = {
        "model": args.model_name,
        "adapter_label": args.adapter_label,
        "retrained_probes": probe_results,
        "calibration_metrics": cal_results,
    }

    out_file = OUTPUT_DIR / f"reviewer_experiments_{args.adapter_label}_{args.model_name}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\n[save] {out_file}")


if __name__ == "__main__":
    main()
