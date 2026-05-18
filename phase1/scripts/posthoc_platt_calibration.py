"""
posthoc_platt_calibration.py — Post-hoc Platt scaling for models where the
probe gap is near zero (e.g. Qwen 72B). Decompresses verbal confidence range
without fine-tuning.

Usage:
    python3 posthoc_platt_calibration.py \
        --model-name Qwen2.5-72B-Instruct-bf16 \
        --results-dir ~/jpwork/metacog-engineering/phase1/results_raw/step1

When to use: when the probe gate check shows probe AUROC₂ ≈ verbal AUROC₂
(no gap to close with PT-CSFT). The model already discriminates correct from
incorrect via verbal confidence, but compresses the signal into a narrow
range near the ceiling (e.g. 94-100%). Platt scaling decompresses the range
while preserving discrimination (AUROC₂ is rank-based, unaffected by
monotonic transforms).

Fits on T-cal, evaluates on T-eval. Outputs JSON with metrics and
calibrated confidence values.
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def compute_vrs(confidences, corrects):
    """Simplified VRS screening: ceiling rate L, TRIN, r(conf, correct)."""
    L = (confidences >= 0.95).mean()
    # TRIN: proportion at the single most common value
    vals, counts = np.unique(np.round(confidences, 2), return_counts=True)
    TRIN = counts.max() / len(confidences)
    # Point-biserial correlation
    if corrects.std() > 0 and confidences.std() > 0:
        r = np.corrcoef(confidences, corrects)[0, 1]
    else:
        r = 0.0
    # Classification
    if L > 0.90 or TRIN > 0.90 or abs(r) < 0.10:
        tier = "Invalid"
    elif L > 0.50 or TRIN > 0.50 or abs(r) < 0.30:
        tier = "Indeterminate"
    else:
        tier = "Valid"
    return {"L": round(L, 3), "TRIN": round(TRIN, 3), "r": round(r, 3), "tier": tier}


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc Platt/isotonic calibration on verbal confidence")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing step1 output files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results-dir/../posthoc)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.results_dir), "posthoc")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    tcal_path = os.path.join(args.results_dir,
                             f"tcal_greedy_responses_{args.model_name}.json")
    teval_path = os.path.join(args.results_dir,
                              f"teval_responses_{args.model_name}.json")

    print(f"Loading T-cal: {tcal_path}")
    tcal = json.load(open(tcal_path))
    print(f"Loading T-eval: {teval_path}")
    teval = json.load(open(teval_path))

    def extract(items):
        pairs = [(float(r["parsed_confidence"]), int(r["correct"]))
                 for r in items if r["parsed_confidence"] is not None]
        confs = np.array([p[0] for p in pairs])
        correct = np.array([p[1] for p in pairs])
        mask = ~np.isnan(confs)
        return confs[mask] / 100.0, correct[mask]

    cal_conf, cal_correct = extract(tcal)
    eval_conf, eval_correct = extract(teval)

    print(f"T-cal: {len(cal_conf)} items, acc={cal_correct.mean():.3f}")
    print(f"T-eval: {len(eval_conf)} items, acc={eval_correct.mean():.3f}")

    # --- Baseline ---
    baseline_auroc = roc_auc_score(eval_correct, eval_conf)
    baseline_ece = compute_ece(eval_correct, eval_conf)
    baseline_brier = brier_score_loss(eval_correct, eval_conf)
    baseline_vrs = compute_vrs(eval_conf, eval_correct)

    print(f"\n{'='*60}")
    print(f"  BASELINE: {args.model_name}")
    print(f"{'='*60}")
    print(f"  AUROC₂:     {baseline_auroc:.3f}")
    print(f"  ECE:        {baseline_ece:.3f}")
    print(f"  Brier:      {baseline_brier:.3f}")
    print(f"  Conf mean:  {eval_conf.mean()*100:.1f}, std: {eval_conf.std()*100:.1f}")
    print(f"  VRS:        {baseline_vrs['tier']} (L={baseline_vrs['L']}, "
          f"TRIN={baseline_vrs['TRIN']}, r={baseline_vrs['r']})")

    results = {
        "model": args.model_name,
        "n_cal": len(cal_conf),
        "n_eval": len(eval_conf),
        "baseline": {
            "auroc2": round(baseline_auroc, 3),
            "ece": round(baseline_ece, 3),
            "brier": round(baseline_brier, 3),
            "conf_mean": round(eval_conf.mean() * 100, 1),
            "conf_std": round(eval_conf.std() * 100, 1),
            "vrs": baseline_vrs,
        },
    }

    # --- Platt scaling ---
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(cal_conf.reshape(-1, 1), cal_correct)
    eval_platt = platt.predict_proba(eval_conf.reshape(-1, 1))[:, 1]

    platt_auroc = roc_auc_score(eval_correct, eval_platt)
    platt_ece = compute_ece(eval_correct, eval_platt)
    platt_brier = brier_score_loss(eval_correct, eval_platt)
    platt_vrs = compute_vrs(eval_platt, eval_correct)

    print(f"\n{'='*60}")
    print(f"  PLATT SCALING")
    print(f"{'='*60}")
    print(f"  AUROC₂:     {platt_auroc:.3f}")
    print(f"  ECE:        {platt_ece:.3f}  (delta: {platt_ece - baseline_ece:+.3f})")
    print(f"  Brier:      {platt_brier:.3f}  (delta: {platt_brier - baseline_brier:+.3f})")
    print(f"  Conf mean:  {eval_platt.mean()*100:.1f}, std: {eval_platt.std()*100:.1f}")
    print(f"  Conf range: [{eval_platt.min()*100:.1f}, {eval_platt.max()*100:.1f}]")
    print(f"  VRS:        {platt_vrs['tier']} (L={platt_vrs['L']}, "
          f"TRIN={platt_vrs['TRIN']}, r={platt_vrs['r']})")

    results["platt"] = {
        "auroc2": round(platt_auroc, 3),
        "ece": round(platt_ece, 3),
        "brier": round(platt_brier, 3),
        "conf_mean": round(eval_platt.mean() * 100, 1),
        "conf_std": round(eval_platt.std() * 100, 1),
        "conf_min": round(eval_platt.min() * 100, 1),
        "conf_max": round(eval_platt.max() * 100, 1),
        "vrs": platt_vrs,
        "coef": round(float(platt.coef_[0][0]), 4),
        "intercept": round(float(platt.intercept_[0]), 4),
    }

    # --- Isotonic regression ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_conf, cal_correct)
    eval_iso = iso.predict(eval_conf)

    iso_auroc = roc_auc_score(eval_correct, eval_iso)
    iso_ece = compute_ece(eval_correct, eval_iso)
    iso_brier = brier_score_loss(eval_correct, eval_iso)
    iso_vrs = compute_vrs(eval_iso, eval_correct)

    print(f"\n{'='*60}")
    print(f"  ISOTONIC REGRESSION")
    print(f"{'='*60}")
    print(f"  AUROC₂:     {iso_auroc:.3f}")
    print(f"  ECE:        {iso_ece:.3f}  (delta: {iso_ece - baseline_ece:+.3f})")
    print(f"  Brier:      {iso_brier:.3f}  (delta: {iso_brier - baseline_brier:+.3f})")
    print(f"  Conf mean:  {eval_iso.mean()*100:.1f}, std: {eval_iso.std()*100:.1f}")
    print(f"  Conf range: [{eval_iso.min()*100:.1f}, {eval_iso.max()*100:.1f}]")
    print(f"  VRS:        {iso_vrs['tier']} (L={iso_vrs['L']}, "
          f"TRIN={iso_vrs['TRIN']}, r={iso_vrs['r']})")

    results["isotonic"] = {
        "auroc2": round(iso_auroc, 3),
        "ece": round(iso_ece, 3),
        "brier": round(iso_brier, 3),
        "conf_mean": round(eval_iso.mean() * 100, 1),
        "conf_std": round(eval_iso.std() * 100, 1),
        "conf_min": round(eval_iso.min() * 100, 1),
        "conf_max": round(eval_iso.max() * 100, 1),
        "vrs": iso_vrs,
    }

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {args.model_name}")
    print(f"{'='*60}")
    print(f"  {'Method':<12} {'AUROC₂':>8} {'ECE':>8} {'Brier':>8} {'VRS':>14}")
    print(f"  {'Baseline':<12} {baseline_auroc:>8.3f} {baseline_ece:>8.3f} "
          f"{baseline_brier:>8.3f} {baseline_vrs['tier']:>14}")
    print(f"  {'Platt':<12} {platt_auroc:>8.3f} {platt_ece:>8.3f} "
          f"{platt_brier:>8.3f} {platt_vrs['tier']:>14}")
    print(f"  {'Isotonic':<12} {iso_auroc:>8.3f} {iso_ece:>8.3f} "
          f"{iso_brier:>8.3f} {iso_vrs['tier']:>14}")

    # Save
    outpath = os.path.join(args.output_dir,
                           f"posthoc_calibration_{args.model_name}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {outpath}")


if __name__ == "__main__":
    main()
