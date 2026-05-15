"""Pre-Registration Compliance: E10 + meta-d'/M-ratio
====================================================

Two analyses required by the Phase 1 pre-registration that aren't
in the existing pipeline:

1. E10: AUROC₂ on answer-unchanged items (BINDING diagnostic).
   Pre-reg says: "if AUROC₂ does not improve on answer-unchanged
   items, H1 is interpreted as policy-shift artifact."

2. E3: meta-d' and M-ratio on smoothed distributions.
   Pre-reg says: compute these if distribution smooths (>30%
   intermediate items). Smoothing is confirmed for successful models.

Usage:
    python3 step_prereg_compliance.py \
        --model-name gemma-3-12b-it \
        --adapter-label probe_target

    # Also run for:
    #   --model-name gemma-3-12b-it --adapter-label probe_target_middle
    #   --model-name gemma-3-27b-it --adapter-label probe_target
    #   --model-name Qwen2.5-7B-Instruct-bf16 --adapter-label probe_target
    #   --model-name Meta-Llama-3.1-8B-Instruct-bf16 --adapter-label probe_target
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1"))
STEP4_DIR = Path(os.path.expanduser("~/jpwork/results/step4"))
OUTPUT_DIR = Path(os.path.expanduser("~/jpwork/results/prereg_compliance"))


def auroc2(confidence, correct):
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    if mask.sum() < 2 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))


def compute_meta_d_prime(confidence, correct):
    """Compute meta-d' and M-ratio using metadpy MLE.

    Uses the Galvin et al. (2003) retrospective Type-2 paradigm:
    - S1 = incorrect trials (error signal present)
    - S2 = correct trials (error signal absent)
    - Confidence ratings are the Type-2 judgment

    nR_S1 and nR_S2 are length-nRatings vectors (not 2×nRatings).

    Bins confidence into 4 quantile levels matching the programme's
    standard (Cacioli 2026, M1/M2 papers).
    """
    try:
        from metadpy import metad
    except ImportError:
        print("  [warn] metadpy not installed — skipping meta-d'/M-ratio")
        return None

    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]

    if len(c) < 20:
        return {"error": "too few items"}

    n_correct = int(y.sum())
    n_incorrect = int(len(y) - y.sum())
    if n_correct < 5 or n_incorrect < 5:
        return {"error": f"too few correct ({n_correct}) or incorrect ({n_incorrect})"}

    # Bin confidence into 4 levels using quartiles
    try:
        quartiles = np.percentile(c, [25, 50, 75])
    except Exception:
        return {"error": "cannot compute quartiles"}

    unique_quartiles = np.unique(quartiles)
    if len(unique_quartiles) < 2:
        return {"error": "degenerate confidence distribution"}

    # Digitize: bin 0 = lowest confidence, bin 3 = highest
    conf_bins = np.digitize(c, quartiles)  # 0, 1, 2, or 3

    # Build count arrays per Galvin et al. (2003) retrospective paradigm:
    # nR_S1 = confidence counts for INCORRECT trials (S1 = error present)
    # nR_S2 = confidence counts for CORRECT trials (S2 = error absent)
    # Ordered from lowest to highest confidence
    nRatings = 4
    nR_S1 = np.zeros(nRatings, dtype=int)  # incorrect trials
    nR_S2 = np.zeros(nRatings, dtype=int)  # correct trials

    for i in range(len(y)):
        b = min(conf_bins[i], nRatings - 1)
        if y[i] == 0:  # incorrect → S1
            nR_S1[b] += 1
        else:          # correct → S2
            nR_S2[b] += 1

    print(f"  nR_S1 (incorrect): {nR_S1.tolist()}")
    print(f"  nR_S2 (correct):   {nR_S2.tolist()}")

    try:
        result = metad(
            nR_S1=nR_S1.astype(float) + 0.5,
            nR_S2=nR_S2.astype(float) + 0.5,
            nRatings=2,
            padding=False,
        )

        # Extract values
        if hasattr(result, 'iloc'):
            d_prime = float(result["dprime"].iloc[0])
            meta_d_val = float(result["meta_d"].iloc[0])
        elif isinstance(result, dict):
            d_prime = float(result["dprime"])
            meta_d_val = float(result["meta_d"])
        else:
            d_prime = float(result["dprime"])
            meta_d_val = float(result["meta_d"])

        m_ratio = meta_d_val / d_prime if abs(d_prime) > 0.001 else float("nan")

        return {
            "d_prime": d_prime,
            "meta_d_prime": meta_d_val,
            "m_ratio": m_ratio,
            "n_items": int(len(y)),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_bins_used": int(len(np.unique(conf_bins))),
            "quartiles": quartiles.tolist(),
            "nR_S1": nR_S1.tolist(),
            "nR_S2": nR_S2.tolist(),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-reg compliance: E10 + meta-d'"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-label", required=True,
                        help="e.g. probe_target or probe_target_middle")
    args = parser.parse_args()

    model_name = args.model_name
    adapter_label = args.adapter_label

    print(f"{'='*60}")
    print(f"Pre-Registration Compliance Check")
    print(f"Model: {model_name}")
    print(f"Adapter: {adapter_label}")
    print(f"{'='*60}\n")

    # Load baseline responses
    baseline_file = STEP1_DIR / f"teval_responses_{model_name}.json"
    if not baseline_file.exists():
        print(f"[fatal] Baseline not found: {baseline_file}")
        sys.exit(1)

    with open(baseline_file) as f:
        baseline_records = json.load(f)

    # Load PT-CSFT responses from step4
    # step4 naming varies: try several patterns
    import glob
    candidates = []
    for pattern_name in [
        f"step4_{adapter_label}_teval_{model_name}.json",
        f"step4_{adapter_label}_{model_name}.json",
        f"step4_{adapter_label}_teval_{model_name}*.json",
        f"step4_{adapter_label}_{model_name}*.json",
    ]:
        found = glob.glob(str(STEP4_DIR / pattern_name))
        candidates.extend(found)

    # Filter out _metrics files
    candidates = [c for c in candidates if "_metrics" not in c]

    if not candidates:
        print(f"[fatal] PT-CSFT results not found in {STEP4_DIR}")
        print(f"  Looked for patterns with adapter_label={adapter_label}, model={model_name}")
        sys.exit(1)

    ft_file = Path(candidates[0])
    print(f"  PT-CSFT file: {ft_file.name}")

    with open(ft_file) as f:
        ft_data = json.load(f)

    # ft_data might be a dict with 'results' key or a list
    if isinstance(ft_data, dict):
        ft_records = ft_data.get("results", ft_data.get("records", []))
    else:
        ft_records = ft_data

    if not ft_records:
        print(f"[fatal] No records found in {ft_file}")
        sys.exit(1)

    # Build lookup by question_id
    # step1 uses "question_id" and "parsed_confidence"
    # step4 uses "qid" and "confidence"
    baseline_by_qid = {}
    for r in baseline_records:
        qid = r.get("question_id", r.get("qid"))
        baseline_by_qid[qid] = {
            "correct": r["correct"],
            "confidence": r.get("parsed_confidence", r.get("confidence")),
            "answer": r.get("parsed_answer", r.get("answer", "")),
        }

    ft_by_qid = {}
    for r in ft_records:
        qid = r.get("qid", r.get("question_id"))
        ft_by_qid[qid] = {
            "correct": r["correct"],
            "confidence": r.get("confidence", r.get("parsed_confidence")),
            "answer": r.get("answer", r.get("parsed_answer", "")),
        }

    # Find common items
    common_qids = set(baseline_by_qid.keys()) & set(ft_by_qid.keys())
    print(f"  Baseline items: {len(baseline_records)}")
    print(f"  PT-CSFT items:  {len(ft_records)}")
    print(f"  Common items:   {len(common_qids)}")

    # ================================================================
    # E10: Answer-Unchanged Items
    # ================================================================
    print(f"\n{'='*60}")
    print(f"E10: AUROC₂ on Answer-Unchanged Items")
    print(f"{'='*60}\n")

    unchanged_baseline_conf = []
    unchanged_baseline_correct = []
    unchanged_ft_conf = []
    unchanged_ft_correct = []

    changed_baseline_conf = []
    changed_baseline_correct = []
    changed_ft_conf = []
    changed_ft_correct = []

    for qid in common_qids:
        br = baseline_by_qid[qid]
        fr = ft_by_qid[qid]

        b_correct = br["correct"]
        f_correct = fr["correct"]
        b_conf = br["confidence"]
        f_conf = fr["confidence"]

        # Answer unchanged = same correctness outcome
        if b_correct == f_correct:
            if b_conf is not None:
                unchanged_baseline_conf.append(b_conf)
                unchanged_baseline_correct.append(int(b_correct))
            if f_conf is not None:
                unchanged_ft_conf.append(f_conf)
                unchanged_ft_correct.append(int(f_correct))
        else:
            if b_conf is not None:
                changed_baseline_conf.append(b_conf)
                changed_baseline_correct.append(int(b_correct))
            if f_conf is not None:
                changed_ft_conf.append(f_conf)
                changed_ft_correct.append(int(f_correct))

    n_unchanged = len(unchanged_baseline_conf)
    n_changed = len(changed_baseline_conf)
    print(f"  Answer-unchanged items: {n_unchanged}")
    print(f"  Answer-changed items:   {n_changed}")

    auroc_baseline_unchanged = auroc2(unchanged_baseline_conf, unchanged_baseline_correct)
    auroc_ft_unchanged = auroc2(unchanged_ft_conf, unchanged_ft_correct)
    auroc_baseline_all = auroc2(
        [baseline_by_qid[q]["confidence"] for q in common_qids],
        [int(baseline_by_qid[q]["correct"]) for q in common_qids]
    )
    auroc_ft_all = auroc2(
        [ft_by_qid[q]["confidence"] for q in common_qids],
        [int(ft_by_qid[q]["correct"]) for q in common_qids]
    )

    delta_all = auroc_ft_all - auroc_baseline_all if not (
        np.isnan(auroc_ft_all) or np.isnan(auroc_baseline_all)
    ) else float("nan")
    delta_unchanged = auroc_ft_unchanged - auroc_baseline_unchanged if not (
        np.isnan(auroc_ft_unchanged) or np.isnan(auroc_baseline_unchanged)
    ) else float("nan")

    print(f"\n  All items:")
    print(f"    Baseline AUROC₂: {auroc_baseline_all:.3f}")
    print(f"    PT-CSFT AUROC₂:  {auroc_ft_all:.3f}")
    print(f"    Delta:           {delta_all:+.3f}")

    print(f"\n  Answer-UNCHANGED items (E10):")
    print(f"    Baseline AUROC₂: {auroc_baseline_unchanged:.3f}")
    print(f"    PT-CSFT AUROC₂:  {auroc_ft_unchanged:.3f}")
    print(f"    Delta:           {delta_unchanged:+.3f}")

    if not np.isnan(delta_unchanged):
        if delta_unchanged > 0:
            print(f"\n  ✓ E10 PASSED: AUROC₂ improves on answer-unchanged items.")
            print(f"    H1 is NOT a policy-shift artifact.")
        else:
            print(f"\n  ✗ E10 FAILED: AUROC₂ does not improve on answer-unchanged items.")
            print(f"    Per pre-reg: H1 must be interpreted as policy-shift artifact.")

    # ================================================================
    # E3: meta-d' and M-ratio
    # ================================================================
    print(f"\n{'='*60}")
    print(f"E3: meta-d' / M-ratio (PT-CSFT distribution)")
    print(f"{'='*60}\n")

    ft_conf_all = [ft_by_qid[q]["confidence"] for q in common_qids
                   if ft_by_qid[q]["confidence"] is not None]
    ft_correct_all = [int(ft_by_qid[q]["correct"]) for q in common_qids
                      if ft_by_qid[q]["confidence"] is not None]

    # Check smoothing criterion
    conf_arr = np.array(ft_conf_all)
    intermediate = np.sum((conf_arr >= 20) & (conf_arr <= 80))
    pct_intermediate = intermediate / len(conf_arr) * 100
    print(f"  PT-CSFT confidence distribution:")
    print(f"    Mean: {np.mean(conf_arr):.1f}")
    print(f"    Std:  {np.std(conf_arr):.1f}")
    print(f"    Intermediate [20-80]: {intermediate}/{len(conf_arr)} ({pct_intermediate:.1f}%)")

    if pct_intermediate > 30:
        print(f"    → Distribution SMOOTHED (>30% intermediate). Computing meta-d'.\n")
        md_result = compute_meta_d_prime(ft_conf_all, ft_correct_all)
        if md_result and "error" not in md_result:
            print(f"  d':       {md_result['d_prime']:.3f}")
            print(f"  meta-d':  {md_result['meta_d_prime']:.3f}")
            print(f"  M-ratio:  {md_result['m_ratio']:.3f}")
            print(f"  n items:  {md_result['n_items']}")
            print(f"  bins used: {md_result['n_bins_used']}")
        elif md_result:
            print(f"  [error] {md_result['error']}")
    else:
        print(f"    → Distribution NOT smoothed (≤30% intermediate). meta-d' not applicable.")
        md_result = None

    # Also compute for baseline
    print(f"\n  Baseline confidence distribution:")
    base_conf = [baseline_by_qid[q]["confidence"] for q in common_qids
                 if baseline_by_qid[q]["confidence"] is not None]
    base_correct = [int(baseline_by_qid[q]["correct"]) for q in common_qids
                    if baseline_by_qid[q]["confidence"] is not None]
    base_arr = np.array(base_conf)
    base_intermediate = np.sum((base_arr >= 20) & (base_arr <= 80))
    base_pct = base_intermediate / len(base_arr) * 100
    print(f"    Mean: {np.mean(base_arr):.1f}")
    print(f"    Std:  {np.std(base_arr):.1f}")
    print(f"    Intermediate [20-80]: {base_intermediate}/{len(base_arr)} ({base_pct:.1f}%)")

    if base_pct > 30:
        print(f"    → Computing baseline meta-d'.\n")
        base_md = compute_meta_d_prime(base_conf, base_correct)
        if base_md and "error" not in base_md:
            print(f"  d':       {base_md['d_prime']:.3f}")
            print(f"  meta-d':  {base_md['meta_d_prime']:.3f}")
            print(f"  M-ratio:  {base_md['m_ratio']:.3f}")
    else:
        print(f"    → Baseline not smoothed. meta-d' not applicable (expected — ceiling prior).")

    # ================================================================
    # Save
    # ================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": model_name,
        "adapter_label": adapter_label,
        "e10": {
            "n_unchanged": n_unchanged,
            "n_changed": n_changed,
            "auroc_baseline_unchanged": auroc_baseline_unchanged,
            "auroc_ft_unchanged": auroc_ft_unchanged,
            "delta_unchanged": delta_unchanged,
            "auroc_baseline_all": auroc_baseline_all,
            "auroc_ft_all": auroc_ft_all,
            "delta_all": delta_all,
            "passed": bool(delta_unchanged > 0) if not np.isnan(delta_unchanged) else None,
        },
        "e3_meta_d": md_result if md_result else {"skipped": True},
        "distribution": {
            "ft_mean": float(np.mean(conf_arr)),
            "ft_std": float(np.std(conf_arr)),
            "ft_pct_intermediate": float(pct_intermediate),
        }
    }

    out_file = OUTPUT_DIR / f"prereg_compliance_{adapter_label}_{model_name}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n[save] {out_file}")

    print(f"\n{'='*60}")
    print(f"COMPLIANCE SUMMARY: {model_name} ({adapter_label})")
    print(f"{'='*60}")
    print(f"  E10 (answer-unchanged): {'PASSED' if output['e10']['passed'] else 'FAILED' if output['e10']['passed'] is not None else 'N/A'}")
    print(f"  E10 delta: {delta_unchanged:+.3f}" if not np.isnan(delta_unchanged) else "  E10 delta: nan")
    print(f"  E3 (meta-d'): {'computed' if md_result and 'error' not in (md_result or {}) else 'skipped/error'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
