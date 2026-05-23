#!/usr/bin/env python3
"""
bootstrap_auroc2.py — Compute bootstrap 95% CIs on AUROC₂ for all results.

Takes response JSON files with per-item correctness and confidence,
computes AUROC₂ with bootstrap CIs, and optionally compares conditions
(paired bootstrap on the difference).

Usage:
    # Single file
    python3 bootstrap_auroc2.py --files result1.json

    # Compare two conditions (paired by item ID)
    python3 bootstrap_auroc2.py --files baseline.json ptcsft.json --paired

    # Batch: all results
    python3 bootstrap_auroc2.py --batch
"""

import argparse
import json
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 42
N_BOOT = 10000


def load_responses(path):
    """Load per-item responses. Returns dict: id → (correct, confidence)."""
    with open(path) as f:
        data = json.load(f)

    items = {}
    if isinstance(data, list):
        for r in data:
            rid = r.get("id", r.get("question_id", r.get("qid", "")))
            correct = r.get("correct", r.get("is_correct", False))
            if isinstance(correct, str):
                correct = correct.lower() == "true"
            conf = r.get("confidence", r.get("parsed_confidence", float("nan")))
            if conf is not None and not (isinstance(conf, float) and np.isnan(conf)):
                items[rid] = (bool(correct), float(conf))
    return items


def bootstrap_auroc2(correct, confidence, n_boot=N_BOOT, seed=SEED):
    """Compute AUROC₂ with bootstrap 95% CI.

    Returns: (auroc2, ci_lo, ci_hi, n)
    """
    correct = np.array(correct, dtype=int)
    confidence = np.array(confidence, dtype=float)

    # Remove NaN
    valid = ~np.isnan(confidence)
    correct = correct[valid]
    confidence = confidence[valid]
    n = len(correct)

    if n < 10 or len(np.unique(correct)) < 2:
        return float("nan"), float("nan"), float("nan"), n

    # Point estimate
    auroc2 = roc_auc_score(correct, confidence)

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bc = correct[idx]
        bf = confidence[idx]
        if len(np.unique(bc)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(bc, bf))

    boot_aurocs = np.array(boot_aurocs)
    ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5])

    return auroc2, ci_lo, ci_hi, n


def paired_bootstrap_diff(correct, conf_a, conf_b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI on AUROC₂(B) - AUROC₂(A) using paired resampling.

    All arrays must be aligned (same items in same order).
    Returns: (diff, ci_lo, ci_hi)
    """
    correct = np.array(correct, dtype=int)
    conf_a = np.array(conf_a, dtype=float)
    conf_b = np.array(conf_b, dtype=float)
    n = len(correct)

    # Point estimate
    if len(np.unique(correct)) < 2:
        return float("nan"), float("nan"), float("nan")
    auroc_a = roc_auc_score(correct, conf_a)
    auroc_b = roc_auc_score(correct, conf_b)
    diff = auroc_b - auroc_a

    # Paired bootstrap
    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bc = correct[idx]
        if len(np.unique(bc)) < 2:
            continue
        ba = roc_auc_score(bc, conf_a[idx])
        bb = roc_auc_score(bc, conf_b[idx])
        boot_diffs.append(bb - ba)

    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    return diff, ci_lo, ci_hi


def batch_analysis():
    """Run bootstrap CIs on all known result files."""
    base = Path.home() / "jpwork/metacog-engineering/phase1/results_raw"

    print("=" * 70)
    print("Bootstrap 95% CIs on AUROC₂ (N=10,000 resamples)")
    print("=" * 70)

    # ── 1. Primary PT-CSFT results (TriviaQA, 8 models) ──
    print("\n--- Primary PT-CSFT Results (TriviaQA) ---\n")

    # Find baseline and PT-CSFT response files
    step4_dir = base / "step4"
    step1_dir = base / "step1"

    # Try to find eval response files
    for pattern_name, search_dirs in [
        ("ablation_*_responses.json", [step4_dir]),
        ("eval_*_responses.json", [step4_dir]),
    ]:
        for d in search_dirs:
            if d.exists():
                for f in sorted(d.glob(pattern_name)):
                    items = load_responses(f)
                    if not items:
                        continue
                    correct = [v[0] for v in items.values()]
                    conf = [v[1] for v in items.values()]
                    auroc, lo, hi, n = bootstrap_auroc2(correct, conf)
                    name = f.stem.replace("_responses", "")
                    if not np.isnan(auroc):
                        print(f"  {name:50s}  AUROC₂={auroc:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")

    # ── 2. Domain gen results (GSM8K) ──
    print("\n--- Domain Generalisation (GSM8K, Gemma 12B) ---\n")

    domain_dir = base / "domain_gen"
    if domain_dir.exists():
        for f in sorted(domain_dir.glob("*_responses.json")):
            items = load_responses(f)
            if not items:
                continue
            correct = [v[0] for v in items.values()]
            conf = [v[1] for v in items.values()]
            auroc, lo, hi, n = bootstrap_auroc2(correct, conf)
            name = f.stem.replace("_responses", "")
            if not np.isnan(auroc):
                print(f"  {name:50s}  AUROC₂={auroc:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")

    # ── 3. Paired comparisons (GSM8K: baseline vs end-to-end) ──
    print("\n--- Paired Comparisons (GSM8K) ---\n")

    baseline_path = domain_dir / "responses_gsm8k_gemma-3-12b-it.json"
    e2e_ce_path = domain_dir / "e2e_ce_gsm8k_gemma-3-12b-it_responses.json"

    if baseline_path.exists() and e2e_ce_path.exists():
        baseline_items = load_responses(baseline_path)
        e2e_items = load_responses(e2e_ce_path)

        # Align by ID
        common_ids = sorted(set(baseline_items.keys()) & set(e2e_items.keys()))
        if len(common_ids) > 50:
            correct = [baseline_items[i][0] for i in common_ids]
            conf_base = [baseline_items[i][1] for i in common_ids]
            conf_e2e = [e2e_items[i][1] for i in common_ids]

            diff, lo, hi = paired_bootstrap_diff(correct, conf_base, conf_e2e)
            sig = "***" if lo > 0 else ("n.s." if lo <= 0 <= hi else "")
            print(f"  E2E CE binary vs Baseline:  Δ={diff:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")
            print(f"    (n={len(common_ids)} paired items)")
        else:
            print(f"  Only {len(common_ids)} common items — eval sets may differ")
            print(f"  Baseline has {len(baseline_items)} items (cal+eval)")
            print(f"  E2E CE has {len(e2e_items)} items (eval only)")
            print(f"  Computing unpaired CIs instead:")

            # Unpaired: just report CIs for each
            for name, items in [("Baseline", baseline_items), ("E2E CE", e2e_items)]:
                correct = [v[0] for v in items.values()]
                conf = [v[1] for v in items.values()]
                auroc, lo, hi, n = bootstrap_auroc2(correct, conf)
                print(f"    {name:20s}  AUROC₂={auroc:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")

    # ── 4. Post-hoc calibration results ──
    print("\n--- Post-hoc Calibration ---\n")
    posthoc_dir = base / "posthoc"
    if posthoc_dir.exists():
        for f in sorted(posthoc_dir.glob("*_responses.json")):
            items = load_responses(f)
            if not items:
                continue
            correct = [v[0] for v in items.values()]
            conf = [v[1] for v in items.values()]
            auroc, lo, hi, n = bootstrap_auroc2(correct, conf)
            name = f.stem.replace("_responses", "")
            if not np.isnan(auroc):
                print(f"  {name:50s}  AUROC₂={auroc:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")

    print("\n" + "=" * 70)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path,
                        help="Response JSON files to analyse")
    parser.add_argument("--paired", action="store_true",
                        help="Paired comparison between two files")
    parser.add_argument("--batch", action="store_true",
                        help="Run on all known result files")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    if args.batch:
        batch_analysis()
        return

    if not args.files:
        parser.error("Provide --files or --batch")


    if len(args.files) == 1:
        items = load_responses(args.files[0])
        correct = [v[0] for v in items.values()]
        conf = [v[1] for v in items.values()]
        auroc, lo, hi, n = bootstrap_auroc2(correct, conf)
        print(f"AUROC₂ = {auroc:.3f} [{lo:.3f}, {hi:.3f}]  (n={n})")

    elif len(args.files) == 2 and args.paired:
        items_a = load_responses(args.files[0])
        items_b = load_responses(args.files[1])
        common = sorted(set(items_a.keys()) & set(items_b.keys()))
        print(f"Paired comparison: {len(common)} common items")

        correct = [items_a[i][0] for i in common]
        conf_a = [items_a[i][1] for i in common]
        conf_b = [items_b[i][1] for i in common]

        auroc_a, lo_a, hi_a, _ = bootstrap_auroc2(correct, conf_a)
        auroc_b, lo_b, hi_b, _ = bootstrap_auroc2(correct, conf_b)
        diff, dlo, dhi = paired_bootstrap_diff(correct, conf_a, conf_b)

        print(f"  A ({args.files[0].stem}): {auroc_a:.3f} [{lo_a:.3f}, {hi_a:.3f}]")
        print(f"  B ({args.files[1].stem}): {auroc_b:.3f} [{lo_b:.3f}, {hi_b:.3f}]")
        print(f"  Δ(B-A): {diff:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")

    else:
        for f in args.files:
            items = load_responses(f)
            correct = [v[0] for v in items.values()]
            conf = [v[1] for v in items.values()]
            auroc, lo, hi, n = bootstrap_auroc2(correct, conf)
            print(f"  {f.stem:50s}  AUROC₂={auroc:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")


if __name__ == "__main__":
    main()
