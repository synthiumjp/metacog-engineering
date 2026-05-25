"""
bootstrap_e10_logit.py — Bootstrap CIs for the gentle E10 logit headline result.

Reads the 10 gentle E10 logit JSONs, computes:
  1. Per-seed AUROC₂ (sanity check against known values)
  2. Pooled bootstrap CI on the mean logit AUROC₂ across seeds (seed-level bootstrap)
  3. Per-seed item-level bootstrap CIs on logit AUROC₂
  4. Paired bootstrap: logit vs argmax per seed (Δ significance)
  5. Paired bootstrap: logit vs probe baseline (0.769)

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/bootstrap_e10_logit.py

Output:
    results_raw/domain_gen/e10_gentle/bootstrap_e10_logit_summary.json

Expects JSON files matching: e2e_ce_gentle_logit_seed{SEED}.json
Each JSON should contain a list of items with at minimum:
    - "correct" (bool or int)
    - "logit_confidence" or "logit_expected_value" (float, 0-1 or 0-100 scale)
Also reads matching text eval: e2e_ce_gentle_seed{SEED}.json for argmax pairing.
"""

import json, os, sys, glob
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1234, 5678, 9012, 3456, 7890, 2468]
N_BOOT = 10_000
RNG_SEED = 42
CI_LEVEL = 0.95
PROBE_AUROC2 = 0.769  # from paper, GSM8K probe at confidence position

RESULTS_DIR = Path("results_raw/domain_gen/e10_gentle")
OUTPUT_PATH = RESULTS_DIR / "bootstrap_e10_logit_summary.json"

# ---------------------------------------------------------------------------
# AUROC₂
# ---------------------------------------------------------------------------
def auroc2(confidence, correct):
    """AUROC₂ via sklearn, NaN-safe."""
    from sklearn.metrics import roc_auc_score
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    if mask.sum() < 2:
        return float("nan")
    if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))

# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------
def bootstrap_auroc2(confidence, correct, n_boot=N_BOOT, seed=RNG_SEED, ci=CI_LEVEL):
    """Item-level bootstrap CI on AUROC₂."""
    rng = np.random.default_rng(seed)
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    n = len(y)
    point = auroc2(c, y)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        boots.append(auroc2(c[idx], y[idx]))

    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    alpha = (1 - ci) / 2
    return {
        "point": point,
        "lo": float(np.quantile(boots, alpha)),
        "hi": float(np.quantile(boots, 1 - alpha)),
        "se": float(np.std(boots)),
        "n_valid": len(boots),
    }

def paired_bootstrap_delta(conf_a, conf_b, correct, n_boot=N_BOOT, seed=RNG_SEED, ci=CI_LEVEL):
    """Paired bootstrap CI on AUROC₂(a) - AUROC₂(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(conf_a, dtype=float)
    b = np.asarray(conf_b, dtype=float)
    y = np.asarray(correct, dtype=int)
    n = len(y)

    point_a = auroc2(a, y)
    point_b = auroc2(b, y)
    point_delta = point_a - point_b

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        da = auroc2(a[idx], y[idx])
        db = auroc2(b[idx], y[idx])
        if not np.isnan(da) and not np.isnan(db):
            deltas.append(da - db)

    deltas = np.array(deltas)
    alpha = (1 - ci) / 2
    return {
        "point_a": point_a,
        "point_b": point_b,
        "point_delta": point_delta,
        "lo": float(np.quantile(deltas, alpha)),
        "hi": float(np.quantile(deltas, 1 - alpha)),
        "p_positive": float(np.mean(deltas > 0)),
        "n_valid": len(deltas),
    }

# ---------------------------------------------------------------------------
# JSON loading with format auto-detection
# ---------------------------------------------------------------------------
def load_logit_json(path):
    """Load a logit eval JSON. Returns (logit_conf, correct) arrays.

    Handles multiple possible field names from eval_logit_general.py:
      - logit_confidence / logit_expected_value / expected_confidence
      - correct / is_correct
    """
    with open(path) as f:
        data = json.load(f)

    # Handle both list-of-items and dict-with-items formats
    if isinstance(data, dict):
        if "items" in data:
            items = data["items"]
        elif "results" in data:
            items = data["results"]
        else:
            # Try top-level keys
            print(f"  Warning: dict format, keys = {list(data.keys())[:10]}")
            print(f"  Looking for summary stats...")
            return None, None, data
    else:
        items = data

    # Detect field names from first item
    sample = items[0]
    logit_key = None
    for k in ["logit_confidence", "logit_expected_value", "expected_confidence",
              "logit_auroc2_confidence", "logit_ev"]:
        if k in sample:
            logit_key = k
            break

    correct_key = None
    for k in ["correct", "is_correct"]:
        if k in sample:
            correct_key = k
            break

    if logit_key is None or correct_key is None:
        print(f"  Warning: could not find logit/correct fields in {path}")
        print(f"  Available keys: {list(sample.keys())}")
        return None, None, None

    logit_conf = np.array([float(it[logit_key]) for it in items])
    correct = np.array([int(it[correct_key]) for it in items])

    # Auto-detect scale: if max > 1, assume 0-100, normalise to 0-1 for AUROC₂
    # (AUROC₂ is scale-invariant, but keep consistent)
    return logit_conf, correct, None

def load_text_json(path):
    """Load a text eval JSON. Returns (argmax_conf, correct) arrays."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get("items", data.get("results", []))
    else:
        items = data

    if not items:
        return None, None

    sample = items[0]
    conf_key = None
    for k in ["confidence", "parsed_confidence", "conf", "conf_mean"]:
        if k in sample:
            conf_key = k
            break

    correct_key = None
    for k in ["correct", "is_correct"]:
        if k in sample:
            correct_key = k
            break

    if conf_key is None or correct_key is None:
        return None, None

    conf = np.array([float(it[conf_key]) for it in items])
    correct = np.array([int(it[correct_key]) for it in items])
    return conf, correct

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Bootstrap CIs for Gentle E10 Logit Results")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} not found. Run from phase1/ root.")
        sys.exit(1)

    # --- 1. Load per-seed logit results ---
    seed_results = {}
    for seed in SEEDS:
        logit_path = RESULTS_DIR / f"e2e_ce_gentle_logit_seed{seed}.json"
        text_path = RESULTS_DIR / f"e2e_ce_gentle_seed{seed}.json"

        if not logit_path.exists():
            print(f"  Seed {seed}: logit JSON not found at {logit_path}")
            continue

        logit_conf, correct, summary = load_logit_json(logit_path)

        if logit_conf is None:
            # Summary-only format — extract AUROC₂ from summary
            if summary and "logit_auroc2" in summary:
                seed_results[seed] = {"logit_auroc2": summary["logit_auroc2"],
                                       "has_items": False}
                print(f"  Seed {seed}: summary-only, logit AUROC₂ = {summary['logit_auroc2']:.3f}")
            else:
                print(f"  Seed {seed}: could not load data")
            continue

        logit_auc = auroc2(logit_conf, correct)
        print(f"  Seed {seed}: logit AUROC₂ = {logit_auc:.3f} (n={len(correct)})")

        result = {
            "logit_auroc2": logit_auc,
            "has_items": True,
            "n_items": len(correct),
        }

        # Item-level bootstrap on logit
        boot = bootstrap_auroc2(logit_conf, correct)
        result["logit_bootstrap"] = boot
        print(f"    Bootstrap CI: [{boot['lo']:.3f}, {boot['hi']:.3f}]")

        # Paired bootstrap: logit vs argmax
        if text_path.exists():
            argmax_conf, argmax_correct = load_text_json(text_path)
            if argmax_conf is not None:
                argmax_auc = auroc2(argmax_conf, argmax_correct)
                result["argmax_auroc2"] = argmax_auc

                # Align items (should be same eval set)
                if len(argmax_conf) == len(logit_conf):
                    delta = paired_bootstrap_delta(logit_conf, argmax_conf, correct)
                    result["logit_vs_argmax"] = delta
                    print(f"    Logit vs argmax: Δ = {delta['point_delta']:+.3f} "
                          f"[{delta['lo']:+.3f}, {delta['hi']:+.3f}], "
                          f"P(Δ>0) = {delta['p_positive']:.3f}")
                else:
                    print(f"    Warning: item count mismatch (logit={len(logit_conf)}, "
                          f"argmax={len(argmax_conf)}), skipping paired test")

        seed_results[seed] = result

    if not seed_results:
        print("ERROR: No seed results loaded.")
        sys.exit(1)

    # --- 2. Seed-level bootstrap on mean AUROC₂ ---
    print("\n" + "=" * 60)
    print("Seed-level Bootstrap")
    print("=" * 60)

    seed_aucs = np.array([r["logit_auroc2"] for r in seed_results.values()])
    n_seeds = len(seed_aucs)
    mean_auc = np.mean(seed_aucs)
    std_auc = np.std(seed_aucs, ddof=1)

    print(f"  N seeds: {n_seeds}")
    print(f"  Mean logit AUROC₂: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  Range: [{np.min(seed_aucs):.3f}, {np.max(seed_aucs):.3f}]")
    print(f"  All above 0.85: {np.all(seed_aucs > 0.85)}")

    # Bootstrap the mean
    rng = np.random.default_rng(RNG_SEED)
    boot_means = np.array([
        np.mean(rng.choice(seed_aucs, size=n_seeds, replace=True))
        for _ in range(N_BOOT)
    ])
    alpha = (1 - CI_LEVEL) / 2
    seed_boot = {
        "mean": float(mean_auc),
        "std": float(std_auc),
        "se_mean": float(std_auc / np.sqrt(n_seeds)),
        "lo": float(np.quantile(boot_means, alpha)),
        "hi": float(np.quantile(boot_means, 1 - alpha)),
        "boot_std": float(np.std(boot_means)),
        "n_seeds": n_seeds,
        "all_above_085": bool(np.all(seed_aucs > 0.85)),
    }
    print(f"  Bootstrap CI on mean: [{seed_boot['lo']:.3f}, {seed_boot['hi']:.3f}]")

    # --- 3. Mean logit vs probe (0.769) ---
    print(f"\n  Mean logit vs probe ({PROBE_AUROC2}):")
    deltas_vs_probe = seed_aucs - PROBE_AUROC2
    boot_deltas_probe = boot_means - PROBE_AUROC2
    probe_comparison = {
        "mean_delta": float(np.mean(deltas_vs_probe)),
        "lo": float(np.quantile(boot_deltas_probe, alpha)),
        "hi": float(np.quantile(boot_deltas_probe, 1 - alpha)),
        "all_seeds_exceed_probe": bool(np.all(seed_aucs > PROBE_AUROC2)),
        "p_exceeds_probe": float(np.mean(boot_means > PROBE_AUROC2)),
    }
    print(f"    Mean Δ: {probe_comparison['mean_delta']:+.3f} "
          f"[{probe_comparison['lo']:+.3f}, {probe_comparison['hi']:+.3f}]")
    print(f"    All seeds exceed probe: {probe_comparison['all_seeds_exceed_probe']}")
    print(f"    P(mean > probe): {probe_comparison['p_exceeds_probe']:.4f}")

    # --- 4. Pooled item-level analysis (if items available) ---
    seeds_with_items = {s: r for s, r in seed_results.items() if r.get("has_items")}
    pooled_result = None
    if seeds_with_items:
        print(f"\n  Pooled item-level analysis ({len(seeds_with_items)} seeds with items)")

        # Note: pooling items across seeds is informative but items are not independent
        # The correct analysis is the seed-level bootstrap above
        # This is supplementary
        all_logit = []
        all_correct = []
        for seed in sorted(seeds_with_items.keys()):
            logit_path = RESULTS_DIR / f"e2e_ce_gentle_logit_seed{seed}.json"
            lc, co, _ = load_logit_json(logit_path)
            if lc is not None:
                all_logit.extend(lc.tolist())
                all_correct.extend(co.tolist())

        all_logit = np.array(all_logit)
        all_correct = np.array(all_correct)
        pooled_auc = auroc2(all_logit, all_correct)
        pooled_boot = bootstrap_auroc2(all_logit, all_correct)
        pooled_result = {
            "pooled_auroc2": pooled_auc,
            "n_items_total": len(all_correct),
            "bootstrap": pooled_boot,
            "note": "Items pooled across seeds; not independent. Seed-level bootstrap is primary."
        }
        print(f"    Pooled AUROC₂: {pooled_auc:.3f} [{pooled_boot['lo']:.3f}, {pooled_boot['hi']:.3f}]")

    # --- 5. Summary ---
    summary = {
        "description": "Bootstrap CIs for Gentle E10 logit AUROC₂ (10 seeds)",
        "n_boot": N_BOOT,
        "ci_level": CI_LEVEL,
        "probe_baseline": PROBE_AUROC2,
        "per_seed": {str(s): r for s, r in seed_results.items()},
        "seed_level_bootstrap": seed_boot,
        "vs_probe": probe_comparison,
    }
    if pooled_result:
        summary["pooled_item_level"] = pooled_result

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    # --- 6. Paper-ready summary ---
    print("\n" + "=" * 60)
    print("Paper-ready numbers:")
    print("=" * 60)
    print(f"  Logit AUROC₂ = {mean_auc:.3f} ± {std_auc:.3f} "
          f"(95% CI [{seed_boot['lo']:.3f}, {seed_boot['hi']:.3f}])")
    print(f"  All 10 seeds > 0.85: {seed_boot['all_above_085']}")
    print(f"  Exceeds probe ({PROBE_AUROC2}) by {probe_comparison['mean_delta']:+.3f} "
          f"[{probe_comparison['lo']:+.3f}, {probe_comparison['hi']:+.3f}]")
    print(f"  P(mean > probe) = {probe_comparison['p_exceeds_probe']:.4f}")

    # Per-seed table
    print("\n  Per-seed item-level CIs:")
    print(f"  {'Seed':>6}  {'Logit':>6}  {'95% CI':>15}  {'Argmax':>7}  {'Δ':>7}  {'Δ CI':>15}")
    for seed in SEEDS:
        s = str(seed)
        if s not in summary["per_seed"]:
            continue
        r = summary["per_seed"][s]
        logit = r["logit_auroc2"]
        if r.get("has_items") and "logit_bootstrap" in r:
            ci_str = f"[{r['logit_bootstrap']['lo']:.3f}, {r['logit_bootstrap']['hi']:.3f}]"
        else:
            ci_str = "n/a"
        argmax = r.get("argmax_auroc2", float("nan"))
        delta = logit - argmax if not np.isnan(argmax) else float("nan")
        delta_ci = ""
        if "logit_vs_argmax" in r:
            d = r["logit_vs_argmax"]
            delta_ci = f"[{d['lo']:+.3f}, {d['hi']:+.3f}]"
        print(f"  {seed:>6}  {logit:.3f}  {ci_str:>15}  {argmax:>7.3f}  {delta:>+7.3f}  {delta_ci:>15}")


if __name__ == "__main__":
    main()
