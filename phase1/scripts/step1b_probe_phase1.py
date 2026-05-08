"""
Step 1b: Linear Probe Control (Phase 1 — MLX .npz format)
==========================================================

Fits L2-regularised logistic regression probes on cached T-cal hidden states
predicting correctness; applies each probe to T-eval; computes AUROC₂ for
each probe configuration.

Primary probe: last layer, last-answer-token position.
E6 sensitivity: {first, middle, last} × {pre_answer_token, last_answer_token}
  -> 6 configurations total.

Computes:
  - AUROC₂_probe per config on T-eval
  - AUROC₂_probe - AUROC₂_baseline_verbal  (does probe beat verbal?)
  - AUROC₂_probe - chance (0.5)             (does probe find signal at all?)
  (The third, AUROC₂_ft - AUROC₂_probe, is computed in Step 4 after SFT.)

Inputs (from Step 1):
    ~/jpwork/results/step1/hidden_states_tcal_{model_name}.npz
    ~/jpwork/results/step1/hidden_states_teval_{model_name}.npz
    ~/jpwork/results/step1/tcal_greedy_responses_{model_name}.json
    ~/jpwork/results/step1/teval_responses_{model_name}.json

Outputs:
    ~/jpwork/results/probe/probe_metrics_{model_name}.json
    ~/jpwork/results/probe/probe_scores_teval_{model_name}.json
    ~/jpwork/results/probe/probe_fits_{model_name}.npz

Runtime: ~15-30 min per model (CPU-only, no GPU needed).

Usage:
    python3 step1b_probe_phase1.py --model-name gemma-3-12b-it
    python3 step1b_probe_phase1.py --model-name gemma-3-27b-it
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config (locked per pre-reg)
# ---------------------------------------------------------------------------

STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1"))
PROBE_DIR = Path(os.path.expanduser("~/jpwork/results/probe"))

PROBE_LAYERS = ["first", "middle", "last"]
PROBE_POSITIONS = ["pre_answer_token", "last_answer_token"]
PRIMARY_CONFIG = ("last", "last_answer_token")

LR_CV_FOLDS = 5
LR_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# AUROC₂
# ---------------------------------------------------------------------------

def auroc2(confidence: np.ndarray, correct: np.ndarray) -> float:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    if mask.sum() < 2:
        return float("nan")
    if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))


def paired_bootstrap_auroc2_delta(
    confidence_a: np.ndarray,
    confidence_b: np.ndarray,
    correct: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    rng = np.random.default_rng(seed)
    a = np.asarray(confidence_a, dtype=float)
    b = np.asarray(confidence_b, dtype=float)
    y = np.asarray(correct, dtype=int)
    n = len(y)

    point_a = auroc2(a, y)
    point_b = auroc2(b, y)
    point_delta = point_a - point_b

    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ya = y[idx]
        if ya.sum() == 0 or ya.sum() == n:
            deltas[i] = np.nan
            continue
        deltas[i] = auroc2(a[idx], ya) - auroc2(b[idx], ya)

    deltas = deltas[~np.isnan(deltas)]
    alpha = (1 - ci) / 2
    return {
        "point_a": point_a,
        "point_b": point_b,
        "point_delta": point_delta,
        "lo": float(np.quantile(deltas, alpha)),
        "hi": float(np.quantile(deltas, 1 - alpha)),
        "ci": ci,
        "n_resamples": n_resamples,
        "n_valid": len(deltas),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# .npz hidden state loader
# ---------------------------------------------------------------------------

def load_hidden_states_and_labels(
    npz_path: Path,
    responses_path: Path,
    layer: str,
    position: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load hidden states + correctness labels for a given layer/position.

    Phase 1 .npz format: keys are "{qid}__{layer}__{position}"
    e.g. "tc-123__last__last_answer_token"

    Returns (X, y, qids).
    """
    npz = np.load(npz_path)
    with open(responses_path) as f:
        records = json.load(f)
    correct_by_qid = {r["question_id"]: int(r["correct"]) for r in records}

    # Build suffix to match
    suffix = f"__{layer}__{position}"

    X_list, y_list, qids = [], [], []
    for key in npz.files:
        if not key.endswith(suffix):
            continue
        # Extract qid: everything before the first "__"
        qid = key[:key.index("__")]
        if qid not in correct_by_qid:
            continue
        X_list.append(npz[key])
        y_list.append(correct_by_qid[qid])
        qids.append(qid)

    if not X_list:
        raise RuntimeError(
            f"No items with hidden states at layer={layer}, position={position}. "
            f"Check npz keys. Sample keys: {list(npz.files)[:5]}"
        )
    return np.stack(X_list, axis=0), np.array(y_list, dtype=int), qids


def load_verbal_confidence_on_teval(
    responses_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load T-eval verbal confidence + correctness aligned by qid."""
    with open(responses_path) as f:
        records = json.load(f)
    confidences, correct, qids = [], [], []
    for r in records:
        conf = r.get("parsed_confidence")
        if conf is None or (isinstance(conf, float) and np.isnan(conf)):
            continue
        confidences.append(float(conf))
        correct.append(int(r["correct"]))
        qids.append(r["question_id"])
    return (
        np.array(confidences, dtype=float),
        np.array(correct, dtype=int),
        qids,
    )


# ---------------------------------------------------------------------------
# Probe fit + eval
# ---------------------------------------------------------------------------

def fit_probe(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Fit L2-regularised logistic regression with CV over C."""
    scaler = StandardScaler().fit(X_train)
    X_s = scaler.transform(X_train)

    if len(np.unique(y_train)) < 2:
        return {
            "degenerate": True,
            "mean_label": float(np.mean(y_train)),
            "n_train": len(y_train),
        }

    clf = LogisticRegressionCV(
        Cs=LR_CS,
        cv=LR_CV_FOLDS,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=BOOTSTRAP_SEED,
    )
    clf.fit(X_s, y_train)

    return {
        "degenerate": False,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "coef": clf.coef_.astype(np.float32),
        "intercept": clf.intercept_.astype(np.float32),
        "best_C": float(clf.C_[0]),
        "n_train": len(y_train),
    }


def predict_probe(probe: dict, X: np.ndarray) -> np.ndarray:
    """Apply a fitted probe to new data. Returns per-item P(correct)."""
    if probe.get("degenerate", False):
        return np.full(len(X), probe["mean_label"], dtype=float)
    X_s = (X - probe["scaler_mean"]) / probe["scaler_scale"]
    logits = X_s @ probe["coef"].T + probe["intercept"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs[:, 0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1b: Linear probe on hidden states")
    parser.add_argument("--model-name", required=True,
                        help="Model name (e.g. gemma-3-12b-it)")
    args = parser.parse_args()

    model_name = args.model_name
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    # Input paths
    tcal_npz = STEP1_DIR / f"hidden_states_tcal_{model_name}.npz"
    teval_npz = STEP1_DIR / f"hidden_states_teval_{model_name}.npz"
    tcal_resp = STEP1_DIR / f"tcal_greedy_responses_{model_name}.json"
    teval_resp = STEP1_DIR / f"teval_responses_{model_name}.json"

    for p in [tcal_npz, teval_npz, tcal_resp, teval_resp]:
        if not p.exists():
            print(f"[fatal] missing input: {p}")
            sys.exit(2)

    # Peek at npz keys to verify format
    npz_peek = np.load(tcal_npz)
    sample_keys = list(npz_peek.files)[:5]
    print(f"[data] T-cal npz: {len(npz_peek.files)} keys")
    print(f"       Sample keys: {sample_keys}")
    npz_peek.close()

    # Load T-eval verbal confidence
    teval_verbal_conf, teval_verbal_correct, teval_verbal_qids = \
        load_verbal_confidence_on_teval(teval_resp)
    print(f"[data] T-eval verbal: {len(teval_verbal_qids)} items with parseable confidence\n")

    all_probe_results = {}
    all_teval_scores = {}
    all_probe_fits = {}
    start = time.time()

    for layer in PROBE_LAYERS:
        for position in PROBE_POSITIONS:
            config_key = f"{layer}_{position}"
            is_primary = (layer, position) == PRIMARY_CONFIG
            tag = "PRIMARY" if is_primary else "sensitivity"

            print(f"=== Probe config: {config_key} ({tag}) ===")

            # Load training data (T-cal)
            try:
                X_train, y_train, train_qids = load_hidden_states_and_labels(
                    tcal_npz, tcal_resp, layer, position
                )
            except RuntimeError as e:
                print(f"[skip] {e}")
                all_probe_results[config_key] = {
                    "layer": layer, "position": position,
                    "is_primary": is_primary, "skipped": True,
                    "reason": str(e),
                }
                continue

            # Load eval data (T-eval)
            try:
                X_eval, y_eval, eval_qids = load_hidden_states_and_labels(
                    teval_npz, teval_resp, layer, position
                )
            except RuntimeError as e:
                print(f"[skip] {e}")
                all_probe_results[config_key] = {
                    "layer": layer, "position": position,
                    "is_primary": is_primary, "skipped": True,
                    "reason": str(e),
                }
                continue

            print(f"[fit]  n_train={len(X_train)}, dim={X_train.shape[1]}, "
                  f"pos_rate_train={y_train.mean():.3f}")

            probe = fit_probe(X_train, y_train)
            if probe.get("degenerate", False):
                print("[skip] degenerate labels in T-cal; cannot fit probe")
                all_probe_results[config_key] = {
                    "layer": layer, "position": position,
                    "is_primary": is_primary, "skipped": True,
                    "reason": "degenerate_labels",
                }
                continue

            # Evaluate on T-eval
            probe_scores = predict_probe(probe, X_eval)
            au_probe = auroc2(probe_scores, y_eval)

            # Paired CI: probe vs verbal
            verbal_by_qid = dict(zip(teval_verbal_qids, teval_verbal_conf))
            correct_by_qid = dict(zip(teval_verbal_qids, teval_verbal_correct))
            probe_by_qid = dict(zip(eval_qids, probe_scores))

            paired_qids = [q for q in eval_qids if q in verbal_by_qid]
            if len(paired_qids) < 10:
                print("[warn] too few paired items for CI; reporting point only")
                paired_ci = None
            else:
                v = np.array([verbal_by_qid[q] for q in paired_qids])
                p = np.array([probe_by_qid[q] for q in paired_qids])
                y = np.array([correct_by_qid[q] for q in paired_qids])
                paired_ci = paired_bootstrap_auroc2_delta(
                    confidence_a=p, confidence_b=v, correct=y,
                    n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
                )

            print(f"[eval] AUROC₂ probe (T-eval, n={len(y_eval)}): {au_probe:.3f}")
            if paired_ci:
                print(f"       delta (probe - verbal) CI95: "
                      f"[{paired_ci['lo']:.3f}, {paired_ci['hi']:.3f}]  "
                      f"point={paired_ci['point_delta']:.3f}")

            all_probe_results[config_key] = {
                "layer": layer,
                "position": position,
                "is_primary": is_primary,
                "skipped": False,
                "n_train": int(len(X_train)),
                "n_eval": int(len(X_eval)),
                "hidden_dim": int(X_train.shape[1]),
                "best_C": probe["best_C"],
                "auroc2_probe_teval": au_probe,
                "auroc2_probe_minus_chance": au_probe - 0.5 if not np.isnan(au_probe) else None,
                "paired_ci_probe_minus_verbal": paired_ci,
            }
            all_teval_scores[config_key] = {
                q: float(s) for q, s in probe_by_qid.items()
            }
            all_probe_fits[config_key] = probe
            print()

    # Aggregate metadata
    all_probe_results["_meta"] = {
        "model_name": model_name,
        "primary_config": f"{PRIMARY_CONFIG[0]}_{PRIMARY_CONFIG[1]}",
        "elapsed_s": time.time() - start,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "lr_cv_folds": LR_CV_FOLDS,
        "lr_cs": LR_CS,
    }

    # Save
    metrics_path = PROBE_DIR / f"probe_metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_probe_results, f, indent=2, default=float)
    print(f"[save] Probe metrics: {metrics_path}")

    scores_path = PROBE_DIR / f"probe_scores_teval_{model_name}.json"
    with open(scores_path, "w") as f:
        json.dump(all_teval_scores, f, indent=2)
    print(f"[save] Probe scores: {scores_path}")

    # Save fitted probes as .npz (numpy arrays for scaler + coef)
    fits_flat = {}
    for config_key, probe in all_probe_fits.items():
        for k, v in probe.items():
            if isinstance(v, np.ndarray):
                fits_flat[f"{config_key}__{k}"] = v
            else:
                # Store scalars as 0-d arrays
                fits_flat[f"{config_key}__{k}"] = np.array(v)
    fits_path = PROBE_DIR / f"probe_fits_{model_name}.npz"
    np.savez_compressed(fits_path, **fits_flat)
    print(f"[save] Probe fits: {fits_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"STEP 1b SUMMARY: {model_name}")
    print(f"{'='*60}")
    print("Probe AUROC₂ on T-eval (per configuration):")
    for k, v in all_probe_results.items():
        if k.startswith("_"):
            continue
        if v.get("skipped"):
            print(f"     {k:30s}  SKIPPED ({v.get('reason', '?')})")
            continue
        marker = "  *" if v["is_primary"] else "   "
        print(f"{marker} {k:30s}  AUROC₂={v['auroc2_probe_teval']:.3f}  "
              f"(n_train={v['n_train']}, n_eval={v['n_eval']})")

    # E6 robustness check
    aus = [v["auroc2_probe_teval"]
           for k, v in all_probe_results.items()
           if not k.startswith("_") and not v.get("skipped")
           and not np.isnan(v["auroc2_probe_teval"])]
    if len(aus) >= 2:
        spread = max(aus) - min(aus)
        print(f"\n[E6] Probe AUROC₂ range: {min(aus):.3f} — {max(aus):.3f}  "
              f"(spread={spread:.3f})")
        if spread > 0.05:
            print("[E6] spread > 0.05 → regime classification may be "
                  "sensitive to probe specification.")

    print(f"\nElapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
