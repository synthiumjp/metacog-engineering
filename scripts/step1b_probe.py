"""
Step 1b: Linear Probe Control
==============================

Phase 0 v4, pre-reg v2. Fits L2-regularised logistic regression probes on
cached T-cal hidden states predicting correctness; applies each probe to
T-eval; computes AUROC₂ for each probe configuration.

Primary probe: last layer, last-answer-token position.
E6 sensitivity: {first, middle, last} × {pre_answer_token, last_answer_token}
  -> 6 configurations total.

Also computes the three primary deltas used in the decision tree:
  - AUROC₂_probe - AUROC₂_baseline_verbal  (does probe beat verbal?)
  - AUROC₂_probe - chance (0.5)             (does probe find signal at all?)
  (The third, AUROC₂_ft - AUROC₂_probe, is computed in Step 4 after SFT.)

Inputs:
    D:\\metacog\\data\\hidden_states\\baseline_tcal.pt
    D:\\metacog\\data\\hidden_states\\baseline_teval.pt
    D:\\metacog\\results\\baseline\\tcal_greedy_responses.json   (correctness labels)
    D:\\metacog\\results\\baseline\\teval_responses.json         (correctness + verbal conf)

Outputs:
    D:\\metacog\\results\\probe\\probe_fits.pt                   (dict of fitted probes)
    D:\\metacog\\results\\probe\\probe_metrics.json              (AUROC₂ per config + deltas)
    D:\\metacog\\results\\probe\\probe_scores_teval.json         (per-item probe scores on T-eval)

Runtime: ~15-30 min, mostly I/O and sklearn CV.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils_phase0 import auroc2, paired_bootstrap_auroc2_delta


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\metacog")
HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states"
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
PROBE_DIR = PROJECT_ROOT / "results" / "probe"

PROBE_LAYERS = ["first", "middle", "last"]
PROBE_POSITIONS = ["pre_answer_token", "last_answer_token"]
PRIMARY_CONFIG = ("last", "last_answer_token")

LR_CV_FOLDS = 5
LR_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_hidden_states_and_labels(
    hs_path: Path,
    responses_path: Path,
    layer: str,
    position: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load hidden states + correctness labels for a given layer/position.

    Returns (X, y, qids) where rows of X are items for which the specified
    (layer, position) hidden state was successfully captured.
    """
    hs = torch.load(hs_path, weights_only=False)
    with open(responses_path) as f:
        records = json.load(f)
    correct_by_qid = {r["question_id"]: int(r["correct"]) for r in records}

    X_list, y_list, qids = [], [], []
    for qid, layer_dict in hs.items():
        if qid not in correct_by_qid:
            continue
        if layer not in layer_dict:
            continue
        if position not in layer_dict[layer]:
            continue
        X_list.append(layer_dict[layer][position])
        y_list.append(correct_by_qid[qid])
        qids.append(qid)

    if not X_list:
        raise RuntimeError(
            f"No items with hidden states at layer={layer}, position={position}. "
            f"Check hs_path={hs_path}"
        )
    return np.stack(X_list, axis=0), np.array(y_list, dtype=int), qids


def load_verbal_confidence_on_teval(responses_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load T-eval verbal confidence + correctness aligned by qid."""
    with open(responses_path) as f:
        records = json.load(f)
    confidences, correct, qids = [], [], []
    for r in records:
        if np.isnan(r["parsed_confidence"]):
            continue
        confidences.append(r["parsed_confidence"])
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

    # If all labels are one class, fitting is degenerate
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
    # Sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs[:, 0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    tcal_hs_path = HIDDEN_DIR / "baseline_tcal.pt"
    teval_hs_path = HIDDEN_DIR / "baseline_teval.pt"
    tcal_resp_path = BASELINE_DIR / "tcal_greedy_responses.json"
    teval_resp_path = BASELINE_DIR / "teval_responses.json"

    for p in [tcal_hs_path, teval_hs_path, tcal_resp_path, teval_resp_path]:
        if not p.exists():
            print(f"[fatal] missing input: {p}")
            sys.exit(2)

    # Load T-eval verbal confidence (for the probe-vs-verbal comparison)
    teval_verbal_conf, teval_verbal_correct, teval_verbal_qids = \
        load_verbal_confidence_on_teval(teval_resp_path)
    print(f"[data] T-eval verbal: {len(teval_verbal_qids)} items with parseable confidence")

    all_probe_results = {}
    all_teval_scores = {}  # {config_key: {qid: prob}}

    start = time.time()

    for layer in PROBE_LAYERS:
        for position in PROBE_POSITIONS:
            config_key = f"{layer}_{position}"
            is_primary = (layer, position) == PRIMARY_CONFIG
            tag = "PRIMARY" if is_primary else "sensitivity"

            print(f"\n=== Probe config: {config_key} ({tag}) ===")

            # Load training data (T-cal)
            try:
                X_train, y_train, _ = load_hidden_states_and_labels(
                    tcal_hs_path, tcal_resp_path, layer, position
                )
            except RuntimeError as e:
                print(f"[skip] {e}")
                continue

            # Load eval data (T-eval)
            try:
                X_eval, y_eval, eval_qids = load_hidden_states_and_labels(
                    teval_hs_path, teval_resp_path, layer, position
                )
            except RuntimeError as e:
                print(f"[skip] {e}")
                continue

            print(f"[fit]  n_train={len(X_train)}, dim={X_train.shape[1]}, "
                  f"pos_rate_train={y_train.mean():.3f}")
            probe = fit_probe(X_train, y_train)
            if probe.get("degenerate", False):
                print("[skip] degenerate labels in T-cal; cannot fit probe")
                continue

            # Evaluate on T-eval
            probe_scores = predict_probe(probe, X_eval)
            au_probe = auroc2(probe_scores, y_eval)

            # Align probe scores with verbal-confidence T-eval items for paired CI
            # (only items where both exist)
            verbal_by_qid = dict(zip(teval_verbal_qids, teval_verbal_conf))
            correct_by_qid = dict(zip(teval_verbal_qids, teval_verbal_correct))
            probe_by_qid = dict(zip(eval_qids, probe_scores))

            paired_qids = [q for q in eval_qids if q in verbal_by_qid]
            if len(paired_qids) < 2:
                print("[warn] too few paired items for CI; reporting point only")
                paired_ci = None
            else:
                v = np.array([verbal_by_qid[q] for q in paired_qids])
                p = np.array([probe_by_qid[q] for q in paired_qids])
                y = np.array([correct_by_qid[q] for q in paired_qids])
                paired_ci = paired_bootstrap_auroc2_delta(
                    confidence_a=p,       # probe
                    confidence_b=v,       # verbal
                    correct=y,
                    n_resamples=BOOTSTRAP_N,
                    seed=BOOTSTRAP_SEED,
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
                "n_train": int(len(X_train)),
                "n_eval": int(len(X_eval)),
                "best_C": probe["best_C"],
                "auroc2_probe_teval": au_probe,
                "paired_ci_probe_minus_verbal": paired_ci,
            }
            all_teval_scores[config_key] = {q: float(s) for q, s in probe_by_qid.items()}

            # Save the fitted probe (small — coef + intercept + scaler stats)
            probe_save_path = PROBE_DIR / f"probe_{config_key}.pt"
            torch.save({
                "probe": probe,
                "layer": layer,
                "position": position,
                "n_train": int(len(X_train)),
            }, probe_save_path)

    # Aggregate
    all_probe_results["_meta"] = {
        "primary_config": f"{PRIMARY_CONFIG[0]}_{PRIMARY_CONFIG[1]}",
        "elapsed_s": time.time() - start,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "lr_cv_folds": LR_CV_FOLDS,
        "lr_cs": LR_CS,
    }

    with open(PROBE_DIR / "probe_metrics.json", "w") as f:
        json.dump(all_probe_results, f, indent=2, default=float)
    with open(PROBE_DIR / "probe_scores_teval.json", "w") as f:
        json.dump(all_teval_scores, f, indent=2)

    # Summary
    print("\n=== Step 1b complete ===")
    print("Summary of probe AUROC₂ on T-eval (per configuration):")
    for k, v in all_probe_results.items():
        if k.startswith("_"):
            continue
        marker = "  *" if v["is_primary"] else "   "
        print(f"{marker} {k:30s}  AUROC₂={v['auroc2_probe_teval']:.3f}  "
              f"(n_train={v['n_train']}, n_eval={v['n_eval']})")

    # E6 robustness check
    aus = [v["auroc2_probe_teval"] for k, v in all_probe_results.items()
           if not k.startswith("_") and not np.isnan(v["auroc2_probe_teval"])]
    if len(aus) >= 2:
        spread = max(aus) - min(aus)
        print(f"\n[E6] Probe AUROC₂ range across configs: "
              f"{min(aus):.3f} -- {max(aus):.3f}  (spread={spread:.3f})")
        if spread > 0.05:
            print("[E6] Note: spread > 0.05 -> Regime classification may be "
                  "sensitive to probe specification. Primary config "
                  "(last layer, last-answer-token) remains authoritative for "
                  "the decision tree, but E6 sensitivity should be reported.")


if __name__ == "__main__":
    main()
