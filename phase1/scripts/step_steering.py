"""Step S: Inference-Time Steering Control Experiment
=====================================================

Tests whether additive activation steering along the probe's correctness
direction improves verbal confidence discrimination.

Motivated by the Miao & Ungar (2026) orthogonality finding: correctness
and confidence-verbalization are linearly encoded in ORTHOGONAL subspaces.
Prediction: additive steering along the correctness direction will NOT
improve verbal AUROC₂, because the verbalization pathway lives in a
different subspace.

Either outcome is informative:
  - Null (predicted): confirms orthogonality, strengthens the probe-target
    CSFT story (training-level intervention needed to bridge the gap)
  - Positive: contradicts Miao & Ungar, novel finding

Method:
  1. Load probe coefficients (fitted in Step 1b) as direction vector d
  2. Normalize d to unit length
  3. Monkey-patch the target layer to inject h → h + α·d at all positions
  4. Sweep α ∈ {-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10}
  5. Evaluate on T-eval: AUROC₂, accuracy, confidence distribution

Usage:
    python3 step_steering.py \\
        --model-name gemma-3-12b-it \\
        --model-path /path/to/gemma-3-12b-it

    # Custom alpha sweep:
    python3 step_steering.py \\
        --model-name gemma-3-12b-it \\
        --model-path /path/to/gemma-3-12b-it \\
        --alphas 0.5 1.0 2.0 5.0
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load
from gen_helpers import generate_greedy
from model_config import get_model_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_TEVAL = 1000
MAX_TOKENS = 256

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

DEFAULT_ALPHAS = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

RESULTS_DIR = Path(os.path.expanduser("~/jpwork/results/steering"))
STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1"))
PROBE_DIR = Path(os.path.expanduser("~/jpwork/results/probe"))


# ---------------------------------------------------------------------------
# Response parsing (identical to step4)
# ---------------------------------------------------------------------------

_CONFIDENCE_PATTERNS = [
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
    re.compile(r"\b(\d{1,3})\b\s*$"),
]

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _ARTICLE_RE.sub("", s)
    return s


def is_correct_triviaqa(pred: str, aliases: list) -> bool:
    p = _normalise(pred)
    if not p:
        return False
    for a in aliases:
        an = _normalise(a)
        if not an:
            continue
        if an in p or p in an:
            return True
    return False


def parse_response(raw: str) -> dict:
    text = raw.strip()
    answer = ""
    confidence = float("nan")

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        first = lines[0]
        m = re.match(r"^(?:answer\s*:?\s*)(.*)", first, flags=re.IGNORECASE)
        answer = m.group(1).strip() if m else first
        answer = re.sub(
            r"\s*[,;]?\s*confidence\s*:?.*$", "", answer, flags=re.IGNORECASE
        ).strip()

    for pat in _CONFIDENCE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    confidence = float(v)
                    break
            except ValueError:
                continue

    return {"answer": answer, "confidence": confidence}


# ---------------------------------------------------------------------------
# AUROC₂
# ---------------------------------------------------------------------------

def auroc2(confidence: np.ndarray, correct: np.ndarray) -> float:
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
# Data loading
# ---------------------------------------------------------------------------

def load_teval_items():
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    items = []
    for i in indices[0:N_TEVAL]:
        ex = ds[i]
        aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        items.append({
            "question_id": ex["question_id"],
            "question": ex["question"],
            "aliases": [a for a in aliases if a],
        })
    return items


# ---------------------------------------------------------------------------
# Steering vector extraction
# ---------------------------------------------------------------------------

def load_steering_direction(probe_fits_path: Path,
                            config_key: str = "middle_pre_answer_token"
                            ) -> np.ndarray:
    """Load probe coefficients and return unit-normalised direction vector.

    Uses middle-layer probe by default (peak AUROC₂ across all families).
    """
    fits = np.load(probe_fits_path)
    coef_key = f"{config_key}__coef"

    if coef_key not in fits:
        available = [f.replace("__coef", "") for f in fits.files if "__coef" in f]
        raise KeyError(
            f"Config '{config_key}' not found. Available: {available}"
        )

    coef = fits[coef_key].flatten()  # (hidden_dim,)
    # Normalise to unit length
    norm = np.linalg.norm(coef)
    if norm < 1e-10:
        raise ValueError("Probe coefficient vector has near-zero norm")

    d = coef / norm
    print(f"[steering] Loaded direction from '{config_key}': "
          f"dim={len(d)}, raw norm={norm:.4f}")
    return d


# ---------------------------------------------------------------------------
# Layer monkey-patching
# ---------------------------------------------------------------------------

class SteeredLayer:
    """Wraps an MLX transformer layer to add a steering vector to outputs.

    Applies h → h + α·d at all sequence positions during forward pass.
    """

    def __init__(self, original_layer, direction: mx.array, alpha: float):
        self._original = original_layer
        self._direction = direction  # (hidden_dim,) — broadcasts over batch, seq
        self._alpha = alpha
        # Copy attributes so the layer still looks like the original
        for attr in dir(original_layer):
            if not attr.startswith('_') and attr != '__call__':
                try:
                    setattr(self, attr, getattr(original_layer, attr))
                except (AttributeError, TypeError):
                    pass

    def __call__(self, *args, **kwargs):
        h = self._original(*args, **kwargs)
        # h shape: (batch, seq_len, hidden_dim)
        h = h + self._alpha * self._direction
        return h

    def __getattr__(self, name):
        # Fallback to original layer for any attribute not on wrapper
        return getattr(self._original, name)


def install_steering(model_cfg: dict, layer_idx: int,
                     direction: np.ndarray, alpha: float):
    """Monkey-patch a layer to add steering. Returns restore function."""
    layers = model_cfg["layers"]
    original_layer = layers[layer_idx]
    d_mx = mx.array(direction.astype(np.float32))

    steered = SteeredLayer(original_layer, d_mx, alpha)
    layers[layer_idx] = steered

    def restore():
        layers[layer_idx] = original_layer

    return restore


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_teval(model, tokenizer, items, label=""):
    results = []
    t_start = time.time()

    for i, item in enumerate(items):
        user_msg = TRIVIAQA_PROMPT.format(question=item["question"])
        messages = [{"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        raw = generate_greedy(model, tokenizer, prompt)
        parsed = parse_response(raw)
        correct = is_correct_triviaqa(parsed["answer"], item["aliases"])

        results.append({
            "question_id": item["question_id"],
            "parsed_answer": parsed["answer"],
            "parsed_confidence": parsed["confidence"],
            "correct": correct,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_min = (len(items) - i - 1) / rate / 60
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{label}] {i+1}/{len(items)}  acc={acc:.3f}  "
                  f"ETA: {eta_min:.1f}min")

    return results


def compute_quick_metrics(results):
    correct = np.array([r["correct"] for r in results], dtype=int)
    confidence = np.array([r["parsed_confidence"] for r in results], dtype=float)
    mask = ~np.isnan(confidence)

    return {
        "accuracy": round(float(correct.mean()), 4),
        "auroc2": round(auroc2(confidence, correct), 4),
        "confidence_mean": round(float(np.nanmean(confidence)), 2),
        "confidence_std": round(float(np.nanstd(confidence)), 2),
        "parse_rate": round(float(mask.mean()), 4),
        "n": len(results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference-time steering control experiment"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--probe-config", default="middle_pre_answer_token",
                        help="Probe config for steering direction "
                             "(default: middle_pre_answer_token)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Alpha values to sweep (default: "
                             f"{DEFAULT_ALPHAS})")
    parser.add_argument("--quick-test", type=int, default=0,
                        help="Test on N items first (0 = skip)")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = os.path.expanduser(args.model_path)
    alphas = args.alphas if args.alphas else DEFAULT_ALPHAS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load steering direction from probe fits
    # -----------------------------------------------------------------------
    probe_fits_path = PROBE_DIR / f"probe_fits_{model_name}.npz"
    if not probe_fits_path.exists():
        print(f"[fatal] Probe fits not found: {probe_fits_path}")
        sys.exit(1)

    direction = load_steering_direction(probe_fits_path,
                                        config_key=args.probe_config)

    # Parse probe config to find layer index
    layer_label = args.probe_config.split("_")[0]  # "middle", "last", "first"

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"\n[model] Loading {model_name}...")
    model, tokenizer = load(model_path)
    model_cfg = get_model_config(model, tokenizer, model_name)

    target_layer_idx = model_cfg["layer_indices"][layer_label]
    print(f"[steering] Target layer: {layer_label} (index {target_layer_idx})")
    print(f"[steering] Direction dim: {len(direction)}, "
          f"hidden dim: {model_cfg['hidden_dim']}")

    assert len(direction) == model_cfg["hidden_dim"], \
        f"Direction dim {len(direction)} != hidden dim {model_cfg['hidden_dim']}"

    # -----------------------------------------------------------------------
    # Load T-eval items
    # -----------------------------------------------------------------------
    items = load_teval_items()

    # Quick test first (if requested)
    if args.quick_test > 0:
        test_items = items[:args.quick_test]
        print(f"\n{'='*60}")
        print(f"QUICK TEST: {args.quick_test} items, α=1.0")
        print(f"{'='*60}")

        # Baseline (α=0)
        print("\n  [baseline] No steering...")
        baseline_results = evaluate_on_teval(
            model, tokenizer, test_items, label="baseline"
        )
        baseline_m = compute_quick_metrics(baseline_results)
        print(f"  baseline: AUROC₂={baseline_m['auroc2']:.3f}  "
              f"acc={baseline_m['accuracy']:.3f}  "
              f"conf={baseline_m['confidence_mean']:.1f}±{baseline_m['confidence_std']:.1f}")

        # Steered (α=1.0)
        print("\n  [steered] α=1.0...")
        restore = install_steering(model_cfg, target_layer_idx, direction, 1.0)
        steered_results = evaluate_on_teval(
            model, tokenizer, test_items, label="α=1.0"
        )
        restore()
        steered_m = compute_quick_metrics(steered_results)
        print(f"  steered:  AUROC₂={steered_m['auroc2']:.3f}  "
              f"acc={steered_m['accuracy']:.3f}  "
              f"conf={steered_m['confidence_mean']:.1f}±{steered_m['confidence_std']:.1f}")

        print("\n  Quick test passed. Proceeding to full sweep...")
        print(f"  (Generation working, KV cache intact)")

    # -----------------------------------------------------------------------
    # Full alpha sweep
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"FULL STEERING SWEEP: {model_name}")
    print(f"Direction: {args.probe_config}")
    print(f"Target layer: {layer_label} (idx {target_layer_idx})")
    print(f"Alphas: {alphas}")
    print(f"T-eval items: {len(items)}")
    print(f"{'='*60}")

    all_results = {}
    sweep_start = time.time()

    for alpha in alphas:
        print(f"\n--- α = {alpha} ---")

        if alpha == 0.0:
            # No steering (baseline on full T-eval)
            results = evaluate_on_teval(
                model, tokenizer, items, label=f"α={alpha}"
            )
        else:
            restore = install_steering(
                model_cfg, target_layer_idx, direction, alpha
            )
            results = evaluate_on_teval(
                model, tokenizer, items, label=f"α={alpha}"
            )
            restore()

        metrics = compute_quick_metrics(results)
        all_results[str(alpha)] = {
            "alpha": alpha,
            "metrics": metrics,
            "results": results,
        }

        print(f"  α={alpha:+.1f}: AUROC₂={metrics['auroc2']:.3f}  "
              f"acc={metrics['accuracy']:.3f}  "
              f"conf={metrics['confidence_mean']:.1f}±{metrics['confidence_std']:.1f}  "
              f"parse={metrics['parse_rate']:.3f}")

    total_time = time.time() - sweep_start

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEERING RESULTS SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  {'Alpha':>8s}  {'AUROC₂':>8s}  {'Accuracy':>8s}  "
          f"{'Conf μ':>8s}  {'Conf σ':>8s}  {'Parse':>6s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

    # Get baseline AUROC₂ for comparison
    baseline_auroc2 = None
    baseline_path = STEP1_DIR / f"teval_responses_{model_name}.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        bl_conf = np.array([r["parsed_confidence"] for r in bl], dtype=float)
        bl_correct = np.array([r["correct"] for r in bl], dtype=int)
        baseline_auroc2 = auroc2(bl_conf, bl_correct)
        print(f"  {'base':>8s}  {baseline_auroc2:>8.3f}  "
              f"{float(bl_correct.mean()):>8.3f}  "
              f"{float(np.nanmean(bl_conf)):>8.1f}  "
              f"{float(np.nanstd(bl_conf)):>8.1f}  {'(ref)':>6s}")

    for alpha in alphas:
        m = all_results[str(alpha)]["metrics"]
        marker = " ←" if alpha == 0.0 else ""
        print(f"  {alpha:>+8.1f}  {m['auroc2']:>8.3f}  "
              f"{m['accuracy']:>8.3f}  "
              f"{m['confidence_mean']:>8.1f}  "
              f"{m['confidence_std']:>8.1f}  "
              f"{m['parse_rate']:>6.3f}{marker}")

    print(f"\n  Total sweep time: {total_time/60:.1f} min")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    # Save summary (without per-item results, for quick inspection)
    summary = {
        "model_name": model_name,
        "probe_config": args.probe_config,
        "target_layer": layer_label,
        "target_layer_idx": target_layer_idx,
        "direction_norm_before_normalisation": float(np.linalg.norm(
            np.load(probe_fits_path)[f"{args.probe_config}__coef"].flatten()
        )),
        "baseline_auroc2": baseline_auroc2,
        "sweep": {
            str(a): all_results[str(a)]["metrics"] for a in alphas
        },
        "total_time_seconds": round(total_time, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    summary_path = RESULTS_DIR / f"steering_summary_{model_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[save] Summary: {summary_path}")

    # Save full per-item results (for detailed analysis)
    full_path = RESULTS_DIR / f"steering_full_{model_name}.json"
    with open(full_path, "w") as f:
        json.dump({
            str(a): all_results[str(a)]["results"] for a in alphas
        }, f, indent=2, default=float)
    print(f"[save] Full results: {full_path}")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    if "0.0" in all_results and baseline_auroc2 is not None:
        best_alpha = max(alphas, key=lambda a: all_results[str(a)]["metrics"]["auroc2"])
        best_auroc2 = all_results[str(best_alpha)]["metrics"]["auroc2"]
        improvement = best_auroc2 - baseline_auroc2

        print(f"\n{'='*60}")
        print(f"INTERPRETATION")
        print(f"{'='*60}")
        print(f"  Best α: {best_alpha} (AUROC₂ = {best_auroc2:.3f})")
        print(f"  vs baseline: delta = {improvement:+.3f}")

        if improvement > 0.03:
            print(f"  UNEXPECTED POSITIVE: Steering improved verbal AUROC₂ by >{improvement:.3f}")
            print(f"  This contradicts the Miao & Ungar orthogonality prediction.")
            print(f"  Investigate: is the improvement genuine, or an artifact of")
            print(f"  accuracy changes shifting the correct/incorrect composition?")
        elif improvement > 0.01:
            print(f"  MARGINAL: Small improvement, possibly noise. Check bootstrap CIs.")
        else:
            print(f"  NULL (as predicted): Steering along correctness direction")
            print(f"  does NOT improve verbal confidence discrimination.")
            print(f"  Consistent with orthogonality: correctness ⊥ verbalization.")
            print(f"  Training-level intervention (probe-target CSFT) is needed")
            print(f"  to bridge the gap.")

    print(f"\n=== Steering experiment complete: {model_name} ===")


if __name__ == "__main__":
    main()
