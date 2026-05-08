"""
Step 4: Post-SFT Evaluation (Phase 1 — MLX on M3 Ultra)
=========================================================

Evaluates fine-tuned models (real + shuffled adapters) on held-out T-eval
(1,000 TriviaQA items). Computes AUROC₂, accuracy, VRS, and all pre-reg
pairwise comparisons with paired bootstrap CIs.

Decision tree from pre-reg:
    H1: AUROC₂_ft_real > AUROC₂_baseline  (primary hypothesis)
    E7: AUROC₂_ft_real > AUROC₂_ft_shuffled  (shuffled control)
    Regime: compare ft_real vs probe (from Step 1b)

Also evaluates on M-eval (MMLU) if available.

Usage:
    python3 step4_eval_phase1.py \
        --model-name gemma-3-12b-it \
        --model-path /Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it

Inputs:
    results/finetune/{model}/real/adapters/adapters.safetensors
    results/finetune/{model}/shuffled/adapters/adapters.safetensors
    results/step1/teval_responses_{model}.json  (baseline)
    results/probe/probe_scores_teval_{model}.json  (probe)

Outputs:
    results/step4/step4_real_{model}.json       (raw responses)
    results/step4/step4_shuffled_{model}.json   (raw responses)
    results/step4/step4_metrics_{model}.json    (all metrics + CIs)
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load
from gen_helpers import generate_greedy


# ---------------------------------------------------------------------------
# Constants (locked per pre-reg)
# ---------------------------------------------------------------------------

SEED = 42
N_TEVAL = 1000
N_TCAL = 2000
MAX_TOKENS = 256

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

MODEL_LAYERS = {
    "gemma-3-12b-it": 48,
    "gemma-3-27b-it": 62,
}

LORA_CONFIG = {"rank": 16, "dropout": 0.05, "scale": 2.0}

RESULTS_DIR = Path(os.path.expanduser("~/jpwork/results/step4"))
STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1"))
PROBE_DIR = Path(os.path.expanduser("~/jpwork/results/probe"))
FINETUNE_DIR = Path(os.path.expanduser("~/jpwork/results/finetune"))


# ---------------------------------------------------------------------------
# Response parsing (identical to Step 1)
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
    """Parse answer and confidence from generated text (TriviaQA format)."""
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
# VRS screen (identical to Step 1)
# ---------------------------------------------------------------------------

def vrs_screen(confidence: np.ndarray, correct: np.ndarray) -> dict:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    n = len(c)
    if n == 0:
        return {"tier": "undefined", "n": 0}

    L = float(np.mean(c >= 95.0))
    Fp = float(np.mean(c <= 5.0))
    bins = np.clip((c // 10).astype(int), 0, 10)
    distinct = len(np.unique(bins))
    RBS = 1 - distinct / 11
    unique, counts = np.unique(bins, return_counts=True)
    TRIN = float(counts.max() / counts.sum())

    if c.std() == 0 or y.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(c, y)[0, 1])

    if L >= 0.70 or TRIN >= 0.80 or abs(r) < 0.05:
        tier = "Invalid"
    elif L >= 0.40 or TRIN >= 0.60:
        tier = "Indeterminate"
    else:
        tier = "Valid"

    return {
        "tier": tier,
        "n": int(n),
        "L_ceiling": round(L, 4),
        "Fp_floor": round(Fp, 4),
        "RBS": round(RBS, 4),
        "TRIN": round(TRIN, 4),
        "r_conf_correct": round(r, 4),
    }


# ---------------------------------------------------------------------------
# AUROC₂ + bootstrap
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


def paired_bootstrap_delta(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    correct: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI for AUROC₂(A) - AUROC₂(B)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(conf_a, dtype=float)
    b = np.asarray(conf_b, dtype=float)
    y = np.asarray(correct, dtype=int)

    # Align on non-NaN for both
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b, y = a[mask], b[mask], y[mask]
    n = len(y)

    point_a = auroc2(a, y)
    point_b = auroc2(b, y)
    point_delta = point_a - point_b

    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n:
            deltas[i] = np.nan
            continue
        deltas[i] = auroc2(a[idx], yi) - auroc2(b[idx], yi)

    deltas = deltas[~np.isnan(deltas)]
    return {
        "auroc2_a": round(point_a, 4),
        "auroc2_b": round(point_b, 4),
        "delta": round(point_delta, 4),
        "ci95_lo": round(float(np.quantile(deltas, 0.025)), 4),
        "ci95_hi": round(float(np.quantile(deltas, 0.975)), 4),
        "significant": bool(
            np.quantile(deltas, 0.025) > 0 or np.quantile(deltas, 0.975) < 0
        ),
        "n_paired": int(n),
        "n_boot": len(deltas),
    }


# ---------------------------------------------------------------------------
# Data loading (same partition as all other steps)
# ---------------------------------------------------------------------------

def load_teval_items():
    """Load T-eval partition (indices[0:1000])."""
    from datasets import load_dataset

    print("[data] Loading TriviaQA rc.nocontext validation...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    indices = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    teval_idx = indices[0:N_TEVAL]
    items = []
    for i in teval_idx:
        ex = ds[i]
        aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        items.append({
            "question_id": ex["question_id"],
            "question": ex["question"],
            "aliases": [a for a in aliases if a],
        })
    print(f"[data] T-eval: {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# Prompt + generation
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, question: str) -> str:
    user_msg = TRIVIAQA_PROMPT.format(question=question)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def evaluate_model_on_teval(
    model, tokenizer, items: list[dict], label: str = "eval",
) -> list[dict]:
    """Generate greedy on all T-eval items, parse response."""
    results = []
    t_start = time.time()

    for i, item in enumerate(items):
        prompt = build_prompt(tokenizer, item["question"])
        raw = generate_greedy(model, tokenizer, prompt)
        parsed = parse_response(raw)
        correct = is_correct_triviaqa(parsed["answer"], item["aliases"])

        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "raw_output": raw,
            "parsed_answer": parsed["answer"],
            "parsed_confidence": parsed["confidence"],
            "correct": correct,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_min = (len(items) - i - 1) / rate / 60
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{label}] {i+1}/{len(items)}  acc={acc:.3f}  ETA: {eta_min:.1f}min")

    elapsed = time.time() - t_start
    print(f"  [{label}] Complete in {elapsed/60:.1f}min")
    return results


def compute_metrics(results: list[dict], label: str) -> dict:
    """Compute accuracy, AUROC₂, VRS from evaluation results."""
    correct = np.array([r["correct"] for r in results], dtype=int)
    confidence = np.array([r["parsed_confidence"] for r in results], dtype=float)
    mask = ~np.isnan(confidence)

    acc = float(correct.mean())
    au = auroc2(confidence, correct)
    vrs = vrs_screen(confidence, correct)
    conf_mean = float(np.nanmean(confidence))
    conf_std = float(np.nanstd(confidence))
    parseable_rate = float(mask.mean())

    print(f"\n  [{label}] Accuracy: {acc:.3f}")
    print(f"  [{label}] AUROC₂: {au:.3f}")
    print(f"  [{label}] VRS: {vrs['tier']} (L={vrs['L_ceiling']}, TRIN={vrs['TRIN']}, r={vrs['r_conf_correct']})")
    print(f"  [{label}] Confidence: mean={conf_mean:.1f}, std={conf_std:.1f}")
    print(f"  [{label}] Parseable: {parseable_rate:.3f}")

    return {
        "label": label,
        "n_items": len(results),
        "accuracy": round(acc, 4),
        "auroc2": round(au, 4),
        "vrs": vrs,
        "confidence_mean": round(conf_mean, 2),
        "confidence_std": round(conf_std, 2),
        "parseable_rate": round(parseable_rate, 4),
    }


# ---------------------------------------------------------------------------
# Model loading with adapters
# ---------------------------------------------------------------------------

def load_model_with_adapter(model_path: str, adapter_path: str, model_name: str):
    """Load base model + LoRA adapter."""
    from mlx_lm.lora import linear_to_lora_layers

    print(f"[model] Loading base model: {model_path}")
    model, tokenizer = load(model_path)

    n_layers = MODEL_LAYERS[model_name]
    linear_to_lora_layers(model, num_layers=n_layers, config=LORA_CONFIG)

    # Load adapter weights
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter not found: {adapter_file}")

    print(f"[model] Loading adapter: {adapter_file}")
    adapter_weights = mx.load(adapter_file)
    model.load_weights(list(adapter_weights.items()), strict=False)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 4: Post-SFT evaluation")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--skip-shuffled", action="store_true",
                        help="Skip shuffled evaluation (if already done)")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = os.path.expanduser(args.model_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Paths
    real_adapter = FINETUNE_DIR / model_name / "real" / "adapters"
    shuffled_adapter = FINETUNE_DIR / model_name / "shuffled" / "adapters"
    baseline_path = STEP1_DIR / f"teval_responses_{model_name}.json"
    probe_path = PROBE_DIR / f"probe_scores_teval_{model_name}.json"

    # Check inputs
    for p, label in [
        (real_adapter / "adapters.safetensors", "real adapter"),
        (shuffled_adapter / "adapters.safetensors", "shuffled adapter"),
        (baseline_path, "baseline responses"),
        (probe_path, "probe scores"),
    ]:
        if not p.exists():
            print(f"[warn] Missing: {p} ({label})")

    # -----------------------------------------------------------------------
    # Load T-eval items
    # -----------------------------------------------------------------------
    teval_items = load_teval_items()

    # -----------------------------------------------------------------------
    # Load baseline data (from Step 1)
    # -----------------------------------------------------------------------
    print("\n[baseline] Loading Step 1 baseline...")
    with open(baseline_path) as f:
        baseline_results = json.load(f)
    baseline_metrics = compute_metrics(baseline_results, "baseline")

    baseline_by_qid = {
        r["question_id"]: r for r in baseline_results
    }

    # -----------------------------------------------------------------------
    # Load probe scores (from Step 1b)
    # -----------------------------------------------------------------------
    probe_scores_by_qid = {}
    if probe_path.exists():
        with open(probe_path) as f:
            probe_data = json.load(f)
        # Use primary config
        primary_key = "last_last_answer_token"
        if primary_key in probe_data:
            probe_scores_by_qid = probe_data[primary_key]
            print(f"[probe] Loaded {len(probe_scores_by_qid)} probe scores (primary config)")
        else:
            print(f"[warn] Primary probe config '{primary_key}' not found")
    else:
        print("[warn] No probe scores found; skipping probe comparison")

    # -----------------------------------------------------------------------
    # Evaluate: real-target fine-tuned model
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating REAL-TARGET fine-tuned model")
    print(f"{'='*60}")

    model_real, tokenizer_real = load_model_with_adapter(
        model_path, str(real_adapter), model_name
    )
    real_results = evaluate_model_on_teval(
        model_real, tokenizer_real, teval_items, label="real-ft"
    )
    real_metrics = compute_metrics(real_results, "real-ft")

    # Save raw results
    real_out = RESULTS_DIR / f"step4_real_{model_name}.json"
    with open(real_out, "w") as f:
        json.dump(real_results, f, indent=2)
    print(f"[save] {real_out}")

    # Free model memory
    del model_real, tokenizer_real
    gc.collect()

    # -----------------------------------------------------------------------
    # Evaluate: shuffled-target fine-tuned model
    # -----------------------------------------------------------------------
    if not args.skip_shuffled:
        print(f"\n{'='*60}")
        print("Evaluating SHUFFLED-TARGET fine-tuned model")
        print(f"{'='*60}")

        model_shuf, tokenizer_shuf = load_model_with_adapter(
            model_path, str(shuffled_adapter), model_name
        )
        shuf_results = evaluate_model_on_teval(
            model_shuf, tokenizer_shuf, teval_items, label="shuffled-ft"
        )
        shuf_metrics = compute_metrics(shuf_results, "shuffled-ft")

        shuf_out = RESULTS_DIR / f"step4_shuffled_{model_name}.json"
        with open(shuf_out, "w") as f:
            json.dump(shuf_results, f, indent=2)
        print(f"[save] {shuf_out}")

        del model_shuf, tokenizer_shuf
        gc.collect()
    else:
        # Load existing
        shuf_out = RESULTS_DIR / f"step4_shuffled_{model_name}.json"
        if shuf_out.exists():
            with open(shuf_out) as f:
                shuf_results = json.load(f)
            shuf_metrics = compute_metrics(shuf_results, "shuffled-ft")
        else:
            shuf_results = None
            shuf_metrics = None

    # -----------------------------------------------------------------------
    # Pairwise comparisons (paired bootstrap)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PAIRWISE COMPARISONS (paired bootstrap, n=10000)")
    print(f"{'='*60}")

    # Align all conditions by QID
    real_by_qid = {r["question_id"]: r for r in real_results}
    shuf_by_qid = {r["question_id"]: r for r in shuf_results} if shuf_results else {}

    # Common QIDs across all conditions
    common_qids = sorted(
        set(baseline_by_qid.keys()) &
        set(real_by_qid.keys())
    )
    print(f"\n  Paired items: {len(common_qids)}")

    # Extract aligned arrays
    baseline_conf = np.array([baseline_by_qid[q]["parsed_confidence"] for q in common_qids])
    real_conf = np.array([real_by_qid[q]["parsed_confidence"] for q in common_qids])
    # Use real-ft correctness for all comparisons (same items, correctness shouldn't change much)
    correct = np.array([real_by_qid[q]["correct"] for q in common_qids], dtype=int)
    # Also get baseline correctness for verification
    baseline_correct = np.array([baseline_by_qid[q]["correct"] for q in common_qids], dtype=int)

    print(f"  Accuracy agreement (real vs baseline): {np.mean(correct == baseline_correct):.3f}")

    # H1: real-ft vs baseline
    print(f"\n  --- H1: AUROC₂(real-ft) vs AUROC₂(baseline) ---")
    h1 = paired_bootstrap_delta(real_conf, baseline_conf, correct,
                                 n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    print(f"  real-ft: {h1['auroc2_a']:.3f},  baseline: {h1['auroc2_b']:.3f}")
    print(f"  delta: {h1['delta']:.3f}  CI95: [{h1['ci95_lo']:.3f}, {h1['ci95_hi']:.3f}]")
    print(f"  H1 {'SUPPORTED' if h1['significant'] and h1['delta'] > 0 else 'NOT SUPPORTED'}")

    # E7: real-ft vs shuffled-ft
    e7 = None
    if shuf_results:
        shuf_conf = np.array([shuf_by_qid[q]["parsed_confidence"]
                              for q in common_qids if q in shuf_by_qid])
        shuf_qids = [q for q in common_qids if q in shuf_by_qid]
        if len(shuf_qids) == len(common_qids):
            print(f"\n  --- E7: AUROC₂(real-ft) vs AUROC₂(shuffled-ft) ---")
            shuf_conf_aligned = np.array([shuf_by_qid[q]["parsed_confidence"] for q in common_qids])
            e7 = paired_bootstrap_delta(real_conf, shuf_conf_aligned, correct,
                                         n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            print(f"  real-ft: {e7['auroc2_a']:.3f},  shuffled-ft: {e7['auroc2_b']:.3f}")
            print(f"  delta: {e7['delta']:.3f}  CI95: [{e7['ci95_lo']:.3f}, {e7['ci95_hi']:.3f}]")
            print(f"  E7 {'REAL > SHUFFLED' if e7['significant'] and e7['delta'] > 0 else 'NO DIFFERENCE'}")

    # Probe comparison: real-ft vs probe
    probe_ci = None
    if probe_scores_by_qid:
        probe_qids = [q for q in common_qids if q in probe_scores_by_qid]
        if len(probe_qids) > 100:
            print(f"\n  --- Regime: AUROC₂(real-ft) vs AUROC₂(probe) ---")
            probe_conf = np.array([probe_scores_by_qid[q] for q in probe_qids])
            real_conf_p = np.array([real_by_qid[q]["parsed_confidence"] for q in probe_qids])
            correct_p = np.array([real_by_qid[q]["correct"] for q in probe_qids], dtype=int)
            # Scale probe to 0-100 for comparison (probe outputs P(correct))
            probe_conf_scaled = probe_conf * 100

            probe_ci = paired_bootstrap_delta(
                real_conf_p, probe_conf_scaled, correct_p,
                n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
            )
            print(f"  real-ft verbal: {probe_ci['auroc2_a']:.3f},  probe: {probe_ci['auroc2_b']:.3f}")
            print(f"  delta: {probe_ci['delta']:.3f}  CI95: [{probe_ci['ci95_lo']:.3f}, {probe_ci['ci95_hi']:.3f}]")

            if probe_ci["significant"] and probe_ci["delta"] > 0:
                regime = "Regime 2: SFT verbal > probe (monitoring enhancement)"
            elif probe_ci["significant"] and probe_ci["delta"] < 0:
                regime = "Regime 3: probe > SFT verbal (engineering of verbalisation)"
            else:
                regime = "Regime 3 (marginal): SFT verbal ≈ probe"
            print(f"  {regime}")
        else:
            print(f"\n  [warn] Only {len(probe_qids)} paired items for probe comparison")

    # -----------------------------------------------------------------------
    # Decision tree summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"DECISION TREE SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  Baseline AUROC₂:   {baseline_metrics['auroc2']:.3f} (VRS: {baseline_metrics['vrs']['tier']})")
    print(f"  Real-ft AUROC₂:    {real_metrics['auroc2']:.3f} (VRS: {real_metrics['vrs']['tier']})")
    if shuf_metrics:
        print(f"  Shuffled-ft AUROC₂: {shuf_metrics['auroc2']:.3f} (VRS: {shuf_metrics['vrs']['tier']})")
    print()
    print(f"  H1 (real > baseline): {'SUPPORTED' if h1['significant'] and h1['delta'] > 0 else 'NOT SUPPORTED'}")
    print(f"    delta={h1['delta']:.3f}  CI95=[{h1['ci95_lo']:.3f}, {h1['ci95_hi']:.3f}]")
    if e7:
        print(f"  E7 (real > shuffled): {'YES' if e7['significant'] and e7['delta'] > 0 else 'NO'}")
        print(f"    delta={e7['delta']:.3f}  CI95=[{e7['ci95_lo']:.3f}, {e7['ci95_hi']:.3f}]")
    if probe_ci:
        print(f"  Regime: real-ft vs probe delta={probe_ci['delta']:.3f}  CI95=[{probe_ci['ci95_lo']:.3f}, {probe_ci['ci95_hi']:.3f}]")

    # -----------------------------------------------------------------------
    # Save comprehensive metrics
    # -----------------------------------------------------------------------
    metrics = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": baseline_metrics,
        "real_ft": real_metrics,
        "shuffled_ft": shuf_metrics,
        "comparisons": {
            "h1_real_vs_baseline": h1,
            "e7_real_vs_shuffled": e7,
            "regime_real_vs_probe": probe_ci,
        },
        "n_teval_items": len(teval_items),
        "n_paired": len(common_qids),
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    metrics_path = RESULTS_DIR / f"step4_metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\n[save] Metrics: {metrics_path}")

    print(f"\n=== Step 4 complete for {model_name} ===")


if __name__ == "__main__":
    main()
