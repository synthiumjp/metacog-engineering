"""
eval_confonly_triviaqa.py — Two-pass evaluation for balanced confidence-only
adapter on TriviaQA.

Pass 1: Generate answers WITHOUT the adapter (preserves accuracy)
Pass 2: Rate confidence WITH the adapter (probe-targeted calibration)

This tests whether the balanced confidence-only training strategy breaks
the Llama 70B ceiling.

Usage:
    python3 eval_confonly_triviaqa.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED \
        --model-name Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED \
        --adapter-path ~/jpwork/metacog-engineering/phase1/results_raw/finetune/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED/balanced_confonly/adapters
"""
import argparse, json, os, random, re, time
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate as mlx_generate
from sklearn.metrics import roc_auc_score

SEED = 42
MAX_TOKENS_ANSWER = 256
MAX_TOKENS_CONF = 30


def _greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Data loading — TriviaQA T-eval
# ---------------------------------------------------------------------------
def load_triviaqa_eval(seed=SEED, n_eval=1000):
    """Match the partition used in step1_baseline.py / eval_ablation.py."""
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    items = []
    for row in ds:
        items.append({
            "id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"]["normalized_value"],
            "aliases": row["answer"].get("normalized_aliases", []),
        })
    random.seed(seed)
    random.shuffle(items)
    # T-eval is the first n_eval items (matching the existing partition)
    return items[:n_eval]


# ---------------------------------------------------------------------------
# Correctness — flexible substring matching
# ---------------------------------------------------------------------------
def _normalise(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct_flex(predicted, gold, aliases=None):
    """Bidirectional substring matching with min length 2."""
    if not predicted or not gold:
        return False
    pred_norm = _normalise(predicted)
    candidates = [gold] + (aliases or [])
    for c in candidates:
        c_norm = _normalise(c)
        if len(c_norm) < 2:
            continue
        if c_norm in pred_norm or pred_norm in c_norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Confidence parsing
# ---------------------------------------------------------------------------
_CONF_PATTERNS = [
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
]

def parse_confidence(text):
    for pat in _CONF_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return float(v)
            except ValueError:
                pass
    return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TRIVIA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--output-dir", default=os.path.expanduser(
        "~/jpwork/metacog-engineering/phase1/results_raw/step4"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Balanced Confidence-Only Two-Pass Eval (TriviaQA)")
    print(f"  Model: {args.model_name}")
    print(f"  Adapter: {args.adapter_path}")
    print(f"{'='*60}\n")

    # Load data
    print(f"[data] Loading TriviaQA T-eval ({args.n_eval} items)...")
    eval_items = load_triviaqa_eval(n_eval=args.n_eval)
    print(f"  Loaded: {len(eval_items)}\n")

    # ── Pass 1: Generate answers WITHOUT adapter ──
    print(f"=== Pass 1: Generate answers (no adapter) ===")
    model, tokenizer = load(args.model_path)
    results = []
    t0 = time.time()

    for i, item in enumerate(eval_items):
        user_msg = TRIVIA_PROMPT.format(question=item["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        raw = mlx_generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_TOKENS_ANSWER, verbose=False,
                           sampler=_greedy_sampler)
        # Strip the confidence the base model might add
        answer_text = re.split(r"(?i)\bconfidence\b", raw)[0].strip()
        correct = is_correct_flex(answer_text, item["answer"], item.get("aliases"))

        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold": item["answer"],
            "raw_answer": raw,
            "answer_text": answer_text,
            "correct": correct,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in results])
            print(f"  [{i+1}/{len(eval_items)}]  acc={acc:.3f}  elapsed={elapsed:.0f}s")

    acc_pass1 = np.mean([r["correct"] for r in results])
    print(f"\n  Pass 1 accuracy: {acc_pass1:.3f}")
    print(f"  Pass 1 time: {time.time() - t0:.0f}s")

    del model
    mx.metal.clear_cache()

    # ── Pass 2: Rate confidence WITH adapter ──
    print(f"\n=== Pass 2: Rate confidence (with adapter) ===")
    model_adapted, tokenizer = load(args.model_path, adapter_path=args.adapter_path)
    t1 = time.time()

    for i, r in enumerate(results):
        conf_prompt_text = (
            "You answered the following trivia question.\n"
            f"Question: {r['question']}\n"
            f"Your answer: {r['answer_text']}\n"
            "How confident are you that your answer is correct? "
            "State your confidence as a percentage from 0 to 100."
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": conf_prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )
        conf_response = mlx_generate(model_adapted, tokenizer, prompt=prompt,
                                      max_tokens=MAX_TOKENS_CONF, verbose=False,
                                      sampler=_greedy_sampler)
        conf = parse_confidence(conf_response)
        r["conf_response"] = conf_response
        r["confidence"] = conf

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t1
            confs = np.array([r2["confidence"] for r2 in results[:i+1]])
            parse_rate = (~np.isnan(confs)).mean()
            print(f"  [{i+1}/{len(results)}]  parse={parse_rate:.3f}  "
                  f"conf_mean={np.nanmean(confs):.1f}  "
                  f"conf_std={np.nanstd(confs):.1f}  elapsed={elapsed:.0f}s")

    # ── Metrics ──
    corrects = np.array([r["correct"] for r in results], dtype=int)
    confs = np.array([r["confidence"] for r in results])
    mask = ~np.isnan(confs)

    parse_rate = mask.mean()
    conf_mean = float(np.nanmean(confs))
    conf_std = float(np.nanstd(confs))

    if mask.sum() > 10 and corrects[mask].sum() > 0 and corrects[mask].sum() < mask.sum():
        auroc2 = roc_auc_score(corrects[mask], confs[mask])
    else:
        auroc2 = float("nan")

    # VRS components
    L = float((confs[mask] >= 95).mean()) if mask.sum() > 0 else float("nan")
    if mask.sum() > 0:
        top_val = float(np.bincount(confs[mask].astype(int), minlength=101).argmax())
        TRIN = float((confs[mask] == top_val).mean())
    else:
        TRIN = float("nan")
    if mask.sum() > 10 and corrects[mask].sum() > 0:
        r_pearson = float(np.corrcoef(corrects[mask], confs[mask])[0, 1])
    else:
        r_pearson = float("nan")

    vrs = "Valid" if (L < 0.9 and TRIN < 0.5 and r_pearson > 0.15) else "Invalid"

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  RESULTS: Balanced Confidence-Only (TriviaQA)")
    print(f"  Model: {args.model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {acc_pass1:.3f}  (preserved — no adapter at gen)")
    print(f"  Parse rate:     {parse_rate:.3f}")
    print(f"  Verbal AUROC₂:  {auroc2:.3f}")
    print(f"  Conf mean:      {conf_mean:.1f}")
    print(f"  Conf std:       {conf_std:.1f}")
    print(f"  VRS:            {vrs} (L={L:.3f}, TRIN={TRIN:.3f}, r={r_pearson:.3f})")
    print(f"\n  Total time: {total_time:.0f}s")

    # Baseline comparison
    baseline_auroc2 = 0.724  # Llama 70B baseline
    baseline_conf_mean = 94.6
    baseline_conf_std = 7.0  # approx
    print(f"\n  Baseline AUROC₂: {baseline_auroc2:.3f}")
    print(f"  Delta:           {auroc2 - baseline_auroc2:+.3f}")
    print(f"  Baseline conf:   {baseline_conf_mean:.1f} ± ?")
    print(f"  Probe AUROC₂:    0.834")
    print(f"  Recovery:        {(auroc2 - 0.5) / (0.834 - 0.5) * 100:.1f}%")

    # Save
    summary = {
        "method": "balanced_confidence_only_two_pass",
        "model": args.model_name,
        "adapter_path": args.adapter_path,
        "accuracy": round(acc_pass1, 3),
        "parse_rate": round(parse_rate, 3),
        "verbal_auroc2": round(auroc2, 3) if not np.isnan(auroc2) else None,
        "conf_mean": round(conf_mean, 1),
        "conf_std": round(conf_std, 1),
        "vrs": vrs,
        "L": round(L, 3),
        "TRIN": round(TRIN, 3),
        "r_pearson": round(r_pearson, 3),
        "baseline_auroc2": baseline_auroc2,
        "delta": round(auroc2 - baseline_auroc2, 3) if not np.isnan(auroc2) else None,
        "probe_auroc2": 0.834,
        "recovery": round((auroc2 - 0.5) / (0.834 - 0.5) * 100, 1) if not np.isnan(auroc2) else None,
        "n_eval": len(eval_items),
        "n_correct": int(corrects.sum()),
        "n_parseable": int(mask.sum()),
        "total_time_s": round(total_time, 0),
    }

    label = f"balanced_confonly_{args.model_name}"
    with open(os.path.join(args.output_dir, f"{label}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"{label}_responses.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {os.path.join(args.output_dir, label + '_metrics.json')}")


if __name__ == "__main__":
    main()
