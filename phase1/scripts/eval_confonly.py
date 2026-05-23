"""
eval_confonly.py — Two-pass evaluation for confidence-only PT-CSFT.
"""
import argparse, json, os, random, re, time
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate as mlx_generate
from sklearn.metrics import roc_auc_score

SEED = 42
MAX_TOKENS_ANSWER = 1024
MAX_TOKENS_CONF = 30

def _greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)

GSM8K_PROMPT = (
    "Solve this math problem step by step. "
    "After your final answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)
ARC_PROMPT = (
    "Answer the following science question by selecting the letter of the correct answer. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
    "{choices}\n"
)

def load_gsm8k(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        answer_text = row["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({"id": f"gsm8k_{len(items)}", "question": row["question"],
                      "gold_answer": gold, "prompt_template": GSM8K_PROMPT})
    random.seed(seed)
    random.shuffle(items)
    return items[:800], items[800:]

def load_arc(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        choices_text = "\n".join(f"{label}) {text}"
            for label, text in zip(row["choices"]["label"], row["choices"]["text"]))
        items.append({"id": row["id"], "question": row["question"],
                      "gold_answer": row["answerKey"], "choices_text": choices_text,
                      "prompt_template": ARC_PROMPT})
    random.seed(seed)
    random.shuffle(items)
    return items[:700], items[700:]

def is_correct_gsm8k(predicted, gold):
    if not predicted or not gold:
        return False
    predicted_clean = re.split(r'(?i)\bconfidence\b', predicted)[0]
    match = re.search(r"####\s*(.+?)(?:\n|$)", predicted_clean)
    if match:
        pred_num = match.group(1).strip().replace(",", "")
    else:
        numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", predicted_clean)
        if not numbers:
            return False
        pred_num = numbers[-1].replace(",", "")
    gold_clean = gold.replace(",", "").strip()
    try:
        return float(pred_num) == float(gold_clean)
    except ValueError:
        return pred_num.strip() == gold_clean.strip()

def is_correct_arc(predicted, gold):
    if not predicted or not gold:
        return False
    pred = predicted.strip().upper()
    match = re.match(r"^([A-D])", pred)
    if match:
        return match.group(1) == gold.upper()
    for letter in ["A", "B", "C", "D"]:
        if f"answer is {letter}" in pred.upper() or f"answer: {letter}" in pred.upper():
            return letter == gold.upper()
    return False

_CONF_PATTERNS = [
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
    re.compile(r"^(\d{1,3})\s*$", re.MULTILINE),
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["gsm8k", "arc"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--baseline-auroc2", type=float, default=None)
    parser.add_argument("--strip-cot", action="store_true",
                        help="Use only final numeric answer in confidence prompt")
    parser.add_argument("--output-dir", default=os.path.expanduser(
        "~/jpwork/metacog-engineering/phase1/results_raw/domain_gen"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Two-Pass Confidence-Only Evaluation")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Model: {args.model_name}")
    print(f"  Strip CoT: {args.strip_cot}")
    print(f"{'='*60}\n")

    print(f"[data] Loading {args.benchmark}...")
    if args.benchmark == "gsm8k":
        _, eval_items = load_gsm8k()
    else:
        _, eval_items = load_arc()
    print(f"  Eval: {len(eval_items)} items\n")
    is_correct_fn = is_correct_gsm8k if args.benchmark == "gsm8k" else is_correct_arc

    # Pass 1: Generate answers WITHOUT adapter
    print(f"=== Pass 1: Generate answers (no adapter) ===")
    model, tokenizer = load(args.model_path)
    results = []
    t0 = time.time()
    for i, item in enumerate(eval_items):
        if args.benchmark == "gsm8k":
            user_msg = item["prompt_template"].format(question=item["question"])
        else:
            user_msg = item["prompt_template"].format(
                question=item["question"], choices=item["choices_text"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True)
        raw = mlx_generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_TOKENS_ANSWER, verbose=False,
                           sampler=_greedy_sampler)
        correct = is_correct_fn(raw, item["gold_answer"])
        results.append({"id": item["id"], "question": item["question"],
                        "gold": item["gold_answer"], "raw_answer": raw,
                        "correct": correct})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in results])
            print(f"  [{i+1}/{len(eval_items)}]  acc={acc:.3f}  elapsed={elapsed:.0f}s")

    acc_pass1 = np.mean([r["correct"] for r in results])
    print(f"\n  Pass 1 accuracy: {acc_pass1:.3f}")
    print(f"  Pass 1 time: {time.time() - t0:.0f}s")
    del model
    mx.clear_cache()

    # Pass 2: Rate confidence WITH adapter
    print(f"\n=== Pass 2: Rate confidence (with adapter) ===")
    model_adapted, tokenizer = load(args.model_path, adapter_path=args.adapter_path)
    t1 = time.time()
    for i, r in enumerate(results):
        answer_text = re.split(r'(?i)\bconfidence\b', r["raw_answer"])[0].strip()

        if args.strip_cot:
            numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer_text)
            final_answer = numbers[-1].replace(",", "") if numbers else answer_text
            conf_prompt_text = (
                "You solved a math question.\n"
                f"Question: {r['question']}\n"
                f"Your final answer: {final_answer}\n"
                f"How confident are you that {final_answer} is correct? "
                "State your confidence as a percentage from 0 to 100."
            )
        else:
            conf_prompt_text = (
                "You answered the following math question.\n"
                f"Question: {r['question']}\n"
                f"Your answer: {answer_text}\n"
                "How confident are you that your answer is correct? "
                "State your confidence as a percentage from 0 to 100."
            )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": conf_prompt_text}],
            tokenize=False, add_generation_prompt=True)
        conf_response = mlx_generate(model_adapted, tokenizer, prompt=prompt,
                                      max_tokens=MAX_TOKENS_CONF, verbose=False,
                                      sampler=_greedy_sampler)
        conf = parse_confidence(conf_response)
        r["conf_response"] = conf_response
        r["confidence"] = conf
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t1
            confs = np.array([r2["confidence"] for r2 in results[:i+1]])
            parse_rate = (~np.isnan(confs)).mean()
            print(f"  [{i+1}/{len(results)}]  parse={parse_rate:.3f}  "
                  f"conf_mean={np.nanmean(confs):.1f}  elapsed={elapsed:.0f}s")

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

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.benchmark.upper()} confidence-only two-pass")
    print(f"  Model: {args.model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {acc_pass1:.3f}  (no adapter — preserved)")
    print(f"  Parse rate:     {parse_rate:.3f}")
    print(f"  Verbal AUROC₂:  {auroc2:.3f}")
    print(f"  Conf mean:      {conf_mean:.1f}")
    print(f"  Conf std:       {conf_std:.1f}")
    if args.baseline_auroc2 is not None and not np.isnan(auroc2):
        delta = auroc2 - args.baseline_auroc2
        print(f"\n  Baseline AUROC₂: {args.baseline_auroc2:.3f}")
        print(f"  Delta:           {delta:+.3f}")
        if delta > 0.02:
            print(f"\n  >>> TRANSFER DETECTED: AUROC₂ improved by {delta:+.3f}")
        elif delta > -0.02:
            print(f"\n  >>> NO CLEAR TRANSFER: within noise ({delta:+.3f})")
        else:
            print(f"\n  >>> NEGATIVE TRANSFER: degraded ({delta:+.3f})")
    print(f"\n  Total time: {total_time:.0f}s")

    summary = {"benchmark": args.benchmark, "model": args.model_name,
               "adapter_path": args.adapter_path, "method": "confidence_only_two_pass",
               "strip_cot": args.strip_cot, "accuracy": round(acc_pass1, 3),
               "parse_rate": round(parse_rate, 3),
               "verbal_auroc2": round(auroc2, 3) if not np.isnan(auroc2) else None,
               "conf_mean": round(conf_mean, 1), "conf_std": round(conf_std, 1),
               "baseline_auroc2": args.baseline_auroc2,
               "delta": round(auroc2 - args.baseline_auroc2, 3) if args.baseline_auroc2 and not np.isnan(auroc2) else None,
               "n_eval": len(eval_items), "n_correct": int(corrects.sum()),
               "n_parseable": int(mask.sum())}
    label = f"confonly_{args.benchmark}_{args.model_name}"
    if args.strip_cot:
        label += "_stripped"
    with open(os.path.join(args.output_dir, f"{label}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"{label}_responses.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {os.path.join(args.output_dir, label + '.json')}")

if __name__ == "__main__":
    main()
