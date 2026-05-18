"""
probe_check_domain.py — Probe-first gating for domain generalisation.
Runs baseline + probe on GSM8K or ARC-Challenge to check if PT-CSFT
has signal to work with before committing to full fine-tuning.

Usage:
    python3 probe_check_domain.py \
        --benchmark gsm8k \
        --model-path ~/jpwork/models/Qwen2.5-32B-Instruct-bf16 \
        --model-name Qwen2.5-32B-Instruct-bf16

    python3 probe_check_domain.py \
        --benchmark arc \
        --model-path ~/mnt/models-lan/.../gemma-3-12b-it \
        --model-name gemma-3-12b-it

Gate: if probe AUROC₂ > 0.65, proceed to PT-CSFT.
"""
import argparse, json, os, random, re, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm import generate
from model_config import MODEL_LAYERS, get_model_config
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

SEED = 42
MAX_TOKENS = 1024  # GSM8K needs longer for chain-of-thought

def generate_greedy(model, tokenizer, prompt):
    sampler = lambda logits: mx.argmax(logits, axis=-1)
    return generate(model, tokenizer, prompt=prompt,
                    max_tokens=MAX_TOKENS, sampler=sampler, verbose=False)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gsm8k(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        # Extract numeric answer from "#### number" format
        answer_text = row["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({
            "id": f"gsm8k_{len(items)}",
            "question": row["question"],
            "gold_answer": gold,
            "prompt_template": GSM8K_PROMPT,
        })
    random.seed(seed)
    random.shuffle(items)
    # Split: 800 cal, rest eval
    n_cal = 800
    return items[:n_cal], items[n_cal:]

def load_arc(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        choices_text = "\n".join(
            f"{label}) {text}"
            for label, text in zip(row["choices"]["label"], row["choices"]["text"])
        )
        items.append({
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["answerKey"],
            "choices_text": choices_text,
            "prompt_template": ARC_PROMPT,
        })
    random.seed(seed)
    random.shuffle(items)
    n_cal = 700
    return items[:n_cal], items[n_cal:]

# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def is_correct_gsm8k(predicted, gold):
    """Extract number from prediction and compare to gold."""
    if not predicted or not gold:
        return False
    # Strip confidence line before extracting answer
    text = re.sub(r"[Cc]onfidence.*$", "", predicted, flags=re.DOTALL).strip()
    # Try "Final Answer" pattern
    match = re.search(r"[Ff]inal [Aa]nswer[:\s]*([\d,.-]+)", text)
    if match:
        pred_num = match.group(1).strip().replace(",", "")
    else:
        # Try "#### number" format
        match = re.search(r"####\s*(.+?)(?:\n|$)", text)
        if match:
            pred_num = match.group(1).strip().replace(",", "")
        else:
            # Last number in text (after stripping confidence)
            numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
            if not numbers:
                return False
            pred_num = numbers[-1].replace(",", "")
    gold_clean = gold.replace(",", "").strip()
    try:
        return abs(float(pred_num) - float(gold_clean)) < 0.01
    except ValueError:
        return pred_num.strip() == gold_clean.strip()

def is_correct_arc(predicted, gold):
    """Check if predicted letter matches gold."""
    if not predicted or not gold:
        return False
    pred = predicted.strip()
    # Strip markdown bold markers
    clean = re.sub(r'\*+', '', pred).upper()
    # Try 'answer is X' pattern
    match = re.search(r'ANSWER IS\s*([A-D])', clean)
    if match:
        return match.group(1) == gold.upper()
    # Try 'X)' at start
    match = re.match(r'^([A-D])\)', clean)
    if match:
        return match.group(1) == gold.upper()
    # First standalone A-D
    match = re.search(r'([A-D])\)', clean)
    if match:
        return match.group(1) == gold.upper()
    return False

# ---------------------------------------------------------------------------
# Confidence parsing (reuse from eval_ablation)
# ---------------------------------------------------------------------------
_CONFIDENCE_PATTERNS = [
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
]

def parse_confidence(text):
    for pat in _CONFIDENCE_PATTERNS:
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
# Hidden state extraction
# ---------------------------------------------------------------------------
def extract_hidden_states(model, tokenizer, prompt, response_text, layer_indices, model_cfg):
    """Extract hidden states at specified layers for the last token of the response."""
    full_text = prompt + response_text
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    if len(tokens) <= prompt_len:
        return None

    x = mx.array([tokens])
    lm = model_cfg["lm"]
    layers = model_cfg["layers"]

    # Get embeddings
    h = lm.model.embed_tokens(x)
    if model_cfg["scale_embeddings"]:
        hidden_size = h.shape[-1]
        h = h * (hidden_size ** 0.5)

    hidden_states = {}
    for i, layer in enumerate(layers):
        h = layer(h, cache=None)
        for label, idx in layer_indices.items():
            if i == idx:
                # Last answer token position
                last_pos = len(tokens) - 1
                hidden_states[f"{label}_last"] = np.array(
                    h[0, last_pos].astype(mx.float32)
                )
    mx.eval(h)  # ensure computation completes
    return hidden_states

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_baseline(model, tokenizer, items, benchmark, model_cfg, layer_indices):
    """Generate responses and extract hidden states."""
    is_correct_fn = is_correct_gsm8k if benchmark == "gsm8k" else is_correct_arc
    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        if benchmark == "gsm8k":
            user_msg = item["prompt_template"].format(question=item["question"])
        else:
            user_msg = item["prompt_template"].format(
                question=item["question"],
                choices=item["choices_text"],
            )

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        raw = generate_greedy(model, tokenizer, prompt)
        conf = parse_confidence(raw)
        correct = is_correct_fn(raw, item["gold_answer"])

        # Extract hidden states
        try:
            hs = extract_hidden_states(model, tokenizer, prompt, raw, layer_indices, model_cfg)
        except Exception as e:
            hs = None

        results.append({
            "id": item["id"],
            "raw_output": raw,
            "gold": item["gold_answer"],
            "correct": correct,
            "confidence": conf,
            "hidden_states": hs,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in results])
            print(f"  [{i+1}/{len(items)}]  acc={acc:.3f}  elapsed={elapsed:.0f}s")

    return results

def train_and_eval_probes(cal_results, eval_results, layer_indices):
    """Train probes on cal hidden states, evaluate on eval."""
    # Collect hidden states by layer config
    probe_results = {}

    for label in layer_indices:
        key = f"{label}_last"

        # Build training data
        X_train, y_train = [], []
        for r in cal_results:
            if r["hidden_states"] is not None and key in r["hidden_states"]:
                X_train.append(r["hidden_states"][key])
                y_train.append(int(r["correct"]))

        # Build eval data
        X_eval, y_eval, conf_eval = [], [], []
        for r in eval_results:
            if r["hidden_states"] is not None and key in r["hidden_states"]:
                X_eval.append(r["hidden_states"][key])
                y_eval.append(int(r["correct"]))
                conf_eval.append(r["confidence"])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_eval, y_eval = np.array(X_eval), np.array(y_eval)
        conf_eval = np.array(conf_eval)

        if len(X_train) < 50 or len(X_eval) < 30:
            print(f"  {key}: insufficient data (train={len(X_train)}, eval={len(X_eval)})")
            continue

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print(f"  {key}: degenerate labels (all same)")
            continue

        # Train probe
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
                cv=5, penalty="l2", max_iter=2000,
                random_state=SEED,
            ))
        ])
        clf.fit(X_train, y_train)

        # Evaluate
        probe_scores = clf.predict_proba(X_eval)[:, 1]
        mask = ~np.isnan(conf_eval)

        if y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
            print(f"  {key}: degenerate eval labels")
            continue

        probe_auc = roc_auc_score(y_eval, probe_scores)

        # Verbal AUROC₂ on same items
        if mask.sum() > 10 and y_eval[mask].sum() > 0 and y_eval[mask].sum() < mask.sum():
            verbal_auc = roc_auc_score(y_eval[mask], conf_eval[mask])
        else:
            verbal_auc = float("nan")

        gap = probe_auc - verbal_auc if not np.isnan(verbal_auc) else probe_auc - 0.5

        probe_results[key] = {
            "probe_auroc2": round(probe_auc, 3),
            "verbal_auroc2": round(verbal_auc, 3),
            "gap": round(gap, 3),
            "n_train": len(X_train),
            "n_eval": len(X_eval),
            "pos_rate_train": round(y_train.mean(), 3),
            "pos_rate_eval": round(y_eval.mean(), 3),
        }
        print(f"  {key}: probe={probe_auc:.3f}  verbal={verbal_auc:.3f}  gap={gap:+.3f}  "
              f"(n_train={len(X_train)}, n_eval={len(X_eval)})")

    return probe_results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["gsm8k", "arc"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default=os.path.expanduser(
        "~/jpwork/metacog-engineering/phase1/results_raw/domain_gen"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Domain Generalisation Probe Check")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Model: {args.model_name}")
    print(f"{'='*60}\n")

    # Load data
    print(f"[data] Loading {args.benchmark}...")
    if args.benchmark == "gsm8k":
        cal_items, eval_items = load_gsm8k()
    else:
        cal_items, eval_items = load_arc()
    print(f"  Cal: {len(cal_items)} items, Eval: {len(eval_items)} items\n")

    # Load model
    print(f"[model] Loading {args.model_path}...")
    model, tokenizer = load(args.model_path)
    model_cfg = get_model_config(model, tokenizer, args.model_name)
    layer_indices = model_cfg["layer_indices"]
    print(f"  Layers: {model_cfg['n_layers']}, indices: {layer_indices}\n")

    # Run baseline on cal set (with hidden states)
    print(f"=== Calibration set ({len(cal_items)} items, with hidden states) ===")
    cal_results = run_baseline(model, tokenizer, cal_items, args.benchmark,
                               model_cfg, layer_indices)

    cal_acc = np.mean([r["correct"] for r in cal_results])
    cal_confs = np.array([r["confidence"] for r in cal_results])
    cal_correct = np.array([r["correct"] for r in cal_results], dtype=int)
    cal_parse = (~np.isnan(cal_confs)).mean()
    print(f"\n  Cal accuracy: {cal_acc:.3f}")
    print(f"  Cal parse rate: {cal_parse:.3f}")
    print(f"  Cal conf mean: {np.nanmean(cal_confs):.1f}")

    # Run baseline on eval set (with hidden states)
    print(f"\n=== Evaluation set ({len(eval_items)} items, with hidden states) ===")
    eval_results = run_baseline(model, tokenizer, eval_items, args.benchmark,
                                model_cfg, layer_indices)

    eval_acc = np.mean([r["correct"] for r in eval_results])
    eval_confs = np.array([r["confidence"] for r in eval_results])
    eval_correct = np.array([r["correct"] for r in eval_results], dtype=int)
    eval_parse = (~np.isnan(eval_confs)).mean()

    mask = ~np.isnan(eval_confs)
    if mask.sum() > 10 and eval_correct[mask].sum() > 0 and eval_correct[mask].sum() < mask.sum():
        verbal_auc = roc_auc_score(eval_correct[mask], eval_confs[mask])
    else:
        verbal_auc = float("nan")

    print(f"\n  Eval accuracy: {eval_acc:.3f}")
    print(f"  Eval parse rate: {eval_parse:.3f}")
    print(f"  Eval verbal AUROC₂: {verbal_auc:.3f}")
    print(f"  Eval conf mean: {np.nanmean(eval_confs):.1f}, std: {np.nanstd(eval_confs):.1f}")

    # Train and evaluate probes
    print(f"\n=== Probe training and evaluation ===")
    probe_results = train_and_eval_probes(cal_results, eval_results, layer_indices)

    # Summary
    best_key = max(probe_results, key=lambda k: probe_results[k]["probe_auroc2"]) if probe_results else None
    best_auc = probe_results[best_key]["probe_auroc2"] if best_key else 0

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {args.benchmark.upper()} on {args.model_name}")
    print(f"{'='*60}")
    print(f"  Eval accuracy:      {eval_acc:.3f}")
    print(f"  Verbal AUROC₂:      {verbal_auc:.3f}")
    print(f"  Best probe AUROC₂:  {best_auc:.3f} ({best_key})")
    print(f"  Gap:                {best_auc - verbal_auc:+.3f}" if not np.isnan(verbal_auc) else f"  Gap:                {best_auc - 0.5:+.3f}")
    print(f"  Parse rate:         {eval_parse:.3f}")

    if best_auc > 0.65:
        print(f"\n  >>> GATE PASSED: Probe discriminates (AUROC₂={best_auc:.3f} > 0.65)")
        print(f"  >>> Proceed to PT-CSFT on {args.benchmark}")
    else:
        print(f"\n  >>> GATE FAILED: Probe near chance (AUROC₂={best_auc:.3f} ≤ 0.65)")
        print(f"  >>> PT-CSFT unlikely to help on {args.benchmark}")

    # Save
    summary = {
        "benchmark": args.benchmark,
        "model": args.model_name,
        "eval_accuracy": round(eval_acc, 3),
        "eval_verbal_auroc2": round(verbal_auc, 3),
        "eval_parse_rate": round(eval_parse, 3),
        "eval_conf_mean": round(float(np.nanmean(eval_confs)), 1),
        "eval_conf_std": round(float(np.nanstd(eval_confs)), 1),
        "probe_results": probe_results,
        "best_probe_key": best_key,
        "best_probe_auroc2": round(best_auc, 3),
        "gate_passed": best_auc > 0.65,
    }
    outpath = os.path.join(args.output_dir,
                           f"probe_check_{args.benchmark}_{args.model_name}.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {outpath}")

    # Save raw responses for potential rescoring
    responses_path = os.path.join(args.output_dir,
                                   f"responses_{args.benchmark}_{args.model_name}.json")
    save_responses = [{k: v for k, v in r.items() if k != "hidden_states"}
                      for r in cal_results + eval_results]
    with open(responses_path, "w") as f:
        json.dump(save_responses, f, indent=2)
    print(f"  Saved responses: {responses_path}")


if __name__ == "__main__":
    main()
