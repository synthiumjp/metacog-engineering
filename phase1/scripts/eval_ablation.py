"""
eval_ablation.py — Standalone eval for LoRA ablation configs.
Usage:
    python3 eval_ablation.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16 \
        --adapter-path /path/to/adapters \
        --label gentle_lr
"""
import argparse, json, os, re, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.lora import linear_to_lora_layers
from gen_helpers import generate_greedy
from model_config import MODEL_LAYERS

SEED = 42
N_TEVAL = 1000
MAX_TOKENS = 256

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

_CONFIDENCE_PATTERNS = [
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
    re.compile(r"\b(\d{1,3})\b\s*$"),
]

# --- Parsing ---
def parse_response(raw):
    text = raw.strip()
    answer, confidence = "", float("nan")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        first = lines[0]
        m = re.match(r"^(?:answer\s*:?\s*)(.*)", first, flags=re.IGNORECASE)
        answer = m.group(1).strip() if m else first
        answer = re.sub(r"\s*[,;]?\s*confidence\s*:?.*$", "", answer, flags=re.IGNORECASE).strip()
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

def is_correct_triviaqa(predicted, aliases):
    if not predicted:
        return False
    pred = predicted.lower().strip().rstrip(".")
    return any(pred == a.lower().strip() for a in aliases)

# --- Metrics ---
def auroc2(confidence, correct):
    from sklearn.metrics import roc_auc_score
    c, y = np.asarray(confidence, float), np.asarray(correct, int)
    mask = ~np.isnan(c)
    if mask.sum() < 2 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))

def vrs_screen(confidence, correct):
    c = np.asarray(confidence, float)
    y = np.asarray(correct, int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    n = len(c)
    if n < 10:
        return {"status": "insufficient_data"}
    L = np.mean(c >= 90)
    modal = float(np.bincount(c.astype(int)).argmax()) if n > 0 else 0
    top_mask = c == modal
    TRIN = np.mean(top_mask) if n > 0 else 0
    r = float(np.corrcoef(c, y)[0, 1]) if np.std(c) > 0 and np.std(y) > 0 else 0.0
    invalid = (L > 0.90) or (TRIN > 0.80) or (abs(r) < 0.10)
    valid = (L < 0.50) and (TRIN < 0.50) and (abs(r) > 0.20)
    status = "Valid" if valid else ("Invalid" if invalid else "Indeterminate")
    return {"status": status, "L": round(L, 4), "TRIN": round(TRIN, 4), "r": round(r, 4)}

# --- Data ---
def load_teval_items():
    from datasets import load_dataset
    print("[data] Loading TriviaQA rc.nocontext validation...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    import random; random.seed(SEED); random.shuffle(indices)
    items = []
    for i in indices[:N_TEVAL]:
        row = ds[int(i)]
        items.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "aliases": row["answer"]["aliases"],
        })
    return items

# --- Model loading ---
def load_model_with_adapter(model_path, adapter_path, model_name):
    print(f"[model] Loading: {model_path}")
    model, tokenizer = load(model_path)
    n_layers = MODEL_LAYERS[model_name]
    # Auto-detect LoRA rank from adapter weights
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter not found: {adapter_file}")
    adapter_weights = mx.load(adapter_file)
    rank = next(v.shape[-1] for k, v in adapter_weights.items() if "lora_a" in k)
    print(f"[model] Detected LoRA rank: {rank}")
    config = {"rank": rank, "dropout": 0.05, "scale": 2.0}
    linear_to_lora_layers(model, num_layers=n_layers, config=config)
    model.load_weights(list(adapter_weights.items()), strict=False)
    return model, tokenizer

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="Meta-Llama-3.1-8B-Instruct-bf16")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", default=os.path.expanduser(
        "~/jpwork/metacog-engineering/phase1/results_raw/step4"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items = load_teval_items()
    model, tokenizer = load_model_with_adapter(args.model_path, args.adapter_path, args.model_name)

    print(f"\n=== Evaluating {args.label} ({len(items)} items) ===")
    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        user_msg = TRIVIAQA_PROMPT.format(question=item["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        raw = generate_greedy(model, tokenizer, prompt)
        parsed = parse_response(raw)
        correct = is_correct_triviaqa(parsed["answer"], item["aliases"])
        results.append({
            "question_id": item["question_id"],
            "raw_output": raw,
            "parsed_answer": parsed["answer"],
            "confidence": parsed["confidence"],
            "correct": correct,
        })
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(items)}] {elapsed:.0f}s")

    # Compute metrics
    confidences = np.array([r["confidence"] for r in results])
    corrects = np.array([r["correct"] for r in results], dtype=int)
    accuracy = corrects.mean()
    parse_rate = (~np.isnan(confidences)).mean()
    auc = auroc2(confidences, corrects)
    vrs = vrs_screen(confidences, corrects)
    conf_mean = float(np.nanmean(confidences)) if parse_rate > 0 else None
    conf_std = float(np.nanstd(confidences)) if parse_rate > 0 else None

    # Probe recovery
    baseline_auroc2 = 0.711  # Llama 8B baseline
    probe_auroc2 = 0.843     # Llama 8B probe (primary)
    recovery = (auc - 0.5) / (probe_auroc2 - 0.5) * 100 if not np.isnan(auc) else None
    acc_drop = accuracy - 0.782  # baseline accuracy

    print(f"\n{'='*50}")
    print(f"  Label:       {args.label}")
    print(f"  Accuracy:    {accuracy:.3f} (drop: {acc_drop:+.1%})")
    print(f"  Parse rate:  {parse_rate:.3f}")
    print(f"  AUROC₂:      {auc:.3f}")
    print(f"  VRS:         {vrs['status']} (L={vrs.get('L','n/a')}, TRIN={vrs.get('TRIN','n/a')}, r={vrs.get('r','n/a')})")
    print(f"  Conf mean:   {conf_mean:.1f}" if conf_mean else "  Conf mean:   n/a")
    print(f"  Conf std:    {conf_std:.1f}" if conf_std else "  Conf std:    n/a")
    print(f"  Recovery:    {recovery:.1f}%" if recovery else "  Recovery:    n/a")
    print(f"{'='*50}")

    # Save
    out = {
        "label": args.label, "accuracy": accuracy, "parse_rate": parse_rate,
        "auroc2": auc, "vrs": vrs, "conf_mean": conf_mean, "conf_std": conf_std,
        "recovery_pct": recovery, "acc_drop": acc_drop,
    }
    outpath = os.path.join(args.output_dir, f"ablation_{args.label}_metrics.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {outpath}")

    # Also save raw responses
    rawpath = os.path.join(args.output_dir, f"ablation_{args.label}_responses.json")
    with open(rawpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {rawpath}")

if __name__ == "__main__":
    main()
