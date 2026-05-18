"""
prep_multitask_ptcsft.py — Prepare multi-task PT-CSFT training data.

Generates probe-targeted confidence labels on a NEW domain (GSM8K or ARC),
then combines with existing TriviaQA PT-CSFT training data for multi-task
LoRA fine-tuning.

Pipeline:
    1. Load model, run inference on new-domain cal items, extract hidden states
    2. Fit probe on hidden states → P(correct) per item
    3. Format as training JSONL (same format as TriviaQA PT-CSFT)
    4. Combine with existing TriviaQA training JSONL
    5. Output shuffled train/valid/test splits + LoRA config

Usage:
    python3 prep_multitask_ptcsft.py \
        --benchmark gsm8k \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --model-name gemma-3-12b-it \
        --triviaqa-jsonl ~/jpwork/metacog-engineering/phase1/results_raw/finetune/gemma-3-12b-it/probe_target/data/train.jsonl \
        --output-dir ~/jpwork/metacog-engineering/phase1/results_raw/finetune/gemma-3-12b-it/multitask_probe_target

Requires: mlx_lm, datasets, sklearn, numpy, yaml
Run from: ~/jpwork/metacog-engineering/phase1/scripts/
"""
import argparse, json, os, random, re, sys, time
import numpy as np
import yaml
import mlx.core as mx
from mlx_lm import load
from gen_helpers import generate_greedy
from model_config import MODEL_LAYERS, get_model_config
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

SEED = 42

# ---------------------------------------------------------------------------
# Prompts (same as probe_check_domain.py)
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
# Data loading (same as probe_check_domain.py)
# ---------------------------------------------------------------------------
def load_gsm8k(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
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
    if not predicted or not gold:
        return False
    match = re.search(r"####\s*(.+?)(?:\n|$)", predicted)
    if match:
        pred_num = match.group(1).strip().replace(",", "")
    else:
        numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", predicted)
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


# ---------------------------------------------------------------------------
# Hidden state extraction (same as probe_check_domain.py)
# ---------------------------------------------------------------------------
def extract_hidden_states(model, tokenizer, prompt, response_text, layer_indices, model_cfg):
    full_text = prompt + response_text
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)
    if len(tokens) <= prompt_len:
        return None
    x = mx.array([tokens])
    lm = model_cfg["lm"]
    layers = model_cfg["layers"]
    h = lm.model.embed_tokens(x)
    if model_cfg["scale_embeddings"]:
        hidden_size = h.shape[-1]
        h = h * (hidden_size ** 0.5)
    hidden_states = {}
    for i, layer in enumerate(layers):
        h = layer(h, cache=None)
        for label, idx in layer_indices.items():
            if i == idx:
                last_pos = len(tokens) - 1
                hidden_states[f"{label}_last"] = np.array(
                    h[0, last_pos].astype(mx.float32)
                )
    mx.eval(h)
    return hidden_states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-task PT-CSFT training data")
    parser.add_argument("--benchmark", required=True, choices=["gsm8k", "arc"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--triviaqa-jsonl", required=True,
                        help="Path to existing TriviaQA PT-CSFT train.jsonl")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for combined training data")
    parser.add_argument("--probe-layer", default="middle_last",
                        help="Which layer config to use for probe targets")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for LoRA config")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Multi-Task PT-CSFT Data Preparation")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Model: {args.model_name}")
    print(f"  Probe layer: {args.probe_layer}")
    print(f"{'='*60}\n")

    # ── Load data ──
    print(f"[data] Loading {args.benchmark}...")
    if args.benchmark == "gsm8k":
        cal_items, eval_items = load_gsm8k()
    else:
        cal_items, eval_items = load_arc()
    print(f"  Cal: {len(cal_items)} items, Eval: {len(eval_items)} items\n")

    # ── Load model ──
    print(f"[model] Loading {args.model_path}...")
    model, tokenizer = load(args.model_path)
    model_cfg = get_model_config(model, tokenizer, args.model_name)
    layer_indices = model_cfg["layer_indices"]
    print(f"  Layers: {model_cfg['n_layers']}, indices: {layer_indices}\n")

    is_correct_fn = is_correct_gsm8k if args.benchmark == "gsm8k" else is_correct_arc

    # ── Run inference on cal set with hidden state extraction ──
    print(f"=== Cal set inference ({len(cal_items)} items) ===")
    cal_results = []
    t0 = time.time()

    for i, item in enumerate(cal_items):
        if args.benchmark == "gsm8k":
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
        correct = is_correct_fn(raw, item["gold_answer"])

        try:
            hs = extract_hidden_states(model, tokenizer, prompt, raw,
                                       layer_indices, model_cfg)
        except Exception:
            hs = None

        cal_results.append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "raw_output": raw,
            "correct": correct,
            "hidden_states": hs,
            "user_msg": user_msg,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in cal_results])
            n_cached = sum(1 for r in cal_results if r["hidden_states"] is not None)
            print(f"  [{i+1}/{len(cal_items)}]  acc={acc:.3f}  "
                  f"cached={n_cached}/{i+1}  elapsed={elapsed:.0f}s")

    cal_acc = np.mean([r["correct"] for r in cal_results])
    n_with_hs = sum(1 for r in cal_results
                    if r["hidden_states"] is not None
                    and args.probe_layer in r["hidden_states"])
    print(f"\n  Cal accuracy: {cal_acc:.3f}")
    print(f"  Items with hidden states ({args.probe_layer}): {n_with_hs}/{len(cal_results)}")

    # ── Fit probe ──
    print(f"\n=== Probe training ({args.probe_layer}) ===")
    X_train, y_train, item_indices = [], [], []
    for idx, r in enumerate(cal_results):
        if r["hidden_states"] is not None and args.probe_layer in r["hidden_states"]:
            X_train.append(r["hidden_states"][args.probe_layer])
            y_train.append(int(r["correct"]))
            item_indices.append(idx)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  Training probe: n={len(X_train)}, dim={X_train.shape[1]}, "
          f"pos_rate={y_train.mean():.3f}")

    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=5, penalty="l2", max_iter=1000, random_state=SEED,
    )
    clf.fit(X_train, y_train)

    # Get P(correct) for each cal item with hidden states
    p_correct = clf.predict_proba(X_train)[:, 1]
    print(f"  Probe P(correct) range: [{p_correct.min():.3f}, {p_correct.max():.3f}]")
    print(f"  Probe P(correct) mean: {p_correct.mean():.3f}, std: {p_correct.std():.3f}")

    # ── Generate training JSONL for new domain ──
    print(f"\n=== Generating {args.benchmark} training JSONL ===")
    new_domain_items = []
    for j, (idx, p) in enumerate(zip(item_indices, p_correct)):
        r = cal_results[idx]
        conf_target = max(0, min(100, round(p * 100)))
        new_domain_items.append({
            "messages": [
                {"role": "user", "content": r["user_msg"]},
                {"role": "assistant",
                 "content": f"{r['raw_output'].strip()}\nConfidence: {conf_target}%"},
            ]
        })

    print(f"  {args.benchmark} items: {len(new_domain_items)}")

    # ── Load existing TriviaQA JSONL ──
    print(f"\n=== Loading TriviaQA training data ===")
    trivia_items = []
    with open(args.triviaqa_jsonl) as f:
        for line in f:
            trivia_items.append(json.loads(line))
    print(f"  TriviaQA items: {len(trivia_items)}")

    # ── Combine and shuffle ──
    combined = trivia_items + new_domain_items
    random.seed(SEED)
    random.shuffle(combined)
    print(f"  Combined: {len(combined)} items "
          f"({len(trivia_items)} TriviaQA + {len(new_domain_items)} {args.benchmark})")

    # Split: 90% train, 5% valid, 5% test
    n = len(combined)
    n_valid = max(20, n // 20)
    n_test = max(20, n // 20)
    n_train = n - n_valid - n_test

    train_items = combined[:n_train]
    valid_items = combined[n_train:n_train + n_valid]
    test_items = combined[n_train + n_valid:]

    # Write JSONL
    for split_name, items in [("train", train_items),
                               ("valid", valid_items),
                               ("test", test_items)]:
        path = os.path.join(data_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"  Wrote {path}: {len(items)} items")

    # ── Generate LoRA config ──
    config = {
        "model": args.model_path,
        "train": True,
        "data": data_dir,
        "seed": SEED,
        "lora_layers": model_cfg["n_layers"],
        "batch_size": 1,
        "iters": len(train_items) * 3 // 1,  # 3 epochs, batch_size 1
        "val_batches": 25,
        "learning_rate": args.lr,
        "steps_per_report": 50,
        "steps_per_eval": len(train_items) // 2,  # eval ~2x per epoch
        "adapter_path": os.path.join(args.output_dir, "adapters"),
        "save_every": len(train_items),  # save each epoch
        "grad_checkpoint": True,
        "lora_parameters": {
            "rank": args.rank,
            "scale": 2.0,
            "dropout": 0.05,
            "keys": ["self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        },
    }
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n  Config: {config_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  READY TO TRAIN")
    print(f"{'='*60}")
    print(f"  Training data: {len(train_items)} items "
          f"({len(trivia_items)} TriviaQA + {len(new_domain_items)} {args.benchmark})")
    print(f"  Validation: {len(valid_items)} items")
    print(f"  Test: {len(test_items)} items")
    print(f"  LoRA rank: {args.rank}, lr: {args.lr}")
    print(f"\n  Run training:")
    print(f"    python3 -m mlx_lm lora --config {config_path}")
    print(f"\n  Then evaluate on both benchmarks:")
    print(f"    python3 eval_zeroshot_transfer.py --benchmark gsm8k \\")
    print(f"        --model-path {args.model_path} \\")
    print(f"        --model-name {args.model_name} \\")
    print(f"        --adapter-path {os.path.join(args.output_dir, 'adapters')} \\")
    print(f"        --baseline-auroc2 0.546")

    # Save metadata
    meta = {
        "benchmark": args.benchmark,
        "model": args.model_name,
        "probe_layer": args.probe_layer,
        "n_triviaqa": len(trivia_items),
        "n_new_domain": len(new_domain_items),
        "n_combined": len(combined),
        "cal_accuracy": round(cal_acc, 3),
        "probe_p_correct_mean": round(float(p_correct.mean()), 3),
        "probe_p_correct_std": round(float(p_correct.std()), 3),
        "lr": args.lr,
        "rank": args.rank,
    }
    meta_path = os.path.join(args.output_dir, "multitask_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
