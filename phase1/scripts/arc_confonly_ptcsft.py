"""
arc_confonly_ptcsft.py — ARC-Challenge confidence-only PT-CSFT pipeline.

Qwen 7B passed the ARC probe gate at 0.834 (first_last). This script:
  1. Loads ARC-Challenge data, splits into cal/eval
  2. Generates baseline responses (greedy) on both splits
  3. Trains probe on cal hidden states
  4. Derives confidence targets from probe on cal items
  5. Prepares confonly training JSONL
  6. Trains LoRA adapter (via mlx_lm)
  7. Evals: two-pass (generate answer sans adapter, rate with adapter)

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/arc_confonly_ptcsft.py --stage all
    # or run individual stages:
    python3 scripts/arc_confonly_ptcsft.py --stage baseline
    python3 scripts/arc_confonly_ptcsft.py --stage probe
    python3 scripts/arc_confonly_ptcsft.py --stage prep
    python3 scripts/arc_confonly_ptcsft.py --stage train
    python3 scripts/arc_confonly_ptcsft.py --stage eval

Environment:
    source ~/jpwork/.venv_metacog/bin/activate
    Model: Qwen 2.5 7B Instruct (~/jpwork/models/Qwen2.5-7B-Instruct)
    Hardware: M3 Ultra 512GB, MLX
"""

import argparse, json, os, re, sys, time
import numpy as np

SEED = 42
MODEL_PATH = os.path.expanduser("~/mnt/models-lan/foresight/synthesis-archive/Qwen2.5-7B-Instruct-bf16")
MODEL_NAME = "Qwen2.5-7B-Instruct-bf16"

# ARC gate check showed first_last was best probe (unusual — not middle)
PROBE_LAYER = "first"
PROBE_POS = "last"

# Paths
RESULTS_DIR = "results_raw/domain_gen/arc_confonly"
ADAPTER_DIR = f"finetune/{MODEL_NAME}/arc_confonly"
TRAIN_JSONL = f"{ADAPTER_DIR}/train.jsonl"
VALID_JSONL = f"{ADAPTER_DIR}/valid.jsonl"

# LoRA config — use gentle-lr (works for Qwen family)
LORA_CONFIG = {
    "rank": 16,
    "lr": 5e-5,  # gentle
    "scale": 2.0,
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,  # effective batch = 16
}

# Splits: 500 cal, 500 eval (ARC-Challenge test has 1172 items)
N_CAL = 500
N_EVAL = 500
MAX_TOKENS = 256

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_arc(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        choices = row["choices"]
        choices_text = "\n".join(
            f"{l}. {t}" for l, t in zip(choices["label"], choices["text"])
        )
        items.append({
            "id": f"arc_{len(items)}",
            "question": row["question"],
            "choices_text": choices_text,
            "gold_answer": row["answerKey"],
        })
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    return items

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ARC_BASELINE_PROMPT = (
    "Answer the following science question by selecting the letter of the correct answer. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
    "{choices}\n"
)

ARC_CONFONLY_USER = (
    "You answered the following science question.\n\n"
    "Question: {question}\n"
    "{choices}\n\n"
    "Your answer: {answer}\n\n"
    "How confident are you in this answer? "
    "State your confidence as a percentage from 0 to 100."
)

ARC_CONFONLY_ASSISTANT = "Confidence: {confidence}%"

# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def is_correct_arc(predicted, gold):
    if not predicted or not gold:
        return False
    pred = predicted.strip().upper()
    # Try direct letter match
    m = re.match(r"^([A-D])", pred)
    if m:
        return m.group(1) == gold.upper()
    # Try "answer is X" or "answer: X"
    for letter in ["A", "B", "C", "D"]:
        if f"ANSWER IS {letter}" in pred.upper() or f"ANSWER: {letter}" in pred.upper():
            return letter == gold.upper()
    return False

def parse_answer_letter(text):
    """Extract the answer letter from model output."""
    text = text.strip()
    # Direct letter at start
    m = re.match(r"^([A-D])\b", text.upper())
    if m:
        return m.group(1)
    # "The answer is X"
    m = re.search(r"answer\s+is\s+([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "Answer: X"
    m = re.search(r"answer\s*:\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""

def parse_confidence(text):
    pats = [
        re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
        re.compile(r"(\d{1,3})\s*%"),
    ]
    for pat in pats:
        m = pat.search(text)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return float(v)
    return float("nan")

# ---------------------------------------------------------------------------
# Stage 1: Baseline generation
# ---------------------------------------------------------------------------
def run_baseline(items, split_name):
    """Generate baseline responses, extract hidden states."""
    import mlx.core as mx
    from mlx_lm import load

    print(f"\n--- Baseline: {split_name} ({len(items)} items) ---")
    model, tokenizer = load(MODEL_PATH)

    # Determine layer indices
    # Qwen uses model.model.layers
    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers
    n_layers = len(layers)
    layer_indices = {
        "first": 0,
        "middle": n_layers // 2,
        "last": n_layers - 1,
    }
    print(f"  Model layers: {n_layers}, indices: {layer_indices}")

    from gen_helpers import generate_greedy
    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        user_msg = ARC_BASELINE_PROMPT.format(
            question=item["question"],
            choices=item["choices_text"],
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        raw = generate_greedy(model, tokenizer, prompt)
        answer = parse_answer_letter(raw)
        conf = parse_confidence(raw)
        correct = is_correct_arc(raw, item["gold_answer"])

        # Extract hidden states at target layer
        try:
            hs = extract_hs_qwen(model, tokenizer, prompt, raw, layer_indices)
        except Exception as e:
            hs = None

        results.append({
            "id": item["id"],
            "question": item["question"],
            "choices_text": item["choices_text"],
            "gold": item["gold_answer"],
            "raw_output": raw,
            "parsed_answer": answer,
            "confidence": conf,
            "correct": correct,
            "hidden_states": hs,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in results])
            print(f"  [{i+1}/{len(items)}]  acc={acc:.3f}  elapsed={elapsed:.0f}s")

    acc = np.mean([r["correct"] for r in results])
    print(f"  Final accuracy: {acc:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")

    # Save (strip hidden states from JSON, save separately as npz)
    results_for_json = [{k: v for k, v in r.items() if k != "hidden_states"} for r in results]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = f"{RESULTS_DIR}/baseline_{split_name}.json"
    with open(json_path, "w") as f:
        json.dump(results_for_json, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save hidden states
    hs_dict = {}
    for r in results:
        if r["hidden_states"] is not None:
            hs_dict[r["id"]] = r["hidden_states"]
    npz_path = f"{RESULTS_DIR}/hidden_states_{split_name}.npz"
    np.savez_compressed(npz_path, **{
        f"{qid}_{key}": arr for qid, hs in hs_dict.items() for key, arr in hs.items()
    })
    print(f"  Saved: {npz_path} ({len(hs_dict)} items)")

    return results


def extract_hs_qwen(model, tokenizer, prompt, response_text, layer_indices):
    """Extract hidden states at specified layers for Qwen."""
    import mlx.core as mx

    full_text = prompt + response_text
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    if len(tokens) <= prompt_len:
        return None

    x = mx.array([tokens])

    # Qwen: model.model.embed_tokens, model.model.layers
    try:
        embed = model.model.embed_tokens
        layers = model.model.layers
    except AttributeError:
        embed = model.embed_tokens
        layers = model.layers

    h = embed(x)

    hidden_states = {}
    for i, layer in enumerate(layers):
        h = layer(h, cache=None)
        for label, idx in layer_indices.items():
            if i == idx:
                last_pos = len(tokens) - 1
                hidden_states[f"{label}_{PROBE_POS}"] = np.array(
                    h[0, last_pos].astype(mx.float32)
                )
    mx.eval(h)
    return hidden_states

# ---------------------------------------------------------------------------
# Stage 2: Probe training + target derivation
# ---------------------------------------------------------------------------
def run_probe_and_targets(cal_results, eval_results):
    """Train probe on cal, derive targets, eval on eval."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    print("\n--- Probe Training ---")
    key = f"{PROBE_LAYER}_{PROBE_POS}"

    # Build training data from cal
    X_train, y_train = [], []
    cal_with_hs = []
    for r in cal_results:
        if r.get("hidden_states") and key in r["hidden_states"]:
            X_train.append(r["hidden_states"][key])
            y_train.append(int(r["correct"]))
            cal_with_hs.append(r)

    # Build eval data
    X_eval, y_eval, conf_eval = [], [], []
    for r in eval_results:
        if r.get("hidden_states") and key in r["hidden_states"]:
            X_eval.append(r["hidden_states"][key])
            y_eval.append(int(r["correct"]))
            conf_eval.append(r.get("confidence", float("nan")))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)

    print(f"  Train: {len(X_train)} items, acc={np.mean(y_train):.3f}")
    print(f"  Eval:  {len(X_eval)} items, acc={np.mean(y_eval):.3f}")

    # Fit probe
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_eval_s = scaler.transform(X_eval)

    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=5, penalty="l2", max_iter=2000,
        scoring="roc_auc", random_state=SEED,
    )
    clf.fit(X_train_s, y_train)

    # Eval probe on eval set
    probe_scores_eval = clf.predict_proba(X_eval_s)[:, 1]
    probe_auc = roc_auc_score(y_eval, probe_scores_eval)
    print(f"  Probe AUROC₂ on eval: {probe_auc:.3f}")

    # Verbal AUROC₂ on eval
    conf_arr = np.array(conf_eval)
    mask = ~np.isnan(conf_arr)
    if mask.sum() > 10 and y_eval[mask].sum() > 0:
        verbal_auc = roc_auc_score(np.array(y_eval)[mask], conf_arr[mask])
        print(f"  Verbal AUROC₂ on eval: {verbal_auc:.3f}")
        print(f"  Gap: {probe_auc - verbal_auc:+.3f}")

    # Derive targets on cal
    probe_scores_cal = clf.predict_proba(scaler.transform(X_train))[:, 1]
    targets = np.round(probe_scores_cal * 100).astype(int).clip(0, 100)

    print(f"  Target distribution: mean={np.mean(targets):.1f}, "
          f"std={np.std(targets):.1f}, "
          f"min={np.min(targets)}, max={np.max(targets)}")

    return cal_with_hs, targets, {"probe_auroc2": probe_auc, "best_C": float(clf.C_[0])}

# ---------------------------------------------------------------------------
# Stage 3: Prepare confonly training JSONL
# ---------------------------------------------------------------------------
def prep_confonly(cal_results, targets):
    """Prepare confonly training JSONL for mlx_lm LoRA."""
    print("\n--- Preparing Confonly Training Data ---")

    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Split cal into train (90%) and valid (10%)
    n = len(cal_results)
    n_valid = max(20, n // 10)
    n_train = n - n_valid

    train_items = []
    valid_items = []

    for i, (r, target) in enumerate(zip(cal_results, targets)):
        user_msg = ARC_CONFONLY_USER.format(
            question=r["question"],
            choices=r["choices_text"],
            answer=r.get("parsed_answer", ""),
        )
        assistant_msg = ARC_CONFONLY_ASSISTANT.format(confidence=int(target))

        item = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

        if i < n_train:
            train_items.append(item)
        else:
            valid_items.append(item)

    with open(TRAIN_JSONL, "w") as f:
        for item in train_items:
            f.write(json.dumps(item) + "\n")

    with open(VALID_JSONL, "w") as f:
        for item in valid_items:
            f.write(json.dumps(item) + "\n")

    print(f"  Train: {len(train_items)} items → {TRAIN_JSONL}")
    print(f"  Valid: {len(valid_items)} items → {VALID_JSONL}")

    # Print training command
    iters = len(train_items) * LORA_CONFIG["epochs"]
    print(f"\n  --- Training Command ---")
    print(f"  mlx_lm.lora \\")
    print(f"    --model {MODEL_PATH} \\")
    print(f"    --train \\")
    print(f"    --data {ADAPTER_DIR} \\")
    print(f"    --adapter-path {ADAPTER_DIR}/adapters \\")
    print(f"    --lora-rank {LORA_CONFIG['rank']} \\")
    print(f"    --lora-scale {LORA_CONFIG['scale']} \\")
    print(f"    --learning-rate {LORA_CONFIG['lr']} \\")
    print(f"    --batch-size {LORA_CONFIG['batch_size']} \\")
    print(f"    --grad-accumulate {LORA_CONFIG['grad_accum']} \\")
    print(f"    --iters {iters} \\")
    print(f"    --val-batches 5 \\")
    print(f"    --steps-per-eval 50")

    return train_items, valid_items

# ---------------------------------------------------------------------------
# Stage 4: Eval (two-pass confonly)
# ---------------------------------------------------------------------------
def run_eval(eval_items):
    """Two-pass evaluation: generate without adapter, rate with adapter."""
    import mlx.core as mx
    from mlx_lm import load

    adapter_path = f"{ADAPTER_DIR}/adapters"
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at {adapter_path}. Run training first.")
        return

    print(f"\n--- Eval: Two-Pass Confonly ---")

    # Pass 1: generate answers without adapter (use baseline results)
    baseline_path = f"{RESULTS_DIR}/baseline_eval.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_results = json.load(f)
        print(f"  Loaded baseline eval: {len(baseline_results)} items")
    else:
        print(f"  ERROR: baseline eval not found at {baseline_path}")
        print(f"  Run --stage baseline first")
        return

    # Pass 2: rate with adapter
    print(f"  Loading model + adapter...")
    model, tokenizer = load(MODEL_PATH, adapter_path=adapter_path)
    from gen_helpers import generate_greedy

    results = []
    t0 = time.time()

    for i, r in enumerate(baseline_results):
        user_msg = ARC_CONFONLY_USER.format(
            question=r["question"],
            choices=r["choices_text"],
            answer=r.get("parsed_answer", ""),
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        raw = generate_greedy(model, tokenizer, prompt)
        conf = parse_confidence(raw)

        results.append({
            "id": r["id"],
            "gold": r["gold"],
            "parsed_answer": r.get("parsed_answer", ""),
            "correct": r["correct"],
            "baseline_confidence": r.get("confidence", float("nan")),
            "ptcsft_confidence": conf,
            "ptcsft_raw": raw,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            confs = [x["ptcsft_confidence"] for x in results if not np.isnan(x["ptcsft_confidence"])]
            print(f"  [{i+1}/{len(baseline_results)}]  "
                  f"conf_mean={np.mean(confs):.1f}  elapsed={elapsed:.0f}s")

    # Compute metrics
    from sklearn.metrics import roc_auc_score

    conf_arr = np.array([r["ptcsft_confidence"] for r in results])
    correct_arr = np.array([int(r["correct"]) for r in results])
    mask = ~np.isnan(conf_arr)

    print(f"\n  --- Results ---")
    print(f"  N items: {len(results)}")
    print(f"  Accuracy: {np.mean(correct_arr):.3f}")
    print(f"  Confidence parsed: {mask.sum()}/{len(results)}")

    if mask.sum() > 10:
        auc = roc_auc_score(correct_arr[mask], conf_arr[mask])
        print(f"  PT-CSFT AUROC₂: {auc:.3f}")
        print(f"  Conf mean: {np.mean(conf_arr[mask]):.1f}")
        print(f"  Conf std:  {np.std(conf_arr[mask]):.1f}")

        # Baseline verbal
        bl_conf = np.array([r["baseline_confidence"] for r in results])
        bl_mask = ~np.isnan(bl_conf)
        if bl_mask.sum() > 10:
            bl_auc = roc_auc_score(correct_arr[bl_mask], bl_conf[bl_mask])
            print(f"  Baseline AUROC₂: {bl_auc:.3f}")
            print(f"  Δ: {auc - bl_auc:+.3f}")

        # VRS screening
        ceiling = np.mean(conf_arr[mask] >= 95)
        print(f"  L (ceiling rate): {ceiling:.3f}")

    # Save
    eval_path = f"{RESULTS_DIR}/confonly_eval.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {eval_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "baseline", "probe", "prep", "train", "eval"])
    args = parser.parse_args()

    items = load_arc()
    cal_items = items[:N_CAL]
    eval_items = items[N_CAL:N_CAL + N_EVAL]
    print(f"ARC-Challenge: {len(items)} total, {len(cal_items)} cal, {len(eval_items)} eval")

    if args.stage in ("all", "baseline"):
        cal_results = run_baseline(cal_items, "cal")
        eval_results = run_baseline(eval_items, "eval")
    else:
        # Load saved results
        cal_path = f"{RESULTS_DIR}/baseline_cal.json"
        eval_path = f"{RESULTS_DIR}/baseline_eval.json"
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                cal_results = json.load(f)
            with open(eval_path) as f:
                eval_results = json.load(f)
            print(f"Loaded saved baselines: {len(cal_results)} cal, {len(eval_results)} eval")

            # Load hidden states
            cal_npz = f"{RESULTS_DIR}/hidden_states_cal.npz"
            if os.path.exists(cal_npz):
                hs_data = np.load(cal_npz)
                key = f"{PROBE_LAYER}_{PROBE_POS}"
                for r in cal_results:
                    hs_key = f"{r['id']}_{key}"
                    if hs_key in hs_data:
                        r["hidden_states"] = {key: hs_data[hs_key]}
                    else:
                        r["hidden_states"] = None
                eval_npz = f"{RESULTS_DIR}/hidden_states_eval.npz"
                if os.path.exists(eval_npz):
                    hs_data_eval = np.load(eval_npz)
                    for r in eval_results:
                        hs_key = f"{r['id']}_{key}"
                        if hs_key in hs_data_eval:
                            r["hidden_states"] = {key: hs_data_eval[hs_key]}
                        else:
                            r["hidden_states"] = None
        else:
            print(f"ERROR: no saved baselines. Run --stage baseline first.")
            sys.exit(1)

    if args.stage in ("all", "probe", "prep"):
        cal_with_hs, targets, probe_info = run_probe_and_targets(cal_results, eval_results)
        train_items, valid_items = prep_confonly(cal_with_hs, targets)

        # Save probe info
        probe_path = f"{RESULTS_DIR}/probe_info.json"
        with open(probe_path, "w") as f:
            json.dump(probe_info, f, indent=2)

    if args.stage == "train":
        print("\n--- Training ---")
        print("Run the mlx_lm.lora command printed by --stage prep")
        print("Or use the shell command below:")
        n_train = N_CAL - max(20, N_CAL // 10)
        iters = n_train * LORA_CONFIG["epochs"]
        cmd = (
            f"mlx_lm.lora "
            f"--model {MODEL_PATH} "
            f"--train "
            f"--data {ADAPTER_DIR} "
            f"--adapter-path {ADAPTER_DIR}/adapters "
            f"--lora-rank {LORA_CONFIG['rank']} "
            f"--lora-scale {LORA_CONFIG['scale']} "
            f"--learning-rate {LORA_CONFIG['lr']} "
            f"--batch-size {LORA_CONFIG['batch_size']} "
            f"--grad-accumulate {LORA_CONFIG['grad_accum']} "
            f"--iters {iters} "
            f"--val-batches 5 "
            f"--steps-per-eval 50"
        )
        print(f"\n  {cmd}")

    if args.stage in ("all", "eval"):
        run_eval(eval_items)


if __name__ == "__main__":
    main()
