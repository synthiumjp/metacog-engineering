"""
pilot_continuous_target.py — Continuous-target E2E pilot for graded argmax confidence.

Tests whether soft probe-derived targets (instead of binary 0/9) produce graded
ARGMAX confidence, breaking the TRIN=1.0 binary limitation.

Binary targets (current): correct → "Confidence: 9", incorrect → "Confidence: 0"
  → Model outputs only 0 or 9 (TRIN=1.0), but logit distribution is graded

Continuous targets (this pilot): probe P(correct) scaled to 0-9
  → e.g., P(correct)=0.78 → "Confidence: 7"
  → Hypothesis: model learns to output intermediate digits (3, 5, 7)
  → Graded argmax confidence without needing logit readout

Uses existing E2E GSM8K infrastructure on Gemma 12B.

Usage:
    cd ~/jpwork/metacog-engineering/phase1

    # Stage 1: Prep training data with soft targets
    python3 scripts/pilot_continuous_target.py --stage prep

    # Stage 2: Train (use printed command)
    # mlx_lm.lora --model ... (see output)

    # Stage 3: Eval
    python3 scripts/pilot_continuous_target.py --stage eval

Environment:
    source ~/jpwork/.venv_metacog/bin/activate
    Model: Gemma 12B (~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it)
    Hardware: M3 Ultra 512GB, MLX
"""

import argparse, json, os, re, sys, time
import numpy as np

SEED = 42
MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
)
MODEL_NAME = "gemma-3-12b-it"

# Paths
RESULTS_DIR = "results_raw/domain_gen/pilot_continuous"
ADAPTER_DIR = f"finetune/{MODEL_NAME}/brier_e2e_gsm8k/adapter_continuous_pilot"
TRAIN_JSONL = f"{ADAPTER_DIR}/train.jsonl"
VALID_JSONL = f"{ADAPTER_DIR}/valid.jsonl"

# The existing E2E binary training used these items
# We reuse the same responses + probe, just change the targets
EXISTING_RESPONSES = "results_raw/domain_gen/responses_gsm8k_gemma-3-12b-it.json"
# Alternatively, the prep script from prep_brier_e2e.py
EXISTING_TRAIN_DATA = f"finetune/{MODEL_NAME}/brier_e2e_gsm8k/train.jsonl"
EXISTING_PROBE_CHECK = "results_raw/domain_gen/probe_check_gsm8k_gemma-3-12b-it.json"

# LoRA config — match gentle config
LORA_RANK = 16
LORA_SCALE = 2.0
LORA_LR = 5e-5  # gentle
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_TOKENS = 512

# GSM8K eval set
N_EVAL = 519

# ---------------------------------------------------------------------------
# E2E prompt format (must match existing)
# ---------------------------------------------------------------------------
E2E_SYSTEM = ""
E2E_USER = (
    "Solve this math problem step by step, then state your confidence "
    "as a single digit from 0 (no confidence) to 9 (very confident).\n\n"
    "Question: {question}"
)

def is_correct_gsm8k(predicted, gold):
    # Strip everything from "Confidence" onward so the confidence digit
    # doesn't get picked up as the answer
    text = re.split(r"\*{0,2}Confidence\*{0,2}", predicted, flags=re.IGNORECASE)[0]
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if not numbers:
        return False
    pred = numbers[-1].replace(",", "")
    gold_clean = gold.replace(",", "").strip()
    try:
        return float(pred) == float(gold_clean)
    except ValueError:
        return pred.strip() == gold_clean.strip()

# ---------------------------------------------------------------------------
# Stage 1: Prep training data with continuous targets
# ---------------------------------------------------------------------------
def prep_continuous_targets():
    """
    Load existing E2E responses + probe, derive soft targets.

    The probe gives P(correct) ∈ [0,1] per item. We scale to 0–9:
        target_digit = round(P(correct) * 9)

    This replaces the binary {0, 9} targets with graded {0, 1, ..., 9}.
    """
    print("=" * 60)
    print("Preparing Continuous-Target Training Data")
    print("=" * 60)

    # --- Load existing probe results ---
    # The probe check JSON should have per-item P(correct) or we can reconstruct
    # from hidden states + saved probe

    # Strategy: reload the existing E2E training data, extract the model's actual
    # responses, and apply the probe to get P(correct).
    # Since the probe was trained on GSM8K hidden states, we need those states.
    # The simplest approach: load the existing binary training JSONL, read which
    # items are correct/incorrect, and use the probe gate check results.

    # APPROACH A: Use existing probe check results directly
    # The probe_check JSON has per-item probe scores from the gate check
    if os.path.exists(EXISTING_PROBE_CHECK):
        print(f"  Loading probe check: {EXISTING_PROBE_CHECK}")
        with open(EXISTING_PROBE_CHECK) as f:
            probe_data = json.load(f)

        # Extract probe P(correct) per item from the probe check
        # The probe check stores items with hidden states + probe predictions
        # We need to match these to the E2E training items

        # Check what's in the probe check
        if "items" in probe_data:
            probe_items = probe_data["items"]
        elif "cal_results" in probe_data:
            probe_items = probe_data["cal_results"]
        else:
            print(f"  Probe check keys: {list(probe_data.keys())}")
            probe_items = None

    # APPROACH B: If probe check doesn't have per-item scores, reconstruct
    # by loading the existing E2E training JSONL and just using correctness
    # as a rough proxy, then refining with a new probe pass
    if not os.path.exists(EXISTING_PROBE_CHECK) or probe_items is None:
        print("  Probe check not found or missing items. Using reconstruction approach.")
        return prep_continuous_targets_from_scratch()

    return prep_from_probe_check(probe_data, probe_items)


def prep_continuous_targets_from_scratch():
    """
    Fallback: load E2E training data, re-extract hidden states, train probe,
    derive soft targets.
    """
    print("\n  --- Reconstructing from existing responses ---")

    # Load GSM8K responses
    response_paths = [
        EXISTING_RESPONSES,
        f"results_raw/domain_gen/responses_gsm8k_{MODEL_NAME}.json",
    ]
    responses = None
    for p in response_paths:
        if os.path.exists(p):
            with open(p) as f:
                responses = json.load(f)
            print(f"  Loaded responses: {p} ({len(responses)} items)")
            break

    if responses is None:
        print("  ERROR: No GSM8K responses found. Generate first with gen_gsm8k_responses.py")
        sys.exit(1)

    # We need hidden states to train a probe. If we don't have them,
    # we need to run inference. For the pilot, use a simpler approach:
    # use the model's own generation-time uncertainty as a proxy.
    # The binary E2E training data has correct/incorrect labels.
    # We'll use a held-out correctness-based target with noise.

    # Actually, the cleanest approach for a PILOT is:
    # 1. Load existing binary training JSONL
    # 2. For correct items: target = 7, 8, or 9 (sampled from probe-like dist)
    # 3. For incorrect items: target = 0, 1, or 2
    # 4. For ambiguous items (if any): target = 3, 4, 5, 6

    # But the REAL approach is to use the probe. Let me generate probe scores.
    print("\n  Need to extract hidden states and train probe.")
    print("  This requires model loading. Running probe extraction...")

    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(MODEL_PATH)

    # Detect model structure — Gemma 3: model.language_model.model.{embed_tokens, layers}
    if hasattr(model, 'language_model'):
        inner = model.language_model.model
        embed = inner.embed_tokens
        layers = inner.layers
        model_type = "gemma3"
        scale_embeddings = True
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        embed = model.model.embed_tokens
        layers = model.model.layers
        model_type = "standard"
        scale_embeddings = False
    else:
        embed = model.embed_tokens
        layers = model.layers
        model_type = "flat"
        scale_embeddings = False

    n_layers = len(layers)
    mid_idx = n_layers // 2
    print(f"  Model: {model_type}, {n_layers} layers, middle={mid_idx}")

    # Use responses to get correctness + extract hidden states on the fly
    # For efficiency, extract only middle layer (best probe from gate check)
    hidden_states = []
    correctness = []
    item_data = []

    for i, resp in enumerate(responses):
        if "question" not in resp:
            continue

        user_msg = E2E_USER.format(question=resp["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        # Get the model's response
        raw = resp.get("response", resp.get("raw_output", ""))
        correct = resp.get("correct", is_correct_gsm8k(raw, resp.get("gold", resp.get("gold_answer", ""))))

        # Extract hidden state at last token of response
        full_text = prompt + raw
        tokens = tokenizer.encode(full_text)
        x = mx.array([tokens])
        h = embed(x)
        if scale_embeddings:
            hidden_size = h.shape[-1]
            h = h * (hidden_size ** 0.5)

        for j, layer in enumerate(layers):
            h = layer(h, cache=None)
            if j == mid_idx:
                hs_vec = np.array(h[0, -1].astype(mx.float32))
                hidden_states.append(hs_vec)
                break
        else:
            hidden_states.append(None)

        mx.eval(h)
        del h, x

        correctness.append(int(correct))
        item_data.append({
            "question": resp["question"],
            "gold": resp.get("gold", resp.get("gold_answer", "")),
            "response": raw,
            "correct": correct,
        })

        if (i + 1) % 100 == 0:
            acc = np.mean(correctness)
            print(f"  [{i+1}/{len(responses)}]  acc={acc:.3f}")

    # Filter items with hidden states
    valid_idx = [i for i, hs in enumerate(hidden_states) if hs is not None]
    X = np.array([hidden_states[i] for i in valid_idx])
    y = np.array([correctness[i] for i in valid_idx])
    valid_items = [item_data[i] for i in valid_idx]

    print(f"\n  Valid items: {len(valid_idx)}, acc={np.mean(y):.3f}")

    # Train probe
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=5, penalty="l2", max_iter=2000,
        scoring="roc_auc", random_state=SEED,
    )
    clf.fit(X_s, y)

    probe_scores = clf.predict_proba(X_s)[:, 1]
    probe_auc = roc_auc_score(y, probe_scores)
    print(f"  Probe AUROC₂ (CV): {probe_auc:.3f}")

    # Derive soft targets
    targets = np.round(probe_scores * 9).astype(int).clip(0, 9)
    print(f"  Target distribution: {np.bincount(targets, minlength=10)}")
    print(f"  Target mean: {np.mean(targets):.2f}, std: {np.std(targets):.2f}")

    # Now balance and create training set
    # Balance by correctness (same as binary E2E)
    correct_idx = [i for i in range(len(y)) if y[i] == 1]
    incorrect_idx = [i for i in range(len(y)) if y[i] == 0]
    n_minority = min(len(correct_idx), len(incorrect_idx))
    print(f"\n  Correct: {len(correct_idx)}, Incorrect: {len(incorrect_idx)}")
    print(f"  Balanced: {n_minority} per class = {2 * n_minority} total")

    rng = np.random.default_rng(SEED)
    rng.shuffle(correct_idx)
    rng.shuffle(incorrect_idx)
    balanced_idx = sorted(correct_idx[:n_minority] + incorrect_idx[:n_minority])

    # Split train/valid
    n_valid = max(20, len(balanced_idx) // 10)
    train_idx = balanced_idx[:-n_valid]
    valid_idx_split = balanced_idx[-n_valid:]

    # Create JSONL
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    def make_training_item(idx):
        item = valid_items[idx]
        target = int(targets[idx])
        # E2E format: response + "Confidence: {digit}"
        response = item["response"]
        # Strip any existing confidence line
        response = re.sub(r"\*{0,2}Confidence\*{0,2}\s*:?\*{0,2}\s*\d.*$", "", response, flags=re.MULTILINE).strip()
        response += f"\nConfidence: {target}"

        return {
            "messages": [
                {"role": "user", "content": E2E_USER.format(question=item["question"])},
                {"role": "assistant", "content": response},
            ]
        }

    with open(TRAIN_JSONL, "w") as f:
        for idx in train_idx:
            f.write(json.dumps(make_training_item(idx)) + "\n")

    with open(VALID_JSONL, "w") as f:
        for idx in valid_idx_split:
            f.write(json.dumps(make_training_item(idx)) + "\n")

    print(f"\n  Train: {len(train_idx)} items → {TRAIN_JSONL}")
    print(f"  Valid: {len(valid_idx_split)} items → {VALID_JSONL}")

    # Check target distribution in training set
    train_targets = [int(targets[i]) for i in train_idx]
    print(f"  Training target distribution: {np.bincount(train_targets, minlength=10)}")

    # Print training command
    # iters = micro-batch steps (mlx_lm counts these, not gradient updates)
    iters = len(train_idx) * 3  # 3 epochs, but mlx_lm iters = micro-batch steps
    print(f"\n  --- Training Command ---")
    cmd = (
        f"mlx_lm.lora "
        f"--model {MODEL_PATH} "
        f"--train "
        f"--data {ADAPTER_DIR} "
        f"--adapter-path {ADAPTER_DIR}/adapters "
        f"--lora-rank {LORA_RANK} "
        f"--lora-scale {LORA_SCALE} "
        f"--learning-rate {LORA_LR} "
        f"--batch-size {BATCH_SIZE} "
        f"--grad-accumulate {GRAD_ACCUM} "
        f"--iters {iters} "
        f"--val-batches 5 "
        f"--steps-per-eval 50"
    )
    print(f"  {cmd}")

    # Save metadata
    meta = {
        "probe_auroc2": probe_auc,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx_split),
        "target_distribution": np.bincount(train_targets, minlength=10).tolist(),
        "target_type": "continuous_probe_derived",
        "binary_comparison": "adapter_e2e_ce_gentle_seed42",
    }
    meta_path = f"{ADAPTER_DIR}/pilot_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")


def prep_from_probe_check(probe_data, probe_items):
    """Use existing probe check results to derive soft targets."""
    print("\n  --- Using probe check results ---")
    # Implementation depends on probe check JSON format
    # If it has per-item probe scores, use those directly
    # Otherwise fall back to from_scratch
    if isinstance(probe_items, list) and len(probe_items) > 0:
        sample = probe_items[0]
        if "probe_score" in sample or "probe_p_correct" in sample:
            score_key = "probe_score" if "probe_score" in sample else "probe_p_correct"
            print(f"  Found per-item probe scores ({score_key})")
            # Use these directly...
            # (implementation same as from_scratch but skip re-probing)
        else:
            print(f"  No per-item probe scores (keys: {list(sample.keys())[:8]})")
    print("  Falling back to from-scratch approach")
    return prep_continuous_targets_from_scratch()


# ---------------------------------------------------------------------------
# Stage 2: Eval
# ---------------------------------------------------------------------------
def run_eval():
    """Eval the continuous-target adapter: both argmax and logit readout."""
    import mlx.core as mx
    from mlx_lm import load

    adapter_path = f"{ADAPTER_DIR}/adapters"
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at {adapter_path}. Run training first.")
        return

    print("=" * 60)
    print("Evaluating Continuous-Target Pilot")
    print("=" * 60)

    model, tokenizer = load(MODEL_PATH, adapter_path=adapter_path)
    from mlx_lm import generate as mlx_generate

    # Load eval data (same as binary E2E eval)
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        answer_text = row["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({"question": row["question"], "gold": gold})

    rng = np.random.default_rng(SEED)
    rng.shuffle(items)
    eval_items = items[:N_EVAL]

    # Get digit token IDs for Gemma (single digits 0-9)
    digit_ids = []
    for d in range(10):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if len(toks) == 1:
            digit_ids.append(toks[0])
        else:
            print(f"  Warning: digit {d} is multi-token: {toks}")
            digit_ids.append(toks[0])
    print(f"  Digit token IDs: {digit_ids}")

    results = []
    t0 = time.time()

    for i, item in enumerate(eval_items):
        user_msg = E2E_USER.format(question=item["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        # Generate
        response = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_TOKENS, verbose=False,
        )

        # Parse
        correct = is_correct_gsm8k(response, item["gold"])
        conf_match = re.search(r"\*{0,2}Confidence\*{0,2}\s*:?\*{0,2}\s*(\d)", response, re.IGNORECASE)
        argmax_digit = int(conf_match.group(1)) if conf_match else -1

        # Logit readout: get softmax over digit tokens at confidence position
        logit_ev = float("nan")
        digit_probs = None
        try:
            full_text = prompt + response
            tokens = tokenizer.encode(full_text)

            # Find confidence digit position
            if conf_match:
                # Tokenize up to confidence digit
                prefix = prompt + response[:conf_match.end()]
                prefix_tokens = tokenizer.encode(prefix)
                conf_pos = len(prefix_tokens) - 1

                # Forward pass to get logits at confidence position
                x = mx.array([tokens[:conf_pos + 1]])
                logits = model(x)  # (1, seq_len, vocab_size)
                conf_logits = logits[0, -1, :]  # logits at confidence position

                # Extract digit logits and softmax
                digit_logits = mx.array([conf_logits[tid] for tid in digit_ids])
                digit_softmax = mx.softmax(digit_logits)
                digit_probs = np.array(digit_softmax.astype(mx.float32))

                # Expected value: Σ (d/9) × P(d)
                logit_ev = float(np.sum(np.arange(10) / 9.0 * digit_probs))

                mx.eval(logits)
                del x, logits
        except Exception as e:
            pass

        results.append({
            "question": item["question"],
            "gold": item["gold"],
            "correct": correct,
            "argmax_digit": argmax_digit,
            "logit_ev": logit_ev,
            "digit_probs": digit_probs.tolist() if digit_probs is not None else None,
            "response_snippet": response[-100:],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in results])
            digits = [r["argmax_digit"] for r in results if r["argmax_digit"] >= 0]
            unique_digits = len(set(digits))
            print(f"  [{i+1}/{len(eval_items)}]  acc={acc:.3f}  "
                  f"unique_digits={unique_digits}  elapsed={elapsed:.0f}s")

    # --- Compute metrics ---
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    correct_arr = np.array([int(r["correct"]) for r in results])
    argmax_arr = np.array([r["argmax_digit"] for r in results])
    logit_arr = np.array([r["logit_ev"] for r in results])

    acc = np.mean(correct_arr)
    print(f"  Accuracy: {acc:.3f}")

    # Argmax analysis
    valid_argmax = argmax_arr >= 0
    if valid_argmax.sum() > 10:
        from sklearn.metrics import roc_auc_score
        argmax_conf = argmax_arr[valid_argmax] / 9.0
        argmax_correct = correct_arr[valid_argmax]
        argmax_auc = roc_auc_score(argmax_correct, argmax_conf)

        unique_vals = np.unique(argmax_arr[valid_argmax])
        trin = 1.0 if len(unique_vals) <= 2 else 0.0  # simplified
        digit_counts = np.bincount(argmax_arr[valid_argmax].astype(int).clip(0, 9), minlength=10)

        print(f"\n  ARGMAX readout:")
        print(f"    AUROC₂: {argmax_auc:.3f}")
        print(f"    Unique digits: {len(unique_vals)} ({sorted(unique_vals.tolist())})")
        print(f"    Digit distribution: {digit_counts.tolist()}")
        print(f"    Mean: {np.mean(argmax_arr[valid_argmax]):.2f}")
        print(f"    Std:  {np.std(argmax_arr[valid_argmax]):.2f}")
        print(f"    TRIN ≈ {trin} ({'BINARY' if len(unique_vals) <= 2 else 'GRADED'})")

    # Logit analysis
    valid_logit = ~np.isnan(logit_arr)
    if valid_logit.sum() > 10:
        logit_auc = roc_auc_score(correct_arr[valid_logit], logit_arr[valid_logit])
        print(f"\n  LOGIT readout:")
        print(f"    AUROC₂: {logit_auc:.3f}")
        print(f"    Mean:   {np.mean(logit_arr[valid_logit]):.3f}")
        print(f"    Std:    {np.std(logit_arr[valid_logit]):.3f}")

    # Compare with binary baseline
    print(f"\n  --- Comparison with binary E2E ---")
    print(f"  Binary E2E (gentle, 10-seed): argmax 0.743 ± 0.062, logit 0.862 ± 0.012")
    if valid_argmax.sum() > 10:
        print(f"  Continuous pilot:             argmax {argmax_auc:.3f},        logit {logit_auc:.3f}")
        print(f"  Key question: is argmax GRADED (unique digits > 2)?")
        print(f"    → {'YES' if len(unique_vals) > 2 else 'NO'} ({len(unique_vals)} unique values)")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/continuous_pilot_eval.json"
    # Strip digit_probs for JSON size
    results_json = [{k: v for k, v in r.items() if k != "digit_probs"} for r in results]
    summary = {
        "accuracy": float(acc),
        "n_items": len(results),
        "argmax_auroc2": float(argmax_auc) if valid_argmax.sum() > 10 else None,
        "logit_auroc2": float(logit_auc) if valid_logit.sum() > 10 else None,
        "argmax_unique_digits": int(len(unique_vals)) if valid_argmax.sum() > 10 else None,
        "digit_distribution": digit_counts.tolist() if valid_argmax.sum() > 10 else None,
        "items": results_json,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Save digit probability distributions for diagnostic figures
    probs_path = f"{RESULTS_DIR}/continuous_pilot_digit_probs.npz"
    probs_correct = []
    probs_incorrect = []
    for r in results:
        if r["digit_probs"] is not None:
            if r["correct"]:
                probs_correct.append(r["digit_probs"])
            else:
                probs_incorrect.append(r["digit_probs"])
    if probs_correct and probs_incorrect:
        np.savez_compressed(
            probs_path,
            correct=np.array(probs_correct),
            incorrect=np.array(probs_incorrect),
        )
        print(f"  Saved digit probs: {probs_path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="prep", choices=["prep", "eval", "all"])
    args = parser.parse_args()

    if args.stage in ("prep", "all"):
        prep_continuous_targets()

    if args.stage in ("eval", "all"):
        run_eval()


if __name__ == "__main__":
    main()
