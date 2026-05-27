"""
gsm8k_e2e_qwen7b.py — GSM8K End-to-End PT-CSFT on Qwen 7B

Replicates the Gemma 12B headline result on a second model.
Multi-stage pipeline:

    python3 scripts/gsm8k_e2e_qwen7b.py --stage baseline    # Generate GSM8K responses (~30 min)
    python3 scripts/gsm8k_e2e_qwen7b.py --stage prep         # Score + create training JSONL (~1 min)
    python3 scripts/gsm8k_e2e_qwen7b.py --stage train        # LoRA fine-tune (~10 min)
    python3 scripts/gsm8k_e2e_qwen7b.py --stage eval         # Generate eval responses + logit readout (~30 min)
    python3 scripts/gsm8k_e2e_qwen7b.py --stage all          # Run everything
"""

import argparse, json, os, re, sys, time, subprocess
import numpy as np
from pathlib import Path

MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/Qwen2.5-7B-Instruct-bf16"
)
OUTPUT_DIR = Path("results_raw/domain_gen/gsm8k_e2e_qwen7b")
SEED = 42
N_EVAL = 500  # held-out eval items
LR = "5e-5"
RANK = 16
ITERS = 350  # ~3 epochs on ~280 items with batch 4 (accounting for mlx_lm micro-steps)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "After your final answer, state your confidence as a single digit from 0 to 9.\n\n"
    "Question: {question}\n"
)


def extract_gsm8k_answer(text):
    """Extract final numeric answer from GSM8K response."""
    # Look for #### pattern first (GSM8K gold format)
    m = re.search(r'####\s*([\-\d,]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    # Look for "answer is X" pattern
    m = re.search(r'(?:answer|result)\s+(?:is|=)\s*([\-\d,]+)', text, re.I)
    if m:
        return m.group(1).replace(',', '').strip()
    # Last number in text (before confidence)
    # Strip everything after "Confidence"
    conf_pos = text.lower().find('confidence')
    if conf_pos > 0:
        text = text[:conf_pos]
    numbers = re.findall(r'[\-]?\d[\d,]*', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return ""


def is_correct_gsm8k(predicted_answer, gold_answer):
    """Check if predicted answer matches gold (numeric comparison)."""
    try:
        pred = float(predicted_answer.replace(',', ''))
        gold = float(gold_answer.replace(',', ''))
        return abs(pred - gold) < 0.01
    except (ValueError, AttributeError):
        return False


def parse_confidence(text):
    """Parse confidence digit (0-9) from response."""
    m = re.search(r'\*{0,2}[Cc]onfidence\*{0,2}\s*:?\s*(\d)\b', text)
    if m:
        return int(m.group(1))
    return -1


# ===================================================================
# Stage 1: Generate baseline GSM8K responses
# ===================================================================
def run_baseline():
    print("=" * 60)
    print("Stage 1: Baseline GSM8K Responses (Qwen 7B)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load GSM8K
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    print(f"  GSM8K test: {len(ds)} items")

    # Load model (no adapter)
    print(f"  Loading Qwen 7B...")
    from mlx_lm import load, generate
    model, tokenizer = load(MODEL_PATH)

    results = []
    t0 = time.time()

    for i, item in enumerate(ds):
        question = item['question']
        gold = item['answer'].split('####')[-1].strip() if '####' in item['answer'] else ""

        prompt_text = PROMPT_TEMPLATE.format(question=question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=512, verbose=False)

        pred_answer = extract_gsm8k_answer(response)
        correct = is_correct_gsm8k(pred_answer, gold)
        conf = parse_confidence(response)

        results.append({
            'id': f'gsm8k_{i}',
            'question': question,
            'gold': gold,
            'raw_output': response[:500],
            'predicted_answer': pred_answer,
            'correct': correct,
            'confidence': conf,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r['correct'] for r in results])
            print(f"  [{i+1}/{len(ds)}]  acc={acc:.3f}  elapsed={elapsed:.0f}s")

    acc = np.mean([r['correct'] for r in results])
    print(f"\n  Final accuracy: {acc:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")

    out = OUTPUT_DIR / "baseline_responses.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}")


# ===================================================================
# Stage 2: Score + create training JSONL
# ===================================================================
def run_prep():
    print("\n" + "=" * 60)
    print("Stage 2: Prepare Training Data")
    print("=" * 60)

    with open(OUTPUT_DIR / "baseline_responses.json") as f:
        results = json.load(f)

    acc = np.mean([r['correct'] for r in results])
    print(f"  Items: {len(results)}, accuracy: {acc:.3f}")

    # Split: first N_EVAL as eval, rest as cal/train
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(results))
    eval_idx = indices[:N_EVAL]
    cal_idx = indices[N_EVAL:]

    eval_items = [results[i] for i in eval_idx]
    cal_items = [results[i] for i in cal_idx]

    print(f"  Eval: {len(eval_items)} (acc={np.mean([i['correct'] for i in eval_items]):.3f})")
    print(f"  Cal/train: {len(cal_items)} (acc={np.mean([i['correct'] for i in cal_items]):.3f})")

    # Create E2E training JSONL
    # Binary labels: correct → "Confidence: 9", incorrect → "Confidence: 0"
    train_lines = []
    for item in cal_items:
        conf_label = 9 if item['correct'] else 0
        # Use the model's own response as the training target
        # Strip any existing confidence from the response
        response = item['raw_output']
        # Remove existing confidence line
        response = re.sub(r'\n*\*{0,2}[Cc]onfidence\*{0,2}\s*:?\s*\d.*$', '', response, flags=re.DOTALL)
        response = response.strip()
        # Append binary confidence
        response += f"\n\nConfidence: {conf_label}"

        train_lines.append({
            "messages": [
                {"role": "user", "content": PROMPT_TEMPLATE.format(question=item['question'])},
                {"role": "assistant", "content": response}
            ]
        })

    # Save training JSONL
    train_path = OUTPUT_DIR / "train.jsonl"
    with open(train_path, 'w') as f:
        for line in train_lines:
            f.write(json.dumps(line) + '\n')

    # Save eval set
    eval_path = OUTPUT_DIR / "eval_items.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_items, f, indent=2)

    # Save valid JSONL (small subset for mlx_lm validation)
    valid_path = OUTPUT_DIR / "valid.jsonl"
    with open(valid_path, 'w') as f:
        for line in train_lines[:50]:
            f.write(json.dumps(line) + '\n')

    # Save test JSONL (required by mlx_lm)
    test_path = OUTPUT_DIR / "test.jsonl"
    with open(test_path, 'w') as f:
        for line in train_lines[:20]:
            f.write(json.dumps(line) + '\n')

    print(f"  Training items: {len(train_lines)}")
    print(f"  Correct: {sum(1 for i in cal_items if i['correct'])}, "
          f"Incorrect: {sum(1 for i in cal_items if not i['correct'])}")
    print(f"  Saved: {train_path}")

    # Create LoRA config
    config = {
        "lora_layers": 16,
        "lora_parameters": {
            "rank": RANK,
            "scale": 2.0,
            "dropout": 0.0,
        }
    }
    config_path = OUTPUT_DIR / "lora_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"  Saved: {config_path}")


# ===================================================================
# Stage 3: Train LoRA adapter
# ===================================================================
def run_train():
    print("\n" + "=" * 60)
    print("Stage 3: LoRA Training")
    print("=" * 60)

    adapter_dir = OUTPUT_DIR / "adapters"
    os.makedirs(adapter_dir, exist_ok=True)

    cmd = [
        "python3", "-m", "mlx_lm.lora",
        "--model", MODEL_PATH,
        "--data", str(OUTPUT_DIR),
        "--train",
        "--iters", str(ITERS),
        "--batch-size", "4",
        "--learning-rate", LR,
        "--adapter-path", str(adapter_dir),
        "--seed", str(SEED),
        "-c", str(OUTPUT_DIR / "lora_config.yaml"),
        "--grad-accumulation-steps", "2",
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Training...")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=False)

    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.0f}s")
    if result.returncode != 0:
        print(f"  ERROR: Training failed with return code {result.returncode}")
        sys.exit(1)


# ===================================================================
# Stage 4: Evaluate with logit readout
# ===================================================================
def run_eval():
    print("\n" + "=" * 60)
    print("Stage 4: Evaluation + Logit Readout")
    print("=" * 60)

    adapter_dir = OUTPUT_DIR / "adapters"
    if not adapter_dir.exists():
        print(f"  ERROR: No adapter at {adapter_dir}")
        sys.exit(1)

    with open(OUTPUT_DIR / "eval_items.json") as f:
        eval_items = json.load(f)
    print(f"  Eval items: {len(eval_items)}")

    # Load model + adapter
    print(f"  Loading Qwen 7B + E2E adapter...")
    import mlx.core as mx
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_PATH, adapter_path=str(adapter_dir))

    # Detect single-token digits
    digit_ids = {}
    for d in range(10):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if len(toks) == 1:
            digit_ids[d] = toks[0]
    print(f"  Single-token digits (0-9): {len(digit_ids)}/10")

    if len(digit_ids) < 10:
        print(f"  WARNING: Not all digits are single-token")
        # Try to find them anyway
        for d in range(10):
            toks = tokenizer.encode(str(d), add_special_tokens=False)
            digit_ids[d] = toks[0]  # take first token

    max_digit = 9
    digits = np.array(sorted(digit_ids.keys()))

    results = []
    t0 = time.time()

    for i, item in enumerate(eval_items):
        question = item['question']
        gold = item['gold']

        prompt_text = PROMPT_TEMPLATE.format(question=question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        # Generate full response
        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=512, verbose=False)

        pred_answer = extract_gsm8k_answer(response)
        correct = is_correct_gsm8k(pred_answer, gold)
        text_conf = parse_confidence(response)

        # Logit readout at confidence position
        conf_match = re.search(r'[Cc]onfidence:?\s*', response)
        logit_ev = float("nan")
        digit_probs = None

        if conf_match:
            prefix = response[:conf_match.end()]
            full_text = prompt + prefix
            tokens = tokenizer.encode(full_text)

            try:
                x = mx.array([tokens])
                logits = model(x)
                last_logits = logits[0, -1, :]

                d_logits = mx.array([last_logits[digit_ids[d]] for d in digits])
                d_probs = mx.softmax(d_logits)
                d_probs_np = np.array(d_probs.astype(mx.float32))

                logit_ev = float(np.sum((digits / max_digit) * d_probs_np)) * 100
                digit_probs = d_probs_np.tolist()

                mx.eval(logits)
                del x, logits
            except Exception as e:
                if i < 3:
                    print(f"  Error: {e}")

        results.append({
            'id': item['id'],
            'correct': correct,
            'text_confidence': text_conf,
            'logit_confidence': logit_ev,
            'digit_probs': digit_probs,
            'predicted_answer': pred_answer,
            'gold': gold,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            valid = [r for r in results if not np.isnan(r['logit_confidence'])]
            c = np.array([int(r['correct']) for r in valid])
            lc = np.array([r['logit_confidence'] for r in valid])
            tc = np.array([r['text_confidence'] for r in valid if r['text_confidence'] >= 0])
            acc = np.mean([r['correct'] for r in results])
            if len(valid) > 10 and c.sum() > 0 and c.sum() < len(c):
                l_auc = roc_auc_score(c, lc)
                print(f"  [{i+1}/{len(eval_items)}]  acc={acc:.3f} logit={l_auc:.3f}  elapsed={elapsed:.0f}s")
            else:
                print(f"  [{i+1}/{len(eval_items)}]  acc={acc:.3f}  elapsed={elapsed:.0f}s")

    # --- Final Analysis ---
    from sklearn.metrics import roc_auc_score

    all_correct = np.array([int(r['correct']) for r in results])
    logit_valid = [r for r in results if not np.isnan(r['logit_confidence'])]
    text_valid = [r for r in results if r['text_confidence'] >= 0]

    print(f"\n  {'='*60}")
    print(f"  Qwen 7B GSM8K E2E Results")
    print(f"  {'='*60}")
    print(f"  Accuracy: {all_correct.mean():.3f}")

    if logit_valid:
        c = np.array([int(r['correct']) for r in logit_valid])
        lc = np.array([r['logit_confidence'] for r in logit_valid])
        if c.sum() > 0 and c.sum() < len(c):
            l_auc = roc_auc_score(c, lc)
            print(f"  Logit AUROC₂: {l_auc:.3f}")
            print(f"  Logit conf mean: {lc.mean():.1f} (std {lc.std():.1f})")

    if text_valid:
        c = np.array([int(r['correct']) for r in text_valid])
        tc = np.array([r['text_confidence'] for r in text_valid])
        if c.sum() > 0 and c.sum() < len(c):
            t_auc = roc_auc_score(c, tc)
            print(f"  Text AUROC₂:  {t_auc:.3f}")

    # Load baseline for comparison
    with open(OUTPUT_DIR / "baseline_responses.json") as f:
        bl = json.load(f)
    bl_eval_ids = {r['id'] for r in eval_items}
    bl_eval = [r for r in bl if r['id'] in bl_eval_ids]
    bl_acc = np.mean([r['correct'] for r in bl_eval])
    bl_confs = [r['confidence'] for r in bl_eval if r['confidence'] >= 0]
    bl_correct_v = [r['correct'] for r in bl_eval if r['confidence'] >= 0]
    if bl_confs and sum(bl_correct_v) > 0 and sum(bl_correct_v) < len(bl_correct_v):
        bl_auc = roc_auc_score(bl_correct_v, bl_confs)
    else:
        bl_auc = 0.5

    print(f"\n  Context:")
    print(f"    Baseline accuracy: {bl_acc:.3f}")
    print(f"    Baseline AUROC₂:  {bl_auc:.3f}")
    print(f"    Gemma 12B logit:  0.862 (headline)")
    if logit_valid:
        print(f"    Qwen 7B logit:    {l_auc:.3f}")
        if l_auc > 0.80:
            print(f"\n  ✓ GSM8K E2E REPLICATES on second model!")
        elif l_auc > 0.70:
            print(f"\n  ~ Partial replication (>0.70)")

    # Save
    out = OUTPUT_DIR / "e2e_eval_results.json"
    save = {
        "model": "Qwen2.5-7B",
        "task": "GSM8K",
        "method": "E2E gentle",
        "lr": LR,
        "rank": RANK,
        "n_eval": len(eval_items),
        "accuracy": float(all_correct.mean()),
        "baseline_accuracy": float(bl_acc),
        "baseline_auroc2": float(bl_auc),
        "logit_auroc2": float(l_auc) if logit_valid else None,
        "text_auroc2": float(t_auc) if text_valid else None,
    }
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved: {out}")

    # Save full results
    with open(OUTPUT_DIR / "e2e_eval_responses.json", 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["baseline", "prep", "train", "eval", "all"])
    args = parser.parse_args()

    if args.stage in ("baseline", "all"):
        run_baseline()
    if args.stage in ("prep", "all"):
        run_prep()
    if args.stage in ("train", "all"):
        run_train()
    if args.stage in ("eval", "all"):
        run_eval()


if __name__ == "__main__":
    main()
