"""
eval_70b_bonus.py — Two 70B evaluations:
  1. r128/gentle adapter: accuracy + AUROC₂ (resolve §6.3 number)
  2. gentlest r4/5e-5 adapter: logit readout (compare to balanced confonly logit 0.746)

Usage:
    python3 scripts/eval_70b_bonus.py --stage r128
    python3 scripts/eval_70b_bonus.py --stage gentlest
    python3 scripts/eval_70b_bonus.py --stage both
"""

import argparse, json, os, re, time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED"
)

ADAPTERS = {
    "r128_gentle": "results_raw/finetune/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED/ablation_r128_gentle/adapters",
    "gentlest": "results_raw/finetune/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED/ablation_gentlest/adapters",
}

BASELINE_PATH = Path("results_raw/step1/teval_responses_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED.json")
OUTPUT_DIR = Path("results_raw/domain_gen")

PROMPT_TEMPLATE = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

CONFONLY_USER = (
    "You answered the following trivia question.\n"
    "Question: {question}\n"
    "Your answer: {answer}\n"
    "How confident are you that your answer is correct? "
    "State your confidence as a percentage from 0 to 100."
)


def parse_confidence(text):
    pats = [
        re.compile(r"\*{0,2}[Cc]onfidence\*{0,2}\s*:?\s*(\d{1,3})\s*%?"),
        re.compile(r"(\d{1,3})\s*%"),
    ]
    for pat in pats:
        m = pat.search(text)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return float(v)
    return float("nan")


def is_correct_flex(predicted, gold):
    """Bidirectional substring, min 2 chars."""
    pred = predicted.strip().lower()
    pred = re.sub(r'[^\w\s]', '', pred)
    g = gold.strip().lower()
    g = re.sub(r'[^\w\s]', '', g)
    if len(pred) < 2 or len(g) < 2:
        return False
    return g in pred or pred in g


# ===================================================================
# Stage 1: r128/gentle — accuracy + AUROC₂
# ===================================================================
def run_r128(model=None, tokenizer=None):
    """Evaluate r128/gentle adapter: just accuracy and AUROC₂."""
    print("=" * 60)
    print("70B r128/gentle Evaluation")
    print("=" * 60)

    adapter_path = ADAPTERS["r128_gentle"]
    if not os.path.exists(adapter_path):
        print(f"  ERROR: Adapter not found at {adapter_path}")
        return

    # Load baseline to get questions
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    # Load gold answers from confonly responses (has 'gold' field)
    confonly_path = Path("results_raw/step4/balanced_confonly_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED_responses.json")
    gold_map = {}
    if confonly_path.exists():
        with open(confonly_path) as f:
            confonly = json.load(f)
        gold_map = {item['id']: item['gold'] for item in confonly}
    print(f"  Baseline items: {len(baseline)}, gold answers: {len(gold_map)}")

    # Load model with r128 adapter
    if model is None:
        print(f"  Loading model + r128/gentle adapter...")
        import mlx.core as mx
        from mlx_lm import load, generate
        model, tokenizer = load(MODEL_PATH, adapter_path=adapter_path)
    else:
        from mlx_lm import generate

    results = []
    t0 = time.time()

    for i, item in enumerate(baseline):
        question = item['question']
        gold = item.get('gold', '')
        if not gold:
            # Try to extract from parsed_answer
            gold = item.get('parsed_answer', '')

        prompt_text = PROMPT_TEMPLATE.format(question=question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=200, verbose=False)

        conf = parse_confidence(response)
        
        # Check correctness against gold
        qid = item.get('question_id', f'q_{i}')
        gold = gold_map.get(qid, '')
        if gold:
            gen_correct = is_correct_flex(response, gold)
        else:
            gen_correct = False  # can't verify without gold

        results.append({
            'question_id': qid,
            'response': response[:200],
            'confidence': conf,
            'correct': gen_correct,
            'baseline_correct': item.get('correct', False),
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r['correct'] for r in results])
            bl_acc = np.mean([r['baseline_correct'] for r in results])
            print(f"  [{i+1}/{len(baseline)}]  acc={acc:.3f} (baseline={bl_acc:.3f})  elapsed={elapsed:.0f}s")

    # Analysis
    correct = np.array([int(r['correct']) for r in results])
    bl_correct = np.array([int(r['baseline_correct']) for r in results])
    conf = np.array([r['confidence'] for r in results])
    valid = ~np.isnan(conf)

    print(f"\n  --- r128/gentle Results ---")
    print(f"  Items: {len(results)}")
    print(f"  Baseline accuracy: {bl_correct.mean():.3f}")
    print(f"  r128/gentle accuracy: {correct.mean():.3f}")
    print(f"  Accuracy drop: {correct.mean() - bl_correct.mean():+.3f} ({(correct.mean() - bl_correct.mean())*100:+.1f}pp)")
    print(f"  Parse rate: {valid.sum()}/{len(results)} ({valid.mean():.1%})")
    if valid.sum() > 0:
        print(f"  Confidence mean: {conf[valid].mean():.1f} (std {conf[valid].std():.1f})")
        print(f"  Ceiling rate (≥95%): {(conf[valid] >= 95).mean():.1%}")
    if valid.sum() > 10 and correct[valid].sum() > 0 and correct[valid].sum() < valid.sum():
        auc = roc_auc_score(correct[valid], conf[valid])
        print(f"  AUROC₂: {auc:.3f}")
    else:
        auc = None

    print(f"\n  Sample responses (first 5):")
    for r in results[:5]:
        print(f"    correct={r['correct']} | {r['response'][:80]}...")

    out = OUTPUT_DIR / "eval_70b_r128_gentle.json"
    with open(out, 'w') as f:
        json.dump({"n": len(results),
                    "baseline_accuracy": float(bl_correct.mean()),
                    "r128_accuracy": float(correct.mean()),
                    "accuracy_drop_pp": float((correct.mean() - bl_correct.mean()) * 100),
                    "parse_rate": float(valid.mean()),
                    "conf_mean": float(conf[valid].mean()) if valid.sum() > 0 else None,
                    "auroc2": float(auc) if auc else None},
                   f, indent=2)
    print(f"  Saved: {out}")

    return model, tokenizer


# ===================================================================
# Stage 2: gentlest — logit readout
# ===================================================================
def run_gentlest(model=None, tokenizer=None):
    """Logit readout for gentlest r4/5e-5 adapter using SINGLE-PASS format."""
    print("\n" + "=" * 60)
    print("70B gentlest r4/5e-5 — Logit Readout (single-pass)")
    print("=" * 60)

    adapter_path = ADAPTERS["gentlest"]

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    # Load gold answers
    confonly_path = Path("results_raw/step4/balanced_confonly_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED_responses.json")
    gold_map = {}
    if confonly_path.exists():
        with open(confonly_path) as f:
            confonly = json.load(f)
        gold_map = {item['id']: item['gold'] for item in confonly}

    print(f"  Loading model + gentlest adapter...")
    import mlx.core as mx
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_PATH, adapter_path=adapter_path)

    # Detect digit token IDs
    digit_ids = {}
    for d in range(101):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if len(toks) == 1:
            digit_ids[d] = toks[0]
    print(f"  Single-token digits: {len(digit_ids)}/101")

    max_digit = max(digit_ids.keys())
    digits = np.array(sorted(digit_ids.keys()))

    results = []
    t0 = time.time()

    for i, item in enumerate(baseline):
        question = item['question']
        correct = item.get('correct', False)

        # Single-pass: generate full response (answer + confidence)
        prompt_text = PROMPT_TEMPLATE.format(question=question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=200, verbose=False)

        text_conf = parse_confidence(response)

        # Now capture logit at confidence digit position
        # Rebuild prompt + response up to "Confidence: "
        conf_match = re.search(r'[Cc]onfidence:?\s*', response)
        logit_ev = float("nan")

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

                mx.eval(logits)
                del x, logits
            except Exception as e:
                if i < 3:
                    print(f"  Error: {e}")

        # Check correctness of generated answer against gold
        qid = item.get('question_id', f'q_{i}')
        gold = gold_map.get(qid, '')
        if gold:
            gen_correct = is_correct_flex(response, gold)
        else:
            gen_correct = correct  # fallback to baseline correctness

        results.append({
            'correct': gen_correct,
            'logit_confidence': logit_ev,
            'text_confidence': text_conf,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            valid = [r for r in results if not np.isnan(r['logit_confidence'])]
            if valid:
                c = np.array([int(r['correct']) for r in valid])
                lc = np.array([r['logit_confidence'] for r in valid])
                tc = np.array([r['text_confidence'] for r in valid])
                tc_valid = ~np.isnan(tc)
                if c.sum() > 0 and c.sum() < len(c):
                    l_auc = roc_auc_score(c, lc)
                    t_auc = roc_auc_score(c[tc_valid], tc[tc_valid]) if tc_valid.sum() > 10 else 0
                    acc = c.mean()
                    print(f"  [{i+1}/{len(baseline)}]  text={t_auc:.3f} logit={l_auc:.3f} acc={acc:.3f}  elapsed={elapsed:.0f}s")

    # Final
    valid = [r for r in results if not np.isnan(r['logit_confidence'])]
    c = np.array([int(r['correct']) for r in valid])
    lc = np.array([r['logit_confidence'] for r in valid])
    tc = np.array([r['text_confidence'] for r in valid])
    tc_valid = ~np.isnan(tc)

    all_correct = np.array([int(r['correct']) for r in results])

    print(f"\n  --- Gentlest Logit Readout Results ---")
    print(f"  Items: {len(valid)}/{len(results)}")
    print(f"  Accuracy: {all_correct.mean():.3f}")
    if c.sum() > 0 and c.sum() < len(c):
        l_auc = roc_auc_score(c, lc)
        print(f"  Logit AUROC₂: {l_auc:.3f}")
        print(f"  Logit conf mean: {lc.mean():.1f} (std {lc.std():.1f})")
        if tc_valid.sum() > 10:
            t_auc = roc_auc_score(c[tc_valid], tc[tc_valid])
            print(f"  Text AUROC₂:  {t_auc:.3f}")
        print(f"\n  Context:")
        print(f"    Gentlest text (unified):    0.740")
        print(f"    Balanced confonly text:      0.682")
        print(f"    Balanced confonly logit:     0.746")
        print(f"    Baseline:                   0.724")
        if l_auc > 0.746:
            print(f"\n  ✓ Gentlest logit EXCEEDS balanced confonly logit!")
        elif l_auc > 0.724:
            print(f"\n  ✓ Gentlest logit exceeds baseline")
        auc_val = float(l_auc)
    else:
        auc_val = None

    out = OUTPUT_DIR / "logit_70b_gentlest.json"
    save = {
        "model": "Llama-3.1-70B",
        "adapter": "gentlest r4/5e-5",
        "n": len(valid),
        "accuracy": float(all_correct.mean()),
        "logit_auroc2": auc_val,
        "logit_conf_mean": float(lc.mean()) if len(lc) > 0 else None,
        "text_auroc2_unified": 0.740,
    }
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="both", choices=["r128", "gentlest", "both"])
    args = parser.parse_args()

    if args.stage in ("r128", "both"):
        model, tokenizer = run_r128()
        # Free model before loading with different adapter
        if args.stage == "both":
            del model, tokenizer
            import mlx.core as mx
            mx.clear_cache()

    if args.stage in ("gentlest", "both"):
        run_gentlest()


if __name__ == "__main__":
    main()
