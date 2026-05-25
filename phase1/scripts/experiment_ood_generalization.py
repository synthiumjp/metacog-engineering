"""
experiment_ood_generalization.py — OOD Confidence Generalization Test

Tests whether PT-CSFT confidence trained on TriviaQA transfers to
NaturalQuestions (same format, different distribution).

Design:
  1. Load Gemma 12B + TriviaQA PT-CSFT adapter
  2. Generate on NQ Open validation (500 items) using SAME prompt format
  3. Compute AUROC₂ — does the trained confidence discriminate on OOD data?
  4. Compare to TriviaQA in-distribution AUROC₂ (0.833)
  5. Also run baseline (no adapter) on same items for contrast

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/experiment_ood_generalization.py
"""

import json, os, re, sys, time
import numpy as np
from pathlib import Path

MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
)
ADAPTER_PATH = "results_raw/finetune/gemma-3-12b-it/probe_target/adapters"
OUTPUT_DIR = Path("results_raw/domain_gen/ood_generalization")
N_EVAL = 500
SEED = 42

# Must match the TriviaQA training prompt EXACTLY
PROMPT_TEMPLATE = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


def is_correct_nq(predicted, gold_answers):
    """Check if predicted answer matches any gold NQ answer."""
    if not predicted or not gold_answers:
        return False
    pred = predicted.strip().lower()
    pred = re.sub(r'[^\w\s]', '', pred)
    # Remove common prefixes
    for prefix in ['the answer is ', 'that would be ', 'it is ', 'answer ']:
        if pred.startswith(prefix):
            pred = pred[len(prefix):]
    pred = pred.strip()
    if not pred:
        return False
    for gold in gold_answers:
        g = gold.strip().lower()
        g = re.sub(r'[^\w\s]', '', g)
        if not g or len(g) < 2:
            continue
        if g in pred or pred in g:
            return True
    return False


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


def run_eval(model, tokenizer, items, label):
    """Run evaluation on items, return per-item results."""
    from mlx_lm import generate

    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        question = item['question']
        gold = item['answer']

        prompt_text = PROMPT_TEMPLATE.format(question=question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=200, verbose=False)

        correct = is_correct_nq(response, gold)
        conf = parse_confidence(response)

        results.append({
            'question': question,
            'gold': gold[:3],  # truncate for storage
            'response': response[:200],
            'correct': correct,
            'confidence': conf,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r['correct'] for r in results])
            confs = [r['confidence'] for r in results if not np.isnan(r['confidence'])]
            conf_mean = np.mean(confs) if confs else float('nan')
            print(f"  [{label}] [{i+1}/{len(items)}]  "
                  f"acc={acc:.3f}  conf_mean={conf_mean:.1f}  "
                  f"elapsed={elapsed:.0f}s")

    return results


def main():
    print("=" * 60)
    print("OOD Confidence Generalization Test")
    print("TriviaQA adapter → NaturalQuestions")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load NQ Open
    print("\n  Loading NQ Open validation...")
    from datasets import load_dataset
    ds = load_dataset('nq_open', split='validation')
    print(f"  NQ Open validation: {len(ds)} items")

    # Sample N_EVAL items
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(ds))[:N_EVAL]
    items = [{'question': ds[int(i)]['question'], 'answer': ds[int(i)]['answer']}
             for i in indices]
    print(f"  Sampled: {N_EVAL} items")

    # --- Condition 1: With PT-CSFT adapter ---
    print(f"\n  Loading model + PT-CSFT adapter...")
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    print(f"  Running PT-CSFT evaluation...")
    ptcsft_results = run_eval(model, tokenizer, items, "PT-CSFT")

    # Free model
    del model
    mx.metal.clear_cache() if hasattr(mx, 'metal') else None

    # --- Condition 2: Baseline (no adapter) ---
    print(f"\n  Loading model (no adapter)...")
    model, tokenizer = load(MODEL_PATH)
    print(f"  Running baseline evaluation...")
    baseline_results = run_eval(model, tokenizer, items, "Baseline")

    del model

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score

    for label, results in [("Baseline", baseline_results), ("PT-CSFT", ptcsft_results)]:
        correct = np.array([int(r['correct']) for r in results])
        conf = np.array([r['confidence'] for r in results])
        valid = ~np.isnan(conf)

        acc = correct.mean()
        parse_rate = valid.sum() / len(results)

        print(f"\n  {label}:")
        print(f"    Accuracy: {acc:.3f}")
        print(f"    Parse rate: {parse_rate:.1%}")

        if valid.sum() > 10 and correct[valid].sum() > 0 and correct[valid].sum() < valid.sum():
            auc = roc_auc_score(correct[valid], conf[valid])
            conf_mean = conf[valid].mean()
            conf_std = conf[valid].std()
            print(f"    AUROC₂: {auc:.3f}")
            print(f"    Conf mean: {conf_mean:.1f} (std {conf_std:.1f})")

            # Ceiling rate
            ceiling = (conf[valid] >= 95).mean()
            print(f"    Ceiling rate (≥95%): {ceiling:.1%}")
        else:
            print(f"    AUROC₂: cannot compute (insufficient correct/incorrect items)")

    # --- Comparison with in-distribution ---
    print(f"\n  --- Comparison ---")
    print(f"  In-distribution (TriviaQA):")
    print(f"    Baseline AUROC₂: ~0.684")
    print(f"    PT-CSFT AUROC₂:  0.833")
    print(f"    Δ: +0.149")

    pt_correct = np.array([int(r['correct']) for r in ptcsft_results])
    pt_conf = np.array([r['confidence'] for r in ptcsft_results])
    pt_valid = ~np.isnan(pt_conf)
    bl_correct = np.array([int(r['correct']) for r in baseline_results])
    bl_conf = np.array([r['confidence'] for r in baseline_results])
    bl_valid = ~np.isnan(bl_conf)

    if pt_valid.sum() > 10 and bl_valid.sum() > 10:
        pt_auc = roc_auc_score(pt_correct[pt_valid], pt_conf[pt_valid])
        bl_auc = roc_auc_score(bl_correct[bl_valid], bl_conf[bl_valid])
        print(f"\n  Out-of-distribution (NaturalQuestions):")
        print(f"    Baseline AUROC₂: {bl_auc:.3f}")
        print(f"    PT-CSFT AUROC₂:  {pt_auc:.3f}")
        print(f"    Δ: {pt_auc - bl_auc:+.3f}")

        transfer_ratio = (pt_auc - bl_auc) / (0.833 - 0.684) if (0.833 - 0.684) > 0 else 0
        print(f"\n  Transfer ratio: {transfer_ratio:.1%}")
        print(f"    (fraction of in-distribution Δ retained on OOD data)")

        if pt_auc > bl_auc + 0.05:
            print(f"\n  ✓ Confidence signal TRANSFERS to OOD data")
        elif pt_auc > bl_auc:
            print(f"\n  ~ Modest transfer to OOD data")
        else:
            print(f"\n  ✗ Confidence signal does NOT transfer to OOD data")

    # Save
    save_data = {
        "dataset": "nq_open",
        "n_eval": N_EVAL,
        "in_distribution": {
            "dataset": "triviaqa",
            "baseline_auroc2": 0.684,
            "ptcsft_auroc2": 0.833,
        },
        "ood": {
            "baseline_accuracy": float(bl_correct.mean()),
            "ptcsft_accuracy": float(pt_correct.mean()),
            "baseline_auroc2": float(bl_auc) if bl_valid.sum() > 10 else None,
            "ptcsft_auroc2": float(pt_auc) if pt_valid.sum() > 10 else None,
        },
        "baseline_results": baseline_results,
        "ptcsft_results": ptcsft_results,
    }
    with open(OUTPUT_DIR / "ood_nq_results.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR / 'ood_nq_results.json'}")


if __name__ == "__main__":
    main()
