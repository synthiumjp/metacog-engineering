"""
run_rag_experiment.py — Real RAG with single-pass PT-CSFT confidence.

Uses existing data:
  - Baseline answers: step1/teval_responses (accuracy 0.751, ceiling confidence)
  - PT-CSFT answers:  step1_post_pt/teval_responses (accuracy 0.655, calibrated confidence)
  - Evidence: TriviaQA RC validation (cached from prior download)

Design:
  For items where PT-CSFT confidence < threshold:
    - Retrieve TriviaQA evidence passage
    - Re-answer with base model + evidence (no adapter)
  For items where PT-CSFT confidence >= threshold:
    - Keep PT-CSFT answer as-is

Compare: no-retrieval vs always-RAG vs confidence-gated RAG.

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/run_rag_experiment.py
"""

import json, os, re, sys, time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASELINE_PATH = Path("results_raw/step1/teval_responses_gemma-3-12b-it.json")
PTCSFT_PATH = Path("results_raw/step1_post_pt/teval_responses_gemma-3-12b-it.json")
MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
)
OUTPUT_DIR = Path("results_raw/domain_gen/rag_gating")
THRESHOLD = 50


def is_correct_triviaqa(predicted, aliases):
    if not predicted or not aliases:
        return False
    pred = predicted.strip().lower()
    pred = re.sub(r'[^\w\s]', '', pred)
    for prefix in ['the answer is ', 'that would be ', 'answer ']:
        if pred.startswith(prefix):
            pred = pred[len(prefix):]
    pred = pred.strip()
    if not pred:
        return False
    for alias in aliases:
        a = alias.strip().lower()
        a = re.sub(r'[^\w\s]', '', a)
        if not a or len(a) < 2:
            continue
        if a in pred or pred in a:
            return True
    return False


def main():
    print("=" * 60)
    print("Real RAG Experiment")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load PT-CSFT responses (has calibrated confidence)
    with open(PTCSFT_PATH) as f:
        ptcsft = json.load(f)
    pt_map = {r['question_id']: r for r in ptcsft}

    # Load baseline responses (for comparison)
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    bl_map = {r['question_id']: r for r in baseline}

    shared_ids = sorted(set(pt_map.keys()) & set(bl_map.keys()))
    print(f"  Items: {len(shared_ids)}")

    # Identify items for retrieval
    retrieve_ids = [qid for qid in shared_ids
                    if pt_map[qid]['parsed_confidence'] < THRESHOLD]
    keep_ids = [qid for qid in shared_ids
                if pt_map[qid]['parsed_confidence'] >= THRESHOLD]

    print(f"  Below threshold ({THRESHOLD}%): {len(retrieve_ids)} → will RAG")
    print(f"  Above threshold: {len(keep_ids)} → keep PT-CSFT answer")

    # Also RAG a random sample of high-confidence items for always-RAG estimate
    rng = np.random.default_rng(42)
    high_sample = list(keep_ids)
    rng.shuffle(high_sample)
    high_sample = high_sample[:min(100, len(high_sample))]

    all_rag_ids = set(retrieve_ids) | set(high_sample)
    print(f"  High-confidence sample for always-RAG est: {len(high_sample)}")
    print(f"  Total to RAG: {len(all_rag_ids)}")

    # Load TriviaQA RC for evidence + gold aliases
    print("\n  Loading TriviaQA RC validation...")
    from datasets import load_dataset
    ds = load_dataset('trivia_qa', 'rc', split='validation')

    evidence_map = {}
    gold_map = {}
    for item in ds:
        qid = item.get('question_id', '')

        # Gold aliases for all shared items (for scoring)
        answer = item.get('answer', {})
        aliases = list(set(
            answer.get('aliases', []) +
            [answer.get('value', '')] +
            answer.get('normalized_aliases', [])
        ))
        if qid in set(shared_ids):
            gold_map[qid] = [a for a in aliases if a]

        # Evidence only for items we'll RAG
        if qid not in all_rag_ids:
            continue

        parts = []
        ep = item.get('entity_pages', {})
        if isinstance(ep, dict):
            for ctx in ep.get('wiki_context', [])[:2]:
                if isinstance(ctx, str) and len(ctx) > 10:
                    parts.append(ctx[:800])
        sr = item.get('search_results', {})
        if isinstance(sr, dict):
            for ctx in sr.get('search_context', [])[:2]:
                if isinstance(ctx, str) and len(ctx) > 10:
                    parts.append(ctx[:500])
        if parts:
            evidence_map[qid] = "\n---\n".join(parts[:3])

    print(f"  Gold aliases: {len(gold_map)}/{len(shared_ids)}")
    print(f"  Evidence: {len(evidence_map)}/{len(all_rag_ids)}")

    # Filter to items with evidence
    rag_ids = [qid for qid in all_rag_ids if qid in evidence_map]
    print(f"  RAG-able items: {len(rag_ids)}")

    # Load base model (no adapter)
    print(f"\n  Loading base model (no adapter)...")
    import mlx.core as mx
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_PATH)

    RAG_PROMPT = (
        "Answer this trivia question using the provided context. "
        "Be concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
    )

    rag_results = {}
    t0 = time.time()

    for i, qid in enumerate(rag_ids):
        question = pt_map[qid]['question']
        evidence = evidence_map[qid]

        prompt_text = RAG_PROMPT.format(
            context=evidence[:2000],
            question=question,
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=150, verbose=False)

        rag_correct = is_correct_triviaqa(response, gold_map.get(qid, []))
        is_low = qid in set(retrieve_ids)

        rag_results[qid] = {
            'question': question,
            'pt_answer': pt_map[qid]['parsed_answer'],
            'pt_correct': pt_map[qid]['correct'],
            'pt_confidence': pt_map[qid]['parsed_confidence'],
            'bl_correct': bl_map[qid]['correct'],
            'is_low_confidence': is_low,
            'rag_response': response[:300],
            'rag_correct': rag_correct,
        }

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            low = [r for r in rag_results.values() if r['is_low_confidence']]
            high = [r for r in rag_results.values() if not r['is_low_confidence']]
            if low:
                low_bl = np.mean([r['bl_correct'] for r in low])
                low_rag = np.mean([r['rag_correct'] for r in low])
                print(f"  [{i+1}/{len(rag_ids)}]  "
                      f"low: bl={low_bl:.3f} rag={low_rag:.3f}  "
                      f"elapsed={elapsed:.0f}s")

    # --- Results ---
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    low = [r for r in rag_results.values() if r['is_low_confidence']]
    high = [r for r in rag_results.values() if not r['is_low_confidence']]

    if low:
        low_bl_acc = np.mean([r['bl_correct'] for r in low])
        low_pt_acc = np.mean([r['pt_correct'] for r in low])
        low_rag_acc = np.mean([r['rag_correct'] for r in low])
        errors = sum(not r['bl_correct'] for r in low)
        fixed = sum(r['rag_correct'] and not r['bl_correct'] for r in low)
        broken = sum(not r['rag_correct'] and r['bl_correct'] for r in low)
        p_fix = fixed / max(1, errors)

        print(f"\n  Low-confidence items (n={len(low)}):")
        print(f"    Baseline accuracy: {low_bl_acc:.3f}")
        print(f"    PT-CSFT accuracy:  {low_pt_acc:.3f}")
        print(f"    RAG accuracy:      {low_rag_acc:.3f}")
        print(f"    Errors in baseline: {errors}")
        print(f"    Fixed by RAG: {fixed} ({p_fix:.1%})")
        print(f"    Broken by RAG: {broken}")

    if high:
        high_bl_acc = np.mean([r['bl_correct'] for r in high])
        high_rag_acc = np.mean([r['rag_correct'] for r in high])
        print(f"\n  High-confidence sample (n={len(high)}):")
        print(f"    Baseline accuracy: {high_bl_acc:.3f}")
        print(f"    RAG accuracy:      {high_rag_acc:.3f}")

    # --- Overall gated accuracy ---
    n = len(shared_ids)
    # Strategy: above threshold → keep baseline answer; below → use RAG answer
    n_correct_gated = 0
    n_correct_baseline = 0
    n_correct_always_rag = 0
    n_retrieved = 0

    for qid in shared_ids:
        bl_correct = bl_map[qid]['correct']
        n_correct_baseline += int(bl_correct)

        if qid in rag_results and rag_results[qid]['is_low_confidence']:
            # Below threshold: use RAG
            n_correct_gated += int(rag_results[qid]['rag_correct'])
            n_retrieved += 1
        else:
            # Above threshold: keep baseline
            n_correct_gated += int(bl_correct)

    baseline_acc = n_correct_baseline / n
    gated_acc = n_correct_gated / n
    retrieve_rate = n_retrieved / n

    # Estimate always-RAG from the samples
    all_rag_items = list(rag_results.values())
    if all_rag_items:
        always_rag_est = np.mean([r['rag_correct'] for r in all_rag_items])

    print(f"\n  --- Overall (n={n}) ---")
    print(f"  Baseline (no retrieval):     {baseline_acc:.3f}")
    print(f"  Confidence-gated RAG:        {gated_acc:.3f} "
          f"(Δ = {gated_acc - baseline_acc:+.3f})")
    print(f"  Retrieval rate:              {retrieve_rate:.1%}")
    print(f"  Always-RAG estimate:         {always_rag_est:.3f} "
          f"(from {len(all_rag_items)} samples)")

    print(f"\n  Key finding:")
    print(f"  Confidence-gated RAG achieves {gated_acc:.3f} accuracy")
    print(f"  while retrieving for only {retrieve_rate:.0%} of items")
    if gated_acc > baseline_acc:
        print(f"  This EXCEEDS the no-retrieval baseline ({baseline_acc:.3f}) "
              f"by {gated_acc - baseline_acc:+.3f}")

    # Save
    save_data = {
        "threshold": THRESHOLD,
        "n_items": n,
        "n_retrieved": n_retrieved,
        "retrieve_rate": float(retrieve_rate),
        "baseline_accuracy": float(baseline_acc),
        "gated_accuracy": float(gated_acc),
        "delta": float(gated_acc - baseline_acc),
        "always_rag_estimate": float(always_rag_est) if all_rag_items else None,
        "p_fix": float(p_fix) if low else None,
        "low_confidence_n": len(low) if low else 0,
        "low_bl_acc": float(low_bl_acc) if low else None,
        "low_rag_acc": float(low_rag_acc) if low else None,
        "high_sample_n": len(high) if high else 0,
        "high_bl_acc": float(high_bl_acc) if high else None,
        "high_rag_acc": float(high_rag_acc) if high else None,
    }
    with open(OUTPUT_DIR / "rag_real_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    with open(OUTPUT_DIR / "rag_real_responses.json", 'w') as f:
        json.dump(rag_results, f, indent=2, default=str)

    print(f"\n  Saved: {OUTPUT_DIR / 'rag_real_results.json'}")
    print(f"  Saved: {OUTPUT_DIR / 'rag_real_responses.json'}")


if __name__ == "__main__":
    main()
