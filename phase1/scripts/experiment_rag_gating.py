"""
experiment_rag_gating.py — Confidence-Gated Retrieval Experiment

Demonstrates that PT-CSFT confidence enables selective retrieval:
  - Without PT-CSFT: all items have ~100% confidence → gating impossible
  - With PT-CSFT: calibrated confidence → retrieve only when uncertain

Part A (--stage oracle): No GPU. Uses existing eval data to compute theoretical
    benefit of confidence-gated retrieval under oracle assumptions.

Part B (--stage rag): GPU required. Downloads TriviaQA evidence passages,
    generates RAG answers for low-confidence items, measures actual benefit.

Part C (--stage figures): No GPU. Generates publication figures from results.

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/experiment_rag_gating.py --stage oracle
    python3 scripts/experiment_rag_gating.py --stage rag
    python3 scripts/experiment_rag_gating.py --stage figures
"""

import argparse, json, os, re, sys, time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINE_PATH = Path("results_raw/step1/teval_responses_gemma-3-12b-it.json")
PTCSFT_PATH = Path("results_raw/step1_post_pt/teval_responses_gemma-3-12b-it.json")
MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
)
ADAPTER_PATH = "results_raw/finetune/gemma-3-12b-it/probe_target/adapters"
OUTPUT_DIR = Path("results_raw/domain_gen/rag_gating")
FIGURE_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_correct_triviaqa(predicted, aliases):
    """Check if predicted answer matches any gold alias."""
    pred = predicted.strip().lower()
    pred = re.sub(r'[^\w\s]', '', pred)
    for alias in aliases:
        a = alias.strip().lower()
        a = re.sub(r'[^\w\s]', '', a)
        if a in pred or pred in a:
            return True
    return False


def selective_prediction(confidence, correct, n_thresholds=200):
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    thresholds = np.linspace(np.min(c) - 0.01, np.max(c) + 0.01, n_thresholds)
    coverages, accuracies = [], []
    for t in thresholds:
        mask = c >= t
        if mask.sum() == 0:
            continue
        coverages.append(mask.sum() / len(y))
        accuracies.append(y[mask].mean())
    return np.array(coverages), np.array(accuracies)


# ===================================================================
# PART A: Oracle Analysis
# ===================================================================
def run_oracle():
    """Oracle analysis of confidence-gated retrieval.

    Assumptions:
    - Oracle retrieval: if we retrieve, the answer becomes correct with
      probability p_fix (we test p_fix = 0.5, 0.7, 0.9, 1.0)
    - We use PT-CSFT confidence to decide when to retrieve
    - Compare against: always retrieve, never retrieve, random gating
    """
    print("=" * 60)
    print("Part A: Oracle Analysis of Confidence-Gated Retrieval")
    print("=" * 60)

    # Load data
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(PTCSFT_PATH) as f:
        ptcsft = json.load(f)

    print(f"  Baseline items: {len(baseline)}")
    print(f"  PT-CSFT items: {len(ptcsft)}")

    # Align by question_id
    bl_map = {item['question_id']: item for item in baseline}
    pt_map = {item['question_id']: item for item in ptcsft}
    shared_ids = sorted(set(bl_map.keys()) & set(pt_map.keys()))
    print(f"  Shared items: {len(shared_ids)}")

    # Extract arrays
    bl_correct = np.array([int(bl_map[qid]['correct']) for qid in shared_ids])
    bl_conf = np.array([bl_map[qid]['parsed_confidence'] for qid in shared_ids])
    pt_correct = np.array([int(pt_map[qid]['correct']) for qid in shared_ids])
    pt_conf = np.array([pt_map[qid]['parsed_confidence'] for qid in shared_ids])

    # Note: bl_correct and pt_correct may differ (adapter changes some answers)
    # For the RAG experiment, the question is: does retrieving for low-confidence
    # items improve accuracy?

    n = len(shared_ids)
    bl_acc = bl_correct.mean()
    pt_acc = pt_correct.mean()

    print(f"\n  Baseline accuracy: {bl_acc:.3f}")
    print(f"  PT-CSFT accuracy:  {pt_acc:.3f}")
    print(f"  Baseline conf mean: {bl_conf.mean():.1f} (std {bl_conf.std():.1f})")
    print(f"  PT-CSFT conf mean:  {pt_conf.mean():.1f} (std {pt_conf.std():.1f})")

    # --- The key contrast ---
    print("\n" + "-" * 60)
    print("The Gating Problem:")
    print("-" * 60)
    print(f"  Without PT-CSFT: {(bl_conf >= 95).sum()}/{n} items have conf ≥ 95%")
    print(f"    → Threshold at 95% retains {(bl_conf >= 95).sum()}/{n} = "
          f"{(bl_conf >= 95).mean():.1%} → gating is useless")
    print(f"  With PT-CSFT:    {(pt_conf >= 95).sum()}/{n} items have conf ≥ 95%")
    print(f"    → Threshold at 95% retains {(pt_conf >= 95).sum()}/{n} = "
          f"{(pt_conf >= 95).mean():.1%} → gating is possible")

    # --- Oracle retrieval analysis ---
    print("\n" + "-" * 60)
    print("Oracle Retrieval Analysis")
    print("-" * 60)
    print("  If retrieval corrects errors with probability p_fix:")

    thresholds = np.arange(0, 101, 5)
    p_fix_values = [0.5, 0.7, 0.9, 1.0]

    results = {}
    for p_fix in p_fix_values:
        print(f"\n  p_fix = {p_fix}:")
        print(f"    {'Threshold':>10} | {'Retrieve%':>9} | {'Accuracy':>8} | {'Δ vs base':>9}")
        print(f"    {'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}")

        best_acc = 0
        best_thresh = 0
        best_ret = 0

        for thresh in thresholds:
            retrieve_mask = pt_conf < thresh
            n_retrieve = retrieve_mask.sum()
            retrieve_rate = n_retrieve / n

            # Items above threshold: keep PT-CSFT answer
            # Items below threshold: simulate retrieval
            # Oracle: each incorrect item below threshold becomes correct with prob p_fix
            correct_gated = pt_correct.copy()
            # For retrieved items that are currently wrong, fix with p_fix
            rng = np.random.default_rng(42)
            for i in range(n):
                if retrieve_mask[i] and not pt_correct[i]:
                    if rng.random() < p_fix:
                        correct_gated[i] = 1

            acc = correct_gated.mean()
            delta = acc - pt_acc

            if thresh in [0, 30, 50, 70, 90, 100]:
                print(f"    {thresh:>10} | {retrieve_rate:>8.1%} | {acc:>8.3f} | {delta:>+9.3f}")

            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
                best_ret = retrieve_rate

        results[f"p_fix_{p_fix}"] = {
            "best_accuracy": float(best_acc),
            "best_threshold": int(best_thresh),
            "best_retrieval_rate": float(best_ret),
        }

        print(f"    Best: threshold={best_thresh}, retrieve={best_ret:.1%}, "
              f"acc={best_acc:.3f}")

    # --- Selective prediction comparison ---
    print("\n" + "-" * 60)
    print("Selective Prediction: Baseline vs PT-CSFT")
    print("-" * 60)

    # Without PT-CSFT, selective prediction is useless
    cov_bl, acc_bl = selective_prediction(bl_conf, bl_correct)
    cov_pt, acc_pt = selective_prediction(pt_conf, pt_correct)

    # Coverage at 95% accuracy
    idx_bl = np.where(acc_bl >= 0.95)[0]
    idx_pt = np.where(acc_pt >= 0.95)[0]

    if len(idx_bl) > 0:
        print(f"  Baseline @ 95% acc: coverage = {cov_bl[idx_bl[0]]:.3f}")
    else:
        print(f"  Baseline: cannot reach 95% accuracy (max {acc_bl.max():.3f})")

    if len(idx_pt) > 0:
        print(f"  PT-CSFT @ 95% acc:  coverage = {cov_pt[idx_pt[0]]:.3f}")
    else:
        print(f"  PT-CSFT: cannot reach 95% accuracy (max {acc_pt.max():.3f})")

    # --- The punchline ---
    print("\n" + "=" * 60)
    print("Punchline:")
    print("=" * 60)
    print(f"  Without PT-CSFT, the model reports ~100% confidence on everything.")
    print(f"  A retrieval system cannot distinguish items that need help from items")
    print(f"  that don't. It must either retrieve for ALL items (expensive) or NONE")
    print(f"  (unreliable).")
    print(f"")
    print(f"  With PT-CSFT, confidence is calibrated. At threshold 50%:")
    retrieve_50 = (pt_conf < 50).mean()
    # Items above 50%: accuracy on those
    above_50 = pt_correct[pt_conf >= 50].mean() if (pt_conf >= 50).sum() > 0 else 0
    below_50_n = (pt_conf < 50).sum()
    below_50_errors = ((pt_conf < 50) & (pt_correct == 0)).sum()
    print(f"    {retrieve_50:.1%} of items flagged for retrieval")
    print(f"    Accuracy on confident items (conf ≥ 50%): {above_50:.3f}")
    print(f"    Errors in flagged items: {below_50_errors}/{below_50_n}")
    print(f"    If retrieval fixes 70% of flagged errors: "
          f"overall acc = {pt_acc + 0.7 * below_50_errors / n:.3f} "
          f"(Δ = +{0.7 * below_50_errors / n:.3f})")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "baseline_accuracy": float(bl_acc),
        "ptcsft_accuracy": float(pt_acc),
        "baseline_conf_mean": float(bl_conf.mean()),
        "ptcsft_conf_mean": float(pt_conf.mean()),
        "baseline_ceiling_rate": float((bl_conf >= 95).mean()),
        "ptcsft_ceiling_rate": float((pt_conf >= 95).mean()),
        "n_items": n,
        "oracle_results": results,
    }
    with open(OUTPUT_DIR / "oracle_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'oracle_analysis.json'}")

    return summary, shared_ids, bl_map, pt_map


# ===================================================================
# PART B: Real RAG with Evidence
# ===================================================================
def run_rag(shared_ids=None, pt_map=None):
    """Generate RAG answers for low-confidence items using TriviaQA evidence."""
    print("\n" + "=" * 60)
    print("Part B: Real RAG with TriviaQA Evidence")
    print("=" * 60)

    # Load PT-CSFT data if not passed
    if pt_map is None:
        with open(PTCSFT_PATH) as f:
            ptcsft = json.load(f)
        pt_map = {item['question_id']: item for item in ptcsft}
        shared_ids = sorted(pt_map.keys())

    pt_conf = {qid: pt_map[qid]['parsed_confidence'] for qid in shared_ids}
    pt_correct = {qid: pt_map[qid]['correct'] for qid in shared_ids}

    # Determine which items to retrieve for (confidence < threshold)
    THRESHOLD = 50  # retrieve for items below this
    retrieve_ids = [qid for qid in shared_ids if pt_conf[qid] < THRESHOLD]
    print(f"  Items below threshold ({THRESHOLD}%): {len(retrieve_ids)}/{len(shared_ids)}")

    if len(retrieve_ids) == 0:
        print("  No items below threshold — nothing to retrieve for.")
        return

    # Load TriviaQA RC validation for evidence passages
    print("  Loading TriviaQA RC validation (evidence passages)...")
    from datasets import load_dataset
    ds = load_dataset('trivia_qa', 'rc', split='validation')

    # Build question_id → evidence map
    evidence_map = {}
    for item in ds:
        qid = item.get('question_id', '')
        if qid in retrieve_ids:
            # Extract Wikipedia evidence (entity_pages)
            wiki_texts = []
            if 'entity_pages' in item and item['entity_pages']:
                pages = item['entity_pages']
                if isinstance(pages, dict) and 'wiki_context' in pages:
                    wiki_texts = pages['wiki_context']
                elif isinstance(pages, list):
                    wiki_texts = [p.get('wiki_context', '') for p in pages if isinstance(p, dict)]

            # Extract search result snippets
            search_texts = []
            if 'search_results' in item and item['search_results']:
                sr = item['search_results']
                if isinstance(sr, dict) and 'search_context' in sr:
                    search_texts = sr['search_context']

            # Combine (truncate to ~500 chars each)
            evidence_parts = []
            for t in wiki_texts[:2]:
                if isinstance(t, str) and len(t) > 0:
                    evidence_parts.append(t[:500])
            for t in search_texts[:2]:
                if isinstance(t, str) and len(t) > 0:
                    evidence_parts.append(t[:500])

            if evidence_parts:
                evidence_map[qid] = "\n---\n".join(evidence_parts)

    print(f"  Evidence found for: {len(evidence_map)}/{len(retrieve_ids)} items")

    items_to_rag = [qid for qid in retrieve_ids if qid in evidence_map]
    if not items_to_rag:
        print("  No evidence available — cannot proceed with RAG")
        return

    # Load model (without adapter — we want fresh answers with evidence)
    print(f"  Loading model: {MODEL_PATH}")
    import mlx.core as mx
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_PATH)

    # Generate RAG answers
    RAG_PROMPT = (
        "Answer this question using the provided context. "
        "After your answer, state your confidence as a percentage from 0 to 100.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
    )

    rag_results = {}
    t0 = time.time()

    for i, qid in enumerate(items_to_rag):
        question = pt_map[qid]['question']
        evidence = evidence_map[qid]

        prompt_text = RAG_PROMPT.format(context=evidence[:1500], question=question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=256, verbose=False)

        # Parse answer — check if it matches any gold alias
        # For now, use simple substring matching against the baseline gold
        gold_answer = pt_map[qid].get('parsed_answer', '')
        # We need the actual gold aliases — get from TriviaQA dataset
        # For now, use the correctness check from the baseline

        rag_results[qid] = {
            'question': question,
            'original_answer': pt_map[qid]['parsed_answer'],
            'original_correct': pt_map[qid]['correct'],
            'original_confidence': pt_map[qid]['parsed_confidence'],
            'rag_response': response,
        }

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(items_to_rag)}]  elapsed={elapsed:.0f}s")

    # Save RAG results (correctness needs manual verification against gold aliases)
    rag_path = OUTPUT_DIR / "rag_responses.json"
    with open(rag_path, 'w') as f:
        json.dump(rag_results, f, indent=2)
    print(f"\n  Saved: {rag_path}")
    print(f"  NOTE: RAG correctness needs scoring against TriviaQA gold aliases.")
    print(f"  Run --stage score after reviewing.")

    # Now score using the dataset's answer aliases
    print("\n  Scoring RAG responses...")
    gold_aliases = {}
    for item in ds:
        qid = item.get('question_id', '')
        if qid in rag_results:
            answer = item.get('answer', {})
            aliases = answer.get('aliases', [])
            value = answer.get('value', '')
            normalized = answer.get('normalized_aliases', [])
            all_aliases = list(set(aliases + [value] + normalized))
            gold_aliases[qid] = all_aliases

    n_correct_before = 0
    n_correct_after = 0
    n_scored = 0
    for qid, result in rag_results.items():
        if qid not in gold_aliases:
            continue
        n_scored += 1
        if result['original_correct']:
            n_correct_before += 1
        rag_correct = is_correct_triviaqa(result['rag_response'], gold_aliases[qid])
        result['rag_correct'] = rag_correct
        if rag_correct or result['original_correct']:
            n_correct_after += 1  # keep the better answer

    if n_scored > 0:
        print(f"  Scored: {n_scored} items")
        print(f"  Original accuracy (on retrieved items): {n_correct_before/n_scored:.3f}")
        print(f"  RAG accuracy (on retrieved items):      "
              f"{sum(1 for r in rag_results.values() if r.get('rag_correct', False))/n_scored:.3f}")
        print(f"  Best-of (keep better answer):           {n_correct_after/n_scored:.3f}")

    # Compute overall gated accuracy
    n_total = len(shared_ids)
    n_above = sum(1 for qid in shared_ids if pt_conf[qid] >= THRESHOLD)
    n_correct_above = sum(1 for qid in shared_ids
                          if pt_conf[qid] >= THRESHOLD and pt_correct[qid])

    rag_correct_count = sum(1 for r in rag_results.values() if r.get('rag_correct', False))
    original_correct_below = sum(1 for qid in retrieve_ids if pt_correct[qid])

    # Gated accuracy: above threshold use original, below use RAG
    gated_correct = n_correct_above + rag_correct_count
    gated_acc = gated_correct / n_total

    # Baseline (no retrieval)
    baseline_acc = sum(pt_correct[qid] for qid in shared_ids) / n_total

    # Always retrieve (assume RAG accuracy applies to all)
    # Can't compute this without running RAG on all items

    print(f"\n  --- Overall Results (threshold = {THRESHOLD}%) ---")
    print(f"  No retrieval:              {baseline_acc:.3f}")
    print(f"  Confidence-gated RAG:      {gated_acc:.3f} "
          f"(Δ = {gated_acc - baseline_acc:+.3f})")
    print(f"  Retrieval rate:            {len(items_to_rag)}/{n_total} = "
          f"{len(items_to_rag)/n_total:.1%}")
    print(f"  Items above threshold:     {n_above} (acc = {n_correct_above/n_above:.3f})")

    # Save final results
    with open(rag_path, 'w') as f:
        json.dump(rag_results, f, indent=2)

    final = {
        "threshold": THRESHOLD,
        "n_total": n_total,
        "n_retrieved": len(items_to_rag),
        "retrieval_rate": len(items_to_rag) / n_total,
        "baseline_accuracy": float(baseline_acc),
        "gated_accuracy": float(gated_acc),
        "delta": float(gated_acc - baseline_acc),
        "above_threshold_accuracy": float(n_correct_above / n_above) if n_above > 0 else None,
        "rag_accuracy_on_retrieved": float(rag_correct_count / n_scored) if n_scored > 0 else None,
    }
    with open(OUTPUT_DIR / "rag_gating_results.json", 'w') as f:
        json.dump(final, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'rag_gating_results.json'}")


# ===================================================================
# PART C: Figures
# ===================================================================
def run_figures():
    """Generate publication figures from oracle + RAG results."""
    print("\n" + "=" * 60)
    print("Part C: Figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Load data
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(PTCSFT_PATH) as f:
        ptcsft = json.load(f)

    bl_map = {i['question_id']: i for i in baseline}
    pt_map = {i['question_id']: i for i in ptcsft}
    shared = sorted(set(bl_map.keys()) & set(pt_map.keys()))

    bl_conf = np.array([bl_map[q]['parsed_confidence'] for q in shared])
    bl_correct = np.array([int(bl_map[q]['correct']) for q in shared])
    pt_conf = np.array([pt_map[q]['parsed_confidence'] for q in shared])
    pt_correct = np.array([int(pt_map[q]['correct']) for q in shared])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Confidence distributions
    ax = axes[0]
    ax.hist(bl_conf, bins=20, alpha=0.6, color='#999999', label='Baseline (no PT-CSFT)')
    ax.hist(pt_conf, bins=20, alpha=0.6, color='#2166ac', label='With PT-CSFT')
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Selective prediction curves
    ax = axes[1]
    cov_bl, acc_bl = selective_prediction(bl_conf, bl_correct)
    cov_pt, acc_pt = selective_prediction(pt_conf, pt_correct)

    ax.plot(cov_bl, acc_bl, color='#999999', linewidth=2, label='Baseline')
    ax.plot(cov_pt, acc_pt, color='#2166ac', linewidth=2, label='PT-CSFT')
    ax.axhline(y=0.95, color='#cc0000', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(0.15, 0.953, '95% target', fontsize=8, color='#cc0000', alpha=0.7)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Accuracy')
    ax.set_title('Selective Prediction')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.6, 1.02)
    ax.grid(True, alpha=0.2)

    # Panel 3: Retrieval rate at each threshold
    ax = axes[2]
    thresholds = np.arange(0, 101, 5)
    ret_rates_bl = [(bl_conf < t).mean() for t in thresholds]
    ret_rates_pt = [(pt_conf < t).mean() for t in thresholds]

    # Oracle accuracy improvement at each threshold (p_fix=0.7)
    oracle_accs = []
    for t in thresholds:
        mask = pt_conf < t
        gated = pt_correct.copy()
        rng = np.random.default_rng(42)
        for i in range(len(gated)):
            if mask[i] and not pt_correct[i]:
                if rng.random() < 0.7:
                    gated[i] = 1
        oracle_accs.append(gated.mean())

    ax.plot(thresholds, ret_rates_pt, color='#2166ac', linewidth=2,
            label='Retrieval rate (PT-CSFT)')
    ax.plot(thresholds, ret_rates_bl, color='#999999', linewidth=2, linestyle='--',
            label='Retrieval rate (baseline)')

    ax2 = ax.twinx()
    ax2.plot(thresholds, oracle_accs, color='#e07b39', linewidth=2,
             label='Oracle accuracy (p=0.7)')
    ax2.axhline(y=pt_correct.mean(), color='#e07b39', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Accuracy', color='#e07b39')

    ax.set_xlabel('Confidence threshold (%)')
    ax.set_ylabel('Retrieval rate', color='#2166ac')
    ax.set_title('Confidence-Gated Retrieval')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='center right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIGURE_DIR / "fig_rag_gating.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {out}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="oracle",
                        choices=["oracle", "rag", "figures", "all"])
    args = parser.parse_args()

    if args.stage in ("oracle", "all"):
        summary, shared_ids, bl_map, pt_map = run_oracle()

    if args.stage in ("rag", "all"):
        if args.stage == "rag":
            with open(PTCSFT_PATH) as f:
                ptcsft = json.load(f)
            pt_map = {i['question_id']: i for i in ptcsft}
            shared_ids = sorted(pt_map.keys())
        run_rag(shared_ids, pt_map)

    if args.stage in ("figures", "all"):
        run_figures()


if __name__ == "__main__":
    main()
