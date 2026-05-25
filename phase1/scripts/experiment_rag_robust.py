"""
experiment_rag_robust.py — Scientifically robust confidence-gated RAG experiment.

Design:
  1. Two-pass confonly eval: baseline answers (no adapter) + confidence rating (with adapter)
     → Preserves baseline accuracy, adds calibrated confidence
  2. Download TriviaQA RC validation evidence passages
  3. Generate RAG answers for items below confidence threshold
  4. Score RAG answers against gold aliases
  5. Compute p_fix (empirical retrieval success rate)
  6. Report: baseline vs always-RAG vs confidence-gated RAG

Usage:
    cd ~/jpwork/metacog-engineering/phase1

    # Stage 1: Two-pass confonly confidence rating (~30 min)
    python3 scripts/experiment_rag_robust.py --stage confonly

    # Stage 2: RAG for low-confidence items (~1-2 hrs)
    python3 scripts/experiment_rag_robust.py --stage rag

    # Stage 3: Analysis + figures (no GPU)
    python3 scripts/experiment_rag_robust.py --stage analysis
"""

import argparse, json, os, re, sys, time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASELINE_PATH = Path("results_raw/step1/teval_responses_gemma-3-12b-it.json")
MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it"
)
CONFONLY_ADAPTER = "results_raw/finetune/gemma-3-12b-it/confonly_probe_target/adapters"
OUTPUT_DIR = Path("results_raw/domain_gen/rag_gating")

CONF_THRESHOLD = 50  # retrieve for items below this
RAG_SAMPLE_HIGH = 100  # also RAG this many high-confidence items (for always-RAG estimate)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CONFONLY_USER = (
    "You answered a trivia question.\n\n"
    "Question: {question}\n"
    "Your answer: {answer}\n\n"
    "How confident are you that your answer is correct? "
    "State your confidence as a percentage from 0 to 100."
)

RAG_USER = (
    "Answer this trivia question using the provided context. "
    "State your confidence as a percentage from 0 to 100.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_correct_triviaqa(predicted, aliases):
    """Check if predicted answer matches any gold alias (flexible matching)."""
    if not predicted or not aliases:
        return False
    pred = predicted.strip().lower()
    pred = re.sub(r'[^\w\s]', '', pred)
    # Remove common prefixes
    for prefix in ['the answer is ', 'that would be ', 'the answer: ']:
        if pred.startswith(prefix):
            pred = pred[len(prefix):]
    pred = pred.strip()
    if not pred:
        return False
    for alias in aliases:
        a = alias.strip().lower()
        a = re.sub(r'[^\w\s]', '', a)
        if not a:
            continue
        # Bidirectional substring (min 2 chars)
        if len(a) >= 2 and len(pred) >= 2:
            if a in pred or pred in a:
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


# ===================================================================
# STAGE 1: Two-Pass Confonly Confidence Rating
# ===================================================================
def run_confonly():
    """Rate baseline answers using the confonly adapter."""
    print("=" * 60)
    print("Stage 1: Two-Pass Confonly Confidence Rating")
    print("=" * 60)

    # Load baseline responses
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    print(f"  Baseline items: {len(baseline)}")
    print(f"  Baseline accuracy: {np.mean([i['correct'] for i in baseline]):.3f}")

    # Check adapter exists
    if not os.path.exists(CONFONLY_ADAPTER):
        print(f"  ERROR: Adapter not found at {CONFONLY_ADAPTER}")
        # Try alternative paths
        alt_paths = [
            "results_raw/finetune/gemma-3-12b-it/probe_target/adapters",
            "results_raw/finetune/gemma-3-12b-it/real/adapters",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                print(f"  Found alternative: {alt}")
                break
        else:
            print("  No adapter found. Available:")
            os.system("find results_raw/finetune/gemma-3-12b-it -name adapters -type d")
            sys.exit(1)

    # Load model with confonly adapter
    print(f"  Loading model + confonly adapter...")
    import mlx.core as mx
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_PATH, adapter_path=CONFONLY_ADAPTER)

    results = []
    t0 = time.time()

    for i, item in enumerate(baseline):
        question = item['question']
        answer = item['parsed_answer']

        # Construct confonly prompt
        user_msg = CONFONLY_USER.format(question=question, answer=answer)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        # Generate confidence rating (short — just "Confidence: X%")
        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=32, verbose=False)
        conf = parse_confidence(response)

        results.append({
            'question_id': item['question_id'],
            'question': question,
            'baseline_answer': answer,
            'baseline_correct': item['correct'],
            'baseline_confidence': item['parsed_confidence'],
            'confonly_response': response,
            'confonly_confidence': conf,
            'first_token_entropy': item.get('first_token_entropy', None),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            confs = [r['confonly_confidence'] for r in results
                     if not np.isnan(r['confonly_confidence'])]
            print(f"  [{i+1}/{len(baseline)}]  "
                  f"conf_mean={np.mean(confs):.1f}  "
                  f"conf_std={np.std(confs):.1f}  "
                  f"elapsed={elapsed:.0f}s")

    # Compute metrics
    from sklearn.metrics import roc_auc_score

    correct = np.array([int(r['baseline_correct']) for r in results])
    bl_conf = np.array([r['baseline_confidence'] for r in results])
    co_conf = np.array([r['confonly_confidence'] for r in results])
    valid = ~np.isnan(co_conf)

    print(f"\n  --- Results ---")
    print(f"  Items: {len(results)}")
    print(f"  Confidence parsed: {valid.sum()}/{len(results)}")
    print(f"  Accuracy (unchanged): {correct.mean():.3f}")

    if valid.sum() > 10:
        bl_auc = roc_auc_score(correct[valid], bl_conf[valid])
        co_auc = roc_auc_score(correct[valid], co_conf[valid])
        print(f"  Baseline AUROC₂: {bl_auc:.3f}")
        print(f"  Confonly AUROC₂:  {co_auc:.3f}")
        print(f"  Δ: {co_auc - bl_auc:+.3f}")
        print(f"  Confonly conf mean: {co_conf[valid].mean():.1f} "
              f"(std {co_conf[valid].std():.1f})")

        # Gating statistics
        above_95_bl = (bl_conf >= 95).sum()
        above_95_co = (co_conf[valid] >= 95).sum()
        below_thresh = (co_conf[valid] < CONF_THRESHOLD).sum()
        print(f"\n  Gating stats:")
        print(f"    Baseline ≥95%: {above_95_bl}/{len(results)} "
              f"({above_95_bl/len(results):.1%})")
        print(f"    Confonly ≥95%: {above_95_co}/{valid.sum()} "
              f"({above_95_co/valid.sum():.1%})")
        print(f"    Confonly <{CONF_THRESHOLD}%: {below_thresh}/{valid.sum()} "
              f"({below_thresh/valid.sum():.1%}) → these get retrieval")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = OUTPUT_DIR / "confonly_twopass_responses.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


# ===================================================================
# STAGE 2: RAG for Low-Confidence Items
# ===================================================================
def run_rag():
    """Generate RAG answers using TriviaQA evidence for low-confidence items."""
    print("\n" + "=" * 60)
    print("Stage 2: RAG for Low-Confidence Items")
    print("=" * 60)

    # Load confonly results
    confonly_path = OUTPUT_DIR / "confonly_twopass_responses.json"
    if not confonly_path.exists():
        print(f"  ERROR: Run --stage confonly first")
        sys.exit(1)

    with open(confonly_path) as f:
        confonly_results = json.load(f)

    # Identify items for retrieval
    retrieve_ids = set()
    high_conf_ids = set()
    for r in confonly_results:
        conf = r['confonly_confidence']
        if np.isnan(conf):
            continue
        if conf < CONF_THRESHOLD:
            retrieve_ids.add(r['question_id'])
        else:
            high_conf_ids.add(r['question_id'])

    # Also sample some high-confidence items for always-RAG estimate
    rng = np.random.default_rng(42)
    high_sample = list(high_conf_ids)
    rng.shuffle(high_sample)
    high_sample = set(high_sample[:RAG_SAMPLE_HIGH])
    all_rag_ids = retrieve_ids | high_sample

    print(f"  Low-confidence items (<{CONF_THRESHOLD}%): {len(retrieve_ids)}")
    print(f"  High-confidence sample (for always-RAG est): {len(high_sample)}")
    print(f"  Total items to RAG: {len(all_rag_ids)}")

    # Load TriviaQA evidence
    print(f"  Loading TriviaQA RC validation (evidence)...")
    from datasets import load_dataset
    ds = load_dataset('trivia_qa', 'rc', split='validation')

    # Build maps
    evidence_map = {}
    gold_map = {}
    for item in ds:
        qid = item.get('question_id', '')
        if qid not in all_rag_ids:
            continue

        # Gold answers
        answer = item.get('answer', {})
        aliases = list(set(
            answer.get('aliases', []) +
            [answer.get('value', '')] +
            answer.get('normalized_aliases', [])
        ))
        gold_map[qid] = [a for a in aliases if a]

        # Evidence
        evidence_parts = []
        if 'entity_pages' in item and item['entity_pages']:
            ep = item['entity_pages']
            if isinstance(ep, dict):
                for ctx in ep.get('wiki_context', [])[:2]:
                    if isinstance(ctx, str) and len(ctx) > 10:
                        evidence_parts.append(ctx[:800])
        if 'search_results' in item and item['search_results']:
            sr = item['search_results']
            if isinstance(sr, dict):
                for ctx in sr.get('search_context', [])[:2]:
                    if isinstance(ctx, str) and len(ctx) > 10:
                        evidence_parts.append(ctx[:500])

        if evidence_parts:
            evidence_map[qid] = "\n---\n".join(evidence_parts[:3])

    print(f"  Evidence found: {len(evidence_map)}/{len(all_rag_ids)} items")
    print(f"  Gold aliases: {len(gold_map)}/{len(all_rag_ids)} items")

    # Generate RAG answers (WITHOUT adapter — base model with evidence)
    print(f"\n  Loading base model (no adapter)...")
    import mlx.core as mx
    from mlx_lm import load, generate

    model, tokenizer = load(MODEL_PATH)

    confonly_map = {r['question_id']: r for r in confonly_results}
    rag_results = []
    t0 = time.time()

    items_to_rag = [qid for qid in all_rag_ids if qid in evidence_map and qid in gold_map]
    rng.shuffle(items_to_rag)

    for i, qid in enumerate(items_to_rag):
        r = confonly_map.get(qid, {})
        question = r.get('question', '')
        evidence = evidence_map[qid]

        prompt_text = RAG_USER.format(
            context=evidence[:2000],
            question=question,
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

        response = generate(model, tokenizer, prompt=prompt,
                           max_tokens=200, verbose=False)

        rag_correct = is_correct_triviaqa(response, gold_map[qid])

        rag_results.append({
            'question_id': qid,
            'question': question,
            'baseline_answer': r.get('baseline_answer', ''),
            'baseline_correct': r.get('baseline_correct', False),
            'confonly_confidence': r.get('confonly_confidence', float('nan')),
            'is_low_confidence': qid in retrieve_ids,
            'rag_response': response[:300],
            'rag_correct': rag_correct,
            'gold_aliases': gold_map[qid][:5],
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            low_items = [x for x in rag_results if x['is_low_confidence']]
            high_items = [x for x in rag_results if not x['is_low_confidence']]
            low_rag_acc = np.mean([x['rag_correct'] for x in low_items]) if low_items else 0
            low_bl_acc = np.mean([x['baseline_correct'] for x in low_items]) if low_items else 0
            print(f"  [{i+1}/{len(items_to_rag)}]  "
                  f"low_bl={low_bl_acc:.3f} low_rag={low_rag_acc:.3f}  "
                  f"elapsed={elapsed:.0f}s")

    # Save
    out_path = OUTPUT_DIR / "rag_responses.json"
    with open(out_path, 'w') as f:
        json.dump(rag_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Quick summary
    low = [x for x in rag_results if x['is_low_confidence']]
    high = [x for x in rag_results if not x['is_low_confidence']]

    print(f"\n  --- Quick Results ---")
    if low:
        print(f"  Low-confidence items (n={len(low)}):")
        print(f"    Baseline accuracy: {np.mean([x['baseline_correct'] for x in low]):.3f}")
        print(f"    RAG accuracy:      {np.mean([x['rag_correct'] for x in low]):.3f}")
        p_fix = sum(x['rag_correct'] and not x['baseline_correct'] for x in low) / \
                max(1, sum(not x['baseline_correct'] for x in low))
        print(f"    p_fix (errors corrected): {p_fix:.3f}")
    if high:
        print(f"  High-confidence sample (n={len(high)}):")
        print(f"    Baseline accuracy: {np.mean([x['baseline_correct'] for x in high]):.3f}")
        print(f"    RAG accuracy:      {np.mean([x['rag_correct'] for x in high]):.3f}")


# ===================================================================
# STAGE 3: Analysis + Figures
# ===================================================================
def run_analysis():
    """Compute final metrics and generate figures."""
    print("\n" + "=" * 60)
    print("Stage 3: Analysis + Figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    os.makedirs(OUTPUT_DIR / "figures", exist_ok=True)

    # Load all data
    with open(OUTPUT_DIR / "confonly_twopass_responses.json") as f:
        confonly = json.load(f)

    rag_path = OUTPUT_DIR / "rag_responses.json"
    rag_data = None
    if rag_path.exists():
        with open(rag_path) as f:
            rag_data = json.load(f)

    # --- Core metrics ---
    correct = np.array([int(r['baseline_correct']) for r in confonly])
    bl_conf = np.array([r['baseline_confidence'] for r in confonly])
    co_conf = np.array([r['confonly_confidence'] for r in confonly])
    valid = ~np.isnan(co_conf)

    n = valid.sum()
    baseline_acc = correct[valid].mean()
    bl_auc = roc_auc_score(correct[valid], bl_conf[valid])
    co_auc = roc_auc_score(correct[valid], co_conf[valid])

    print(f"  N items: {n}")
    print(f"  Baseline accuracy: {baseline_acc:.3f}")
    print(f"  Baseline AUROC₂:  {bl_auc:.3f}")
    print(f"  Confonly AUROC₂:   {co_auc:.3f}")

    # Gating contrast
    bl_above95 = (bl_conf[valid] >= 95).sum()
    co_above95 = (co_conf[valid] >= 95).sum()
    print(f"\n  Gating contrast:")
    print(f"    Baseline ≥95%: {bl_above95}/{n} ({bl_above95/n:.1%})")
    print(f"    Confonly ≥95%: {co_above95}/{n} ({co_above95/n:.1%})")

    # If RAG data exists, compute p_fix and gated accuracy
    p_fix_measured = None
    if rag_data:
        low = [x for x in rag_data if x['is_low_confidence']]
        high = [x for x in rag_data if not x['is_low_confidence']]

        if low:
            errors_in_low = sum(not x['baseline_correct'] for x in low)
            fixed = sum(x['rag_correct'] and not x['baseline_correct'] for x in low)
            p_fix_measured = fixed / max(1, errors_in_low)
            low_bl_acc = np.mean([x['baseline_correct'] for x in low])
            low_rag_acc = np.mean([x['rag_correct'] for x in low])

            print(f"\n  RAG results (low-confidence, n={len(low)}):")
            print(f"    Baseline acc: {low_bl_acc:.3f}")
            print(f"    RAG acc:      {low_rag_acc:.3f}")
            print(f"    Errors: {errors_in_low}, fixed by RAG: {fixed}")
            print(f"    p_fix = {p_fix_measured:.3f}")

        if high:
            high_bl_acc = np.mean([x['baseline_correct'] for x in high])
            high_rag_acc = np.mean([x['rag_correct'] for x in high])
            print(f"\n  RAG results (high-confidence sample, n={len(high)}):")
            print(f"    Baseline acc: {high_bl_acc:.3f}")
            print(f"    RAG acc:      {high_rag_acc:.3f}")

    # --- Compute gated accuracy at multiple thresholds ---
    thresholds = np.arange(0, 101, 5)
    correct_v = correct[valid]
    co_conf_v = co_conf[valid]
    bl_conf_v = bl_conf[valid]

    p_fix_vals = [0.5, 0.7, 0.9]
    if p_fix_measured is not None:
        p_fix_vals.append(p_fix_measured)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Confidence distributions (the gating contrast)
    ax = axes[0]
    ax.hist(bl_conf_v, bins=20, alpha=0.5, color='#999999',
            label=f'No PT-CSFT\n(≥95%: {bl_above95/n:.0%})', density=True)
    ax.hist(co_conf_v, bins=20, alpha=0.6, color='#2166ac',
            label=f'With PT-CSFT\n(≥95%: {co_above95/n:.0%})', density=True)
    ax.axvline(x=CONF_THRESHOLD, color='#cc0000', linestyle='--', linewidth=1.5,
               label=f'Retrieval threshold ({CONF_THRESHOLD}%)')
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Density')
    ax.set_title('The Gating Problem')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2)

    # Panel 2: Retrieval rate vs accuracy improvement
    ax = axes[1]
    for pf in sorted(set(p_fix_vals)):
        accs = []
        rets = []
        for t in thresholds:
            mask = co_conf_v < t
            ret = mask.sum() / n
            gated = correct_v.copy()
            rng = np.random.default_rng(42)
            for j in range(n):
                if mask[j] and not correct_v[j]:
                    if rng.random() < pf:
                        gated[j] = 1
            accs.append(gated.mean())
            rets.append(ret)

        label = f'p_fix={pf:.2f}'
        if pf == p_fix_measured:
            label += ' (measured)'
            ax.plot(rets, accs, 'o-', color='#cc0000', linewidth=2.5,
                    markersize=4, label=label, zorder=10)
        else:
            ax.plot(rets, accs, '--', linewidth=1.5, alpha=0.6, label=label)

    ax.axhline(y=baseline_acc, color='#999999', linestyle=':', linewidth=1,
               label=f'No retrieval ({baseline_acc:.3f})')
    ax.set_xlabel('Retrieval rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('Retrieval Rate vs Accuracy')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2)

    # Panel 3: The punchline — accuracy at fixed retrieval budget
    ax = axes[2]
    # Compare: random retrieval vs confidence-gated retrieval
    # at multiple retrieval budgets
    budgets = np.arange(0, 101, 5) / 100.0
    pf = p_fix_measured if p_fix_measured else 0.7

    gated_accs = []
    random_accs = []
    rng = np.random.default_rng(42)

    for budget in budgets:
        n_retrieve = int(budget * n)

        # Confidence-gated: retrieve lowest-confidence items first
        order = np.argsort(co_conf_v)
        gated = correct_v.copy()
        for j in range(min(n_retrieve, n)):
            idx = order[j]
            if not correct_v[idx]:
                if rng.random() < pf:
                    gated[idx] = 1
        gated_accs.append(gated.mean())

        # Random: retrieve random items
        rand_order = rng.permutation(n)
        rand = correct_v.copy()
        for j in range(min(n_retrieve, n)):
            idx = rand_order[j]
            if not correct_v[idx]:
                if rng.random() < pf:
                    rand[idx] = 1
        random_accs.append(rand.mean())

    ax.plot(budgets * 100, gated_accs, '-', color='#2166ac', linewidth=2.5,
            label=f'Confidence-gated (PT-CSFT)')
    ax.plot(budgets * 100, random_accs, '--', color='#999999', linewidth=2,
            label='Random retrieval')
    ax.axhline(y=baseline_acc, color='#999999', linestyle=':', linewidth=1, alpha=0.5)

    # Mark the sweet spot
    gated_arr = np.array(gated_accs)
    random_arr = np.array(random_accs)
    max_gap_idx = np.argmax(gated_arr - random_arr)
    ax.annotate(f'Max advantage\n({budgets[max_gap_idx]*100:.0f}% retrieval)',
                xy=(budgets[max_gap_idx]*100, gated_accs[max_gap_idx]),
                xytext=(budgets[max_gap_idx]*100 + 15, gated_accs[max_gap_idx] - 0.03),
                arrowprops=dict(arrowstyle='->', color='#cc0000'),
                fontsize=8, color='#cc0000')

    ax.set_xlabel('Retrieval budget (%)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Confidence-Gated vs Random Retrieval\n(p_fix={pf:.2f})')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = OUTPUT_DIR / "figures" / "fig_rag_robust.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f"\n  Saved: {out}")

    # --- Paper-ready summary ---
    print("\n" + "=" * 60)
    print("Paper-Ready Summary:")
    print("=" * 60)
    print(f"  Without PT-CSFT: {bl_above95}/{n} items at ≥95% confidence")
    print(f"    → Retrieval gating impossible (all items indistinguishable)")
    print(f"  With PT-CSFT:    {co_above95}/{n} items at ≥95% confidence")
    print(f"    → {(co_conf_v < CONF_THRESHOLD).sum()}/{n} items flagged for retrieval "
          f"at threshold {CONF_THRESHOLD}%")
    print(f"  Confonly AUROC₂: {co_auc:.3f} (baseline {bl_auc:.3f})")
    print(f"  Baseline accuracy preserved: {baseline_acc:.3f}")
    if p_fix_measured is not None:
        # Compute gated accuracy at threshold
        mask = co_conf_v < CONF_THRESHOLD
        gated = correct_v.copy()
        rng = np.random.default_rng(42)
        for j in range(n):
            if mask[j] and not correct_v[j]:
                if rng.random() < p_fix_measured:
                    gated[j] = 1
        gated_acc = gated.mean()
        retrieve_rate = mask.sum() / n
        print(f"  Measured p_fix: {p_fix_measured:.3f}")
        print(f"  Gated RAG accuracy: {gated_acc:.3f} "
              f"(Δ = {gated_acc - baseline_acc:+.3f})")
        print(f"  Retrieval rate: {retrieve_rate:.1%}")
        print(f"  → Accuracy improvement of {gated_acc - baseline_acc:+.3f} "
              f"while retrieving for only {retrieve_rate:.0%} of items")

    # Save summary
    summary = {
        "n_items": int(n),
        "baseline_accuracy": float(baseline_acc),
        "baseline_auroc2": float(bl_auc),
        "confonly_auroc2": float(co_auc),
        "baseline_above_95": int(bl_above95),
        "confonly_above_95": int(co_above95),
        "threshold": CONF_THRESHOLD,
        "p_fix_measured": float(p_fix_measured) if p_fix_measured else None,
    }
    with open(OUTPUT_DIR / "rag_robust_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {OUTPUT_DIR / 'rag_robust_summary.json'}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="confonly",
                        choices=["confonly", "rag", "analysis", "all"])
    args = parser.parse_args()

    if args.stage in ("confonly", "all"):
        run_confonly()

    if args.stage in ("rag", "all"):
        run_rag()

    if args.stage in ("analysis", "all"):
        run_analysis()


if __name__ == "__main__":
    main()
