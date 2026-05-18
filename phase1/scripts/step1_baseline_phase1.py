"""
Step 1: Baseline Characterisation (Phase 1 — MLX on M3 Ultra)
==============================================================

Pre-reg: Greedy generation on T-eval (1,000 TriviaQA) and M-eval (498 MMLU)
using the unmodified baseline model. For each item, records:

    - raw response
    - parsed answer
    - parsed verbal confidence (0-100)
    - correctness
    - hidden states at 3 layers × 2 positions (for Step 1b probe)
    - first-token logit entropy (for E5 baseline)

Also computes baseline metrics:
    - accuracy, AUROC₂, VRS screen, ceiling rate, confidence histogram

After T-eval and M-eval, runs a hidden-state-only pass on T-cal (2,000 items)
for the Step 1b probe fit.

Usage:
    python step1_baseline_phase1.py --model_path /path/to/gemma-3-12b-it
    python step1_baseline_phase1.py --model_path /path/to/gemma-3-27b-it
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load
from gen_helpers import generate_greedy, generate_sampled


# ---------------------------------------------------------------------------
# Constants (locked in pre-reg)
# ---------------------------------------------------------------------------
SEED = 42
N_TEVAL = 1000
N_TCAL = 2000
N_MEVAL = 498
MAX_TOKENS = 256

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

MMLU_PROMPT = (
    "Answer this multiple-choice question.\n"
    "{question}\n"
    "First, state your answer (A, B, C, or D).\n"
    "Then, state your confidence that your answer is correct as a number "
    "from 0 (pure guess) to 100 (certain).\n"
    "Reply in EXACTLY this format:\n"
    "Answer: [letter]\n"
    "Confidence: [number]"
)

# Atlas 6 domains for MMLU stratification
ATLAS_DOMAINS = {
    "stem": ["abstract_algebra", "college_physics", "college_chemistry",
             "college_biology", "college_computer_science",
             "college_mathematics", "high_school_mathematics",
             "high_school_physics", "high_school_chemistry",
             "high_school_biology", "high_school_computer_science",
             "astronomy", "electrical_engineering"],
    "humanities": ["world_religions", "philosophy", "prehistory",
                   "high_school_world_history", "high_school_us_history",
                   "high_school_european_history", "formal_logic",
                   "moral_disputes", "moral_scenarios"],
    "social_science": ["sociology", "public_relations", "security_studies",
                       "us_foreign_policy", "human_sexuality",
                       "high_school_geography", "high_school_government_and_politics",
                       "high_school_psychology", "high_school_microeconomics",
                       "high_school_macroeconomics", "econometrics"],
    "medical": ["anatomy", "clinical_knowledge", "college_medicine",
                "human_aging", "medical_genetics", "nutrition",
                "professional_medicine", "virology"],
    "law": ["international_law", "jurisprudence", "professional_law"],
    "business": ["business_ethics", "management", "marketing",
                 "professional_accounting", "global_facts",
                 "miscellaneous", "conceptual_physics", "machine_learning",
                 "computer_security", "logical_fallacies"],
}


# ---------------------------------------------------------------------------
# Response parsing (from utils_phase0)
# ---------------------------------------------------------------------------
_CONFIDENCE_PATTERNS = [
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
    re.compile(r"\b(\d{1,3})\b\s*$"),
]

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _ARTICLE_RE.sub("", s)
    return s


def is_correct_triviaqa(pred: str, aliases: list) -> bool:
    p = _normalise(pred)
    if not p:
        return False
    for a in aliases:
        an = _normalise(a)
        if not an:
            continue
        if an in p or p in an:
            return True
    return False


def is_correct_mmlu(pred: str, gold: str) -> bool:
    return pred.strip().upper() == gold.strip().upper()


def parse_response(raw: str, prompt_format: str = "triviaqa") -> dict:
    """Parse answer and confidence from generated text."""
    text = raw.strip()
    answer = ""
    confidence = float("nan")

    if prompt_format == "triviaqa":
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            first = lines[0]
            m = re.match(r"^(?:answer\s*:?\s*)(.*)", first, flags=re.IGNORECASE)
            answer = m.group(1).strip() if m else first
            answer = re.sub(
                r"\s*[,;]?\s*confidence\s*:?.*$", "", answer, flags=re.IGNORECASE
            ).strip()
    elif prompt_format == "mmlu":
        m = re.search(r"answer\s*:?\s*([A-Da-d])", text, flags=re.IGNORECASE)
        if m:
            answer = m.group(1).upper()

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


# ---------------------------------------------------------------------------
# VRS screen
# ---------------------------------------------------------------------------
def vrs_screen(confidence: np.ndarray, correct: np.ndarray) -> dict:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    n = len(c)
    if n == 0:
        return {"tier": "undefined", "n": 0}

    L = float(np.mean(c >= 95.0))
    Fp = float(np.mean(c <= 5.0))
    bins = np.clip((c // 10).astype(int), 0, 10)
    distinct = len(np.unique(bins))
    RBS = 1 - distinct / 11
    unique, counts = np.unique(bins, return_counts=True)
    TRIN = float(counts.max() / counts.sum())

    if c.std() == 0 or y.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(c, y)[0, 1])

    if L >= 0.70 or TRIN >= 0.80 or abs(r) < 0.05:
        tier = "Invalid"
    elif L >= 0.40 or TRIN >= 0.60:
        tier = "Indeterminate"
    else:
        tier = "Valid"

    return {"L": L, "Fp": Fp, "RBS": RBS, "TRIN": TRIN, "r": r,
            "tier": tier, "n": n}


# ---------------------------------------------------------------------------
# AUROC₂
# ---------------------------------------------------------------------------
def auroc2(confidence: np.ndarray, correct: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    if mask.sum() < 2:
        return float("nan")
    if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))


# ---------------------------------------------------------------------------
# Data loading — identical partitions to Phase 0
# ---------------------------------------------------------------------------
def load_triviaqa_splits(seed: int):
    """Load T-eval (1,000), T-cal (2,000) from TriviaQA rc.nocontext val."""
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    teval_idx = indices[0:1000]
    tcal_idx = indices[1000:3000]

    def _to_items(idxs):
        items = []
        for i in idxs:
            ex = ds[i]
            aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
            items.append({
                "question_id": ex["question_id"],
                "question": ex["question"],
                "aliases": [a for a in aliases if a],
            })
        return items

    return _to_items(teval_idx), _to_items(tcal_idx)


def load_mmlu_stratified(n_total: int, seed: int) -> list:
    """Load M-eval: ~n_total items stratified across Atlas 6 domains."""
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    n_per_domain = n_total // len(ATLAS_DOMAINS)

    # Build subject -> domain mapping
    subj_to_domain = {}
    for domain, subjects in ATLAS_DOMAINS.items():
        for s in subjects:
            subj_to_domain[s] = domain

    ds = load_dataset("cais/mmlu", "all", split="test")

    # Pool by domain
    pools = {d: [] for d in ATLAS_DOMAINS}
    for i, ex in enumerate(ds):
        subj = ex["subject"]
        if subj in subj_to_domain:
            pools[subj_to_domain[subj]].append(i)

    items = []
    for domain, pool in pools.items():
        if not pool:
            continue
        chosen = rng.choice(len(pool), size=min(n_per_domain, len(pool)),
                            replace=False)
        for idx in chosen:
            ex = ds[pool[idx]]
            answer_letter = ["A", "B", "C", "D"][ex["answer"]]
            items.append({
                "question_id": f"mmlu_{pool[idx]}",
                "question": ex["question"],
                "choices": ex["choices"],
                "subject": ex["subject"],
                "domain": subj_to_domain[ex["subject"]],
                "answer_letter": answer_letter,
            })
    return items


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, kind: str, item: dict) -> str:
    if kind == "triviaqa":
        user_msg = TRIVIAQA_PROMPT.format(question=item["question"])
    elif kind == "mmlu":
        q = item["question"]
        choices = item["choices"]
        q_full = (
            f"{q}\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}"
        )
        user_msg = MMLU_PROMPT.format(question=q_full)
    else:
        raise ValueError(kind)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Hidden-state extraction via MLX
# ---------------------------------------------------------------------------
def get_hidden_states_and_logits(model, tokenizer, prompt: str,
                                 layer_indices: dict) -> dict:
    """Run a forward pass and extract hidden states at specified layers,
    plus first-token logits for entropy computation.

    layer_indices: {"first": 0, "middle": 24, "last": 47} (for 48-layer model)

    Returns:
        {
            "generated_text": str,
            "hidden_states": {layer_label: (seq_len, hidden_dim) numpy},
            "first_token_logits": (vocab_size,) numpy,
            "token_ids": list[int],
            "prompt_len": int,
        }
    """
    # First, do the generation to get the output text
    generated_text = generate_greedy(model, tokenizer, prompt)

    # Now, do a forward pass on the full sequence (prompt + generated)
    # to extract hidden states
    full_text = prompt + generated_text
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    x = mx.array([tokens])

    # Get hidden states by hooking into the transformer layers
    # We need to run through the model's layers manually
    if hasattr(model, "language_model"):
        lm = model.language_model
        scale_embeddings = True
    else:
        lm = model
        scale_embeddings = False

    # Get embeddings
    if hasattr(lm.model, 'embed_tokens'):
        h = lm.model.embed_tokens(x)
    else:
        h = lm.model.embed_tokens(x)

    # Apply RMS norm scaling for Gemma only
    if scale_embeddings:
        hidden_size = h.shape[-1]
        h = h * (hidden_size ** 0.5)

    hidden_states = {}
    cache = None

    for i, layer in enumerate(lm.model.layers):
        h = layer(h, cache=cache)
        # Check if this layer index matches any we want
        for label, idx in layer_indices.items():
            if i == idx:
                hidden_states[label] = np.array(h[0].astype(mx.float32))

    # Get logits from the final hidden state
    final_h = lm.model.norm(h) if hasattr(lm.model, 'norm') else h
    if lm.tie_word_embeddings if hasattr(lm, 'tie_word_embeddings') else True:
        logits = lm.model.embed_tokens.as_linear(final_h)
    else:
        logits = lm.lm_head(final_h)

    # First generated token logits
    first_gen_logits = np.array(logits[0, prompt_len - 1].astype(mx.float32))

    return {
        "generated_text": generated_text,
        "hidden_states": hidden_states,
        "first_token_logits": first_gen_logits,
        "token_ids": tokens,
        "prompt_len": prompt_len,
    }


def shannon_entropy_from_logits(logits: np.ndarray) -> float:
    x = logits - logits.max()
    p = np.exp(x)
    p = p / p.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def find_answer_token_positions(tokenizer, token_ids, prompt_len, answer_text):
    """Find the answer span in generated tokens."""
    gen_ids = list(token_ids[prompt_len:])
    if not answer_text:
        return (-1, -1)

    candidates = [answer_text, " " + answer_text]
    for cand in candidates:
        ans_ids = tokenizer.encode(cand)
        # Remove BOS token if present
        if ans_ids and ans_ids[0] == 2:  # Gemma BOS
            ans_ids = ans_ids[1:]
        if not ans_ids:
            continue
        n = len(ans_ids)
        for i in range(len(gen_ids) - n + 1):
            if gen_ids[i:i + n] == ans_ids:
                return (i, i + n)
    return (-1, -1)


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------
def run_eval_split(model, tokenizer, items, kind, layer_indices,
                   cache_hidden=True, label=""):
    """Evaluate a split with greedy generation + hidden state extraction."""
    records = []
    hs_cache = {}
    start = time.time()

    for i, item in enumerate(items):
        prompt = build_prompt(tokenizer, kind, item)

        try:
            result = get_hidden_states_and_logits(
                model, tokenizer, prompt, layer_indices
            )

            parsed = parse_response(result["generated_text"], prompt_format=kind)

            if kind == "triviaqa":
                correct = is_correct_triviaqa(parsed["answer"], item["aliases"])
            else:
                correct = is_correct_mmlu(parsed["answer"], item["answer_letter"])

            entropy = shannon_entropy_from_logits(result["first_token_logits"])

            # Find answer token positions for hidden state slicing
            ans_start, ans_end = find_answer_token_positions(
                tokenizer, result["token_ids"], result["prompt_len"],
                parsed["answer"]
            )

            # Cache hidden states at answer positions
            hs_item = {}
            if cache_hidden and ans_start >= 0:
                for layer_label, hs in result["hidden_states"].items():
                    hs_item[layer_label] = {}
                    # pre_answer_token
                    pre_idx = max(0, ans_start - 1)
                    hs_item[layer_label]["pre_answer_token"] = \
                        hs[result["prompt_len"] + pre_idx].astype(np.float32)
                    # last_answer_token
                    last_idx = ans_end - 1
                    if result["prompt_len"] + last_idx < hs.shape[0]:
                        hs_item[layer_label]["last_answer_token"] = \
                            hs[result["prompt_len"] + last_idx].astype(np.float32)

            qid = item["question_id"]
            records.append({
                "question_id": qid,
                "question": item["question"],
                "raw_response": result["generated_text"],
                "parsed_answer": parsed["answer"],
                "parsed_confidence": parsed["confidence"]
                    if not np.isnan(parsed["confidence"]) else None,
                "correct": bool(correct),
                "answer_start_token": int(ans_start),
                "answer_end_token": int(ans_end),
                "first_token_entropy": entropy,
            })
            if cache_hidden and hs_item:
                hs_cache[qid] = hs_item

        except Exception as e:
            qid = item["question_id"]
            records.append({
                "question_id": qid,
                "question": item["question"],
                "raw_response": f"ERROR: {str(e)}",
                "parsed_answer": "",
                "parsed_confidence": None,
                "correct": False,
                "answer_start_token": -1,
                "answer_end_token": -1,
                "first_token_entropy": float("nan"),
            })

        if (i + 1) % 50 == 0 or i == len(items) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate if rate > 0 else float("inf")
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"[{label}] {i+1}/{len(items)}  acc={acc:.3f}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    return records, hs_cache


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------
def compute_split_metrics(records: list, split_name: str) -> dict:
    confidences = np.array(
        [r["parsed_confidence"] if r["parsed_confidence"] is not None
         else float("nan") for r in records],
        dtype=float
    )
    correct = np.array([r["correct"] for r in records], dtype=int)
    accuracy = float(np.mean(correct))
    parse_rate = float(np.mean(~np.isnan(confidences)))

    mask = ~np.isnan(confidences)
    if mask.sum() >= 2 and 0 < correct[mask].sum() < mask.sum():
        au = auroc2(confidences[mask], correct[mask])
    else:
        au = float("nan")

    vrs = vrs_screen(
        confidences[mask] if mask.any() else np.array([]),
        correct[mask] if mask.any() else np.array([])
    )

    # Histogram (deciles)
    hist = {}
    if mask.any():
        bins = np.clip((confidences[mask] // 10).astype(int), 0, 10)
        for b in range(11):
            hist[str(b * 10)] = int(np.sum(bins == b))

    # Entropy stats
    entropies = np.array([r["first_token_entropy"] for r in records], dtype=float)
    ent_mask = ~np.isnan(entropies)
    if ent_mask.sum() >= 2 and 0 < correct[ent_mask].sum() < ent_mask.sum():
        # Entropy predicts correctness inversely (higher entropy → less sure)
        auroc2_entropy = auroc2(-entropies[ent_mask], correct[ent_mask])
    else:
        auroc2_entropy = float("nan")

    return {
        "split": split_name,
        "n": len(records),
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "auroc2_verbal": au,
        "auroc2_entropy": auroc2_entropy,
        "vrs": vrs,
        "confidence_histogram": hist,
        "entropy_mean": float(np.nanmean(entropies)),
        "entropy_std": float(np.nanstd(entropies)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 1 Step 1: Baseline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/step1")
    parser.add_argument("--skip_tcal_hidden", action="store_true",
                        help="Skip T-cal hidden-state pass (faster, no probe)")
    args = parser.parse_args()

    model_name = Path(args.model_path).name
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Phase 1 Step 1: Baseline Characterisation")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load(args.model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # Determine layer indices for probe
    if hasattr(model, "language_model"):
        n_layers = len(model.language_model.model.layers)
    else:
        n_layers = len(model.model.layers)
    layer_indices = {
        "first": 0,
        "middle": n_layers // 2,
        "last": n_layers - 1,
    }
    print(f"  Model has {n_layers} layers")
    print(f"  Probe layers: {layer_indices}\n")

    # Load data
    print("Loading data splits...")
    teval, tcal = load_triviaqa_splits(SEED)
    meval = load_mmlu_stratified(N_MEVAL, SEED)
    print(f"  T-eval: {len(teval)} items")
    print(f"  T-cal: {len(tcal)} items")
    print(f"  M-eval: {len(meval)} items\n")

    # ------------------------------------------------------------------
    # T-eval
    # ------------------------------------------------------------------
    print("=== T-eval (TriviaQA, 1000 items) ===")
    teval_records, teval_hs = run_eval_split(
        model, tokenizer, teval, kind="triviaqa",
        layer_indices=layer_indices, cache_hidden=True, label="T-eval"
    )
    teval_metrics = compute_split_metrics(teval_records, "T-eval")

    with open(os.path.join(args.output_dir,
              f"teval_responses_{model_name}.json"), "w") as f:
        json.dump(teval_records, f, indent=2)

    # Save hidden states as numpy archive
    hs_file = os.path.join(args.output_dir,
                           f"hidden_states_teval_{model_name}.npz")
    hs_flat = {}
    for qid, layers in teval_hs.items():
        for layer_label, positions in layers.items():
            for pos, vec in positions.items():
                hs_flat[f"{qid}__{layer_label}__{pos}"] = vec
    np.savez_compressed(hs_file, **hs_flat)

    print(f"\n  T-eval: accuracy={teval_metrics['accuracy']:.3f}  "
          f"AUROC₂_verbal={teval_metrics['auroc2_verbal']:.3f}  "
          f"AUROC₂_entropy={teval_metrics['auroc2_entropy']:.3f}  "
          f"VRS={teval_metrics['vrs']['tier']}")
    print(f"  Confidence parse rate: {teval_metrics['parse_rate']:.3f}")
    print(f"  Hidden states cached: {len(teval_hs)} items\n")

    # ------------------------------------------------------------------
    # M-eval
    # ------------------------------------------------------------------
    print("=== M-eval (MMLU, ~498 items) ===")
    meval_records, meval_hs = run_eval_split(
        model, tokenizer, meval, kind="mmlu",
        layer_indices=layer_indices, cache_hidden=True, label="M-eval"
    )
    meval_metrics = compute_split_metrics(meval_records, "M-eval")

    with open(os.path.join(args.output_dir,
              f"meval_responses_{model_name}.json"), "w") as f:
        json.dump(meval_records, f, indent=2)

    hs_file = os.path.join(args.output_dir,
                           f"hidden_states_meval_{model_name}.npz")
    hs_flat = {}
    for qid, layers in meval_hs.items():
        for layer_label, positions in layers.items():
            for pos, vec in positions.items():
                hs_flat[f"{qid}__{layer_label}__{pos}"] = vec
    np.savez_compressed(hs_file, **hs_flat)

    print(f"\n  M-eval: accuracy={meval_metrics['accuracy']:.3f}  "
          f"AUROC₂_verbal={meval_metrics['auroc2_verbal']:.3f}  "
          f"AUROC₂_entropy={meval_metrics['auroc2_entropy']:.3f}  "
          f"VRS={meval_metrics['vrs']['tier']}")
    print(f"  Confidence parse rate: {meval_metrics['parse_rate']:.3f}")
    print(f"  Hidden states cached: {len(meval_hs)} items\n")

    # ------------------------------------------------------------------
    # T-cal hidden-state pass (for Step 1b probe)
    # ------------------------------------------------------------------
    if not args.skip_tcal_hidden:
        print("=== T-cal hidden-state pass (for Step 1b probe) ===")
        tcal_records, tcal_hs = run_eval_split(
            model, tokenizer, tcal, kind="triviaqa",
            layer_indices=layer_indices, cache_hidden=True, label="T-cal"
        )

        with open(os.path.join(args.output_dir,
                  f"tcal_greedy_responses_{model_name}.json"), "w") as f:
            json.dump(tcal_records, f, indent=2)

        hs_file = os.path.join(args.output_dir,
                               f"hidden_states_tcal_{model_name}.npz")
        hs_flat = {}
        for qid, layers in tcal_hs.items():
            for layer_label, positions in layers.items():
                for pos, vec in positions.items():
                    hs_flat[f"{qid}__{layer_label}__{pos}"] = vec
        np.savez_compressed(hs_file, **hs_flat)

        tcal_acc = sum(r["correct"] for r in tcal_records) / len(tcal_records)
        print(f"\n  T-cal: accuracy={tcal_acc:.3f}  "
              f"Hidden states cached: {len(tcal_hs)} items\n")

    # ------------------------------------------------------------------
    # Save aggregate metrics
    # ------------------------------------------------------------------
    all_metrics = {
        "model": model_name,
        "model_path": args.model_path,
        "seed": SEED,
        "n_layers": n_layers,
        "layer_indices": layer_indices,
        "teval": teval_metrics,
        "meval": meval_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_file = os.path.join(args.output_dir,
                                f"baseline_metrics_{model_name}.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 1 SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  T-eval:")
    print(f"    Accuracy:       {teval_metrics['accuracy']:.3f}")
    print(f"    AUROC₂ verbal:  {teval_metrics['auroc2_verbal']:.3f}")
    print(f"    AUROC₂ entropy: {teval_metrics['auroc2_entropy']:.3f}")
    print(f"    VRS tier:       {teval_metrics['vrs']['tier']}")
    print(f"    Parse rate:     {teval_metrics['parse_rate']:.3f}")
    print(f"  M-eval:")
    print(f"    Accuracy:       {meval_metrics['accuracy']:.3f}")
    print(f"    AUROC₂ verbal:  {meval_metrics['auroc2_verbal']:.3f}")
    print(f"    AUROC₂ entropy: {meval_metrics['auroc2_entropy']:.3f}")
    print(f"    VRS tier:       {meval_metrics['vrs']['tier']}")
    print(f"    Parse rate:     {meval_metrics['parse_rate']:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
