"""
Step 1: Baseline Characterisation
==================================

Phase 0 v4, pre-reg v2. Runs greedy (temperature=0) decoding on T-eval
(1,000 TriviaQA) and M-eval (500 MMLU) using the unmodified baseline
Gemma 3 4B-it. For each item, records:

    - raw response
    - parsed answer
    - parsed verbal confidence (0-100)
    - correctness
    - last-layer hidden state at the last-answer-token position (for Step 1b
      and the post-SFT probe comparison)
    - multi-layer / multi-position hidden states (for E6 sensitivity)
    - answer-token logit entropy (for E5 single-pass baseline)

Also computes baseline metrics:
    - ceiling rate, accuracy, AUROC₂ with paired-bootstrap prep data
    - VRS screen
    - confidence histogram

Outputs:
    /home/claude work: cache everything locally, then copy final to outputs
    D:\\metacog\\results\\baseline\\
        teval_responses.json
        meval_responses.json
        baseline_metrics.json
    D:\\metacog\\data\\hidden_states\\
        baseline_teval.pt      # dict: {qid: {layer: {pos: tensor}}}
        baseline_meval.pt
        baseline_tcal.pt       # added here so Step 1b can fit probe on T-cal

Note on scope: this script also does a fast pass over T-cal to cache hidden
states only (no sampling, no correctness check — just greedy decode + caching
for the Step 1b probe fit). The actual T-cal sampling for confidence targets
happens in Step 2.

Hardware: AMD RX 7900 GRE, ROCm PyTorch. fp16. ~2-3 hours wall time for
T-eval + M-eval + T-cal hidden-state pass.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local
sys.path.insert(0, str(Path(__file__).parent))
from utils_phase0 import (
    ParsedResponse,
    auroc2,
    extract_answer_token_hidden_state,
    find_answer_token_positions,
    is_correct_mmlu,
    is_correct_triviaqa,
    parse_response,
    partition_triviaqa_pool,
    shannon_entropy_from_logits,
    vrs_screen,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-3-4b-it"
SEED = 42
N_TEVAL = 1000
N_MEVAL = 500
N_TCAL = 2000
MAX_NEW_TOKENS = 64

PROJECT_ROOT = Path(r"D:\metacog")
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"
HIDDEN_DIR = PROJECT_ROOT / "data" / "hidden_states"

# Probe configurations captured for E6 sensitivity analysis
# Layer selection is relative; will be resolved at load time based on model depth.
PROBE_LAYERS = {"first": None, "middle": None, "last": None}  # filled in main()
PROBE_POSITIONS = ["pre_answer_token", "last_answer_token"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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


def build_prompt(tokenizer, kind: str, item: dict) -> str:
    if kind == "triviaqa":
        user_msg = TRIVIAQA_PROMPT.format(question=item["question"])
    elif kind == "mmlu":
        q = item["question"]
        choices = item["choices"]  # list of 4 strings
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
# Data loading (stubs where disjointness requires external lists)
# ---------------------------------------------------------------------------

def load_triviaqa_splits(seed: int) -> tuple[list, list]:
    """Load T-eval (1,000) and T-cal (2,000) from TriviaQA rc.nocontext val.

    Disjoint from the saturation paper's 524 and from the Step 0 pre-check set.
    Disjointness is enforced programmatically via partition_triviaqa_pool(),
    which reproduces the saturation paper's exact seed-42 draw, excludes those
    indices, then shuffles and slices the remainder.
    """
    partition = partition_triviaqa_pool(seed=seed)
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    # Verify dataset size matches the expected constant
    assert len(ds) == 17_944, (
        f"TriviaQA rc.nocontext validation has {len(ds)} items, expected 17,944. "
        f"Dataset version may have changed."
    )

    def _to_items(indices: list[int]) -> list[dict]:
        items = []
        for i in indices:
            ex = ds[i]
            aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
            items.append({
                "ds_index": i,
                "question_id": ex["question_id"],
                "question": ex["question"],
                "aliases": [a for a in aliases if a],
            })
        return items

    teval = _to_items(partition["teval"])
    tcal = _to_items(partition["tcal"])

    print(f"[disjointness] saturation excluded: {len(partition['saturation'])}")
    print(f"[disjointness] T-eval: {len(teval)}, T-cal: {len(tcal)}, "
          f"Step 0 reserved: {len(partition['step0'])}")

    return teval, tcal


def load_mmlu_stratified(n_total: int, seed: int) -> list:
    """Load M-eval: n_total items stratified across Atlas 6 domains."""
    # Atlas 6 domains mapping — simplified here. Replace with the project's
    # canonical mapping when integrating.
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
        "law_business": ["business_ethics", "international_law", "jurisprudence",
                         "marketing", "management", "professional_accounting",
                         "professional_law"],
        "other": ["global_facts", "miscellaneous", "computer_security",
                  "machine_learning", "conceptual_physics", "logical_fallacies",
                  "elementary_mathematics"],
    }

    n_per_domain = n_total // len(ATLAS_DOMAINS)
    rng = np.random.default_rng(seed)
    items = []
    for domain, subjects in ATLAS_DOMAINS.items():
        pool = []
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test")
            except Exception:
                continue
            for ex in ds:
                pool.append({
                    "question_id": f"{subj}:{ex.get('idx', len(pool))}",
                    "subject": subj,
                    "domain": domain,
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "answer_idx": ex["answer"],
                    "answer_letter": "ABCD"[ex["answer"]],
                })
        if not pool:
            continue
        idxs = rng.choice(len(pool), size=min(n_per_domain, len(pool)), replace=False)
        items.extend([pool[i] for i in idxs])
    return items


# ---------------------------------------------------------------------------
# Generation with hidden-state capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode_with_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    probe_layers: dict,
    probe_positions: list,
    device: str,
) -> dict:
    """Greedy decode and capture:
        - generated tokens + text
        - last-layer logits at first-generated-token position (for E5 entropy)
        - hidden states at configured {layer × position} combinations, to be
          sliced once the answer span is known (caller does the slicing)

    Returns dict with full hidden-state stacks per layer so the caller can
    pick out the positions it needs after parsing the answer.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=max_new_tokens,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    generated_ids = out.sequences[0, prompt_len:].cpu().numpy()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # out.hidden_states is a tuple with one element per generated token;
    # each element is a tuple of (n_layers + 1) tensors of shape (1, 1, hidden_dim)
    # except the first which also covers the prompt.
    # We stack the per-token hidden states for each layer of interest.
    n_generated = len(out.hidden_states)
    hs_by_layer: dict[str, np.ndarray] = {}

    for layer_label, layer_idx in probe_layers.items():
        per_token = []
        for step in range(n_generated):
            # For step 0, hidden_states[0][layer_idx] has prompt-length sequence;
            # we want only the first newly generated token's hidden state.
            # For step > 0, hidden_states[step][layer_idx] is shape (1, 1, d).
            h = out.hidden_states[step][layer_idx]
            if step == 0:
                # Take the last position (first generated token's hidden state)
                per_token.append(h[0, -1, :].float().cpu().numpy())
            else:
                per_token.append(h[0, 0, :].float().cpu().numpy())
        hs_by_layer[layer_label] = np.stack(per_token, axis=0)

    # Logit entropy at first generated token (for E5 baseline).
    # out.scores[0] is shape (1, vocab_size).
    first_token_logits = out.scores[0][0].float().cpu().numpy()
    first_token_entropy = shannon_entropy_from_logits(first_token_logits)

    return {
        "generated_ids": generated_ids.tolist(),
        "generated_text": text,
        "hidden_states_by_layer": hs_by_layer,  # {layer_label: (n_gen, hidden_dim)}
        "first_token_entropy": first_token_entropy,
        "full_ids": out.sequences[0].cpu().numpy().tolist(),
        "prompt_len": prompt_len,
    }


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def run_eval_split(
    model,
    tokenizer,
    items: list,
    kind: str,
    probe_layers: dict,
    probe_positions: list,
    device: str,
    cache_hidden: bool = True,
    label: str = "",
) -> tuple[list, dict]:
    """Run the baseline model on a split; return per-item records and HS cache."""
    records = []
    hs_cache: dict = {}  # {qid: {layer_label: {position: np.ndarray}}}
    start = time.time()

    for i, item in enumerate(items):
        prompt = build_prompt(tokenizer, kind, item)
        gen = greedy_decode_with_hidden_states(
            model, tokenizer, prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            probe_layers=probe_layers,
            probe_positions=probe_positions,
            device=device,
        )

        # Parse answer + confidence
        parsed = parse_response(gen["generated_text"], prompt_format=kind)

        # Correctness
        if kind == "triviaqa":
            correct = is_correct_triviaqa(parsed.answer, item["aliases"])
        else:
            correct = is_correct_mmlu(parsed.answer, item["answer_letter"])

        # Locate answer-token span in generated tokens
        ans_start, ans_end = find_answer_token_positions(
            tokenizer, gen["full_ids"], gen["prompt_len"], parsed.answer
        )

        # Pull hidden states at the configured positions (if answer span found
        # and we're caching for this split)
        hs_item: dict = {}
        if cache_hidden and ans_start >= 0:
            for layer_label, stack in gen["hidden_states_by_layer"].items():
                hs_item[layer_label] = {}
                for pos in probe_positions:
                    try:
                        v = extract_answer_token_hidden_state(
                            stack, ans_start, ans_end, position=pos
                        )
                        hs_item[layer_label][pos] = v.astype(np.float32)
                    except (ValueError, IndexError):
                        pass

        records.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "raw_response": gen["generated_text"],
            "parsed_answer": parsed.answer,
            "parsed_confidence": parsed.confidence,
            "correct": bool(correct),
            "answer_start_token": int(ans_start),
            "answer_end_token": int(ans_end),
            "first_token_entropy": float(gen["first_token_entropy"]),
        })
        if cache_hidden and hs_item:
            hs_cache[item["question_id"]] = hs_item

        if (i + 1) % 50 == 0 or i == len(items) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate if rate > 0 else float("inf")
            print(f"[{label}] {i+1}/{len(items)}  "
                  f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

    return records, hs_cache


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_split_metrics(records: list, split_name: str) -> dict:
    confidences = np.array([r["parsed_confidence"] for r in records], dtype=float)
    correct = np.array([r["correct"] for r in records], dtype=int)
    accuracy = float(np.mean(correct))
    parse_rate = float(np.mean(~np.isnan(confidences)))

    # AUROC₂ only over items where confidence was parseable
    mask = ~np.isnan(confidences)
    if mask.sum() >= 2 and 0 < correct[mask].sum() < mask.sum():
        au = auroc2(confidences[mask], correct[mask])
    else:
        au = float("nan")

    # VRS screen
    vrs = vrs_screen(confidences[mask] if mask.any() else np.array([]),
                     correct[mask] if mask.any() else np.array([]))

    # Histogram (deciles)
    hist = {}
    if mask.any():
        bins = np.clip((confidences[mask] // 10).astype(int), 0, 10)
        for b in range(11):
            hist[str(b * 10)] = int(np.sum(bins == b))

    # Entropy-vs-correct AUROC₂ for E5 baseline
    entropies = np.array([r["first_token_entropy"] for r in records], dtype=float)
    # Higher entropy -> lower confidence, so use -entropy as confidence proxy
    if 0 < correct.sum() < len(correct):
        au_entropy = auroc2(-entropies, correct)
    else:
        au_entropy = float("nan")

    return {
        "split": split_name,
        "n": len(records),
        "accuracy": accuracy,
        "confidence_parse_rate": parse_rate,
        "auroc2_verbal": au,
        "auroc2_entropy_E5": au_entropy,
        "vrs": vrs,
        "confidence_histogram": hist,
        "mean_confidence": float(np.nanmean(confidences)) if mask.any() else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tcal-hidden", action="store_true",
                        help="Skip the T-cal hidden-state pass (useful if Step 1b "
                             "will run separately with its own T-cal pass).")
    args = parser.parse_args()

    # Env check
    print(f"[env] HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'UNSET')}")
    assert torch.cuda.is_available(), "ROCm not detected"
    print(f"[env] device: {torch.cuda.get_device_name(0)}")

    # Seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Paths
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[load] {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    device = next(model.parameters()).device
    print(f"[load] VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Resolve probe layer indices
    # Gemma 3 is a multimodal model; language config is nested under text_config.
    # num_hidden_layers counts transformer blocks; hidden_states tuple has
    # length n_layers + 1 (0 = embeddings, 1..n = block outputs).
    text_config = getattr(model.config, "text_config", model.config)
    n_layers = text_config.num_hidden_layers
    probe_layers = {
        "first":  1,
        "middle": n_layers // 2,
        "last":   n_layers,
    }
    print(f"[probe] layers: {probe_layers} (n_hidden_layers={n_layers})")

    # Load data
    if args.dry_run:
        teval, tcal = load_triviaqa_splits(SEED)
        teval, tcal = teval[:3], tcal[:3]
        meval = load_mmlu_stratified(n_total=N_MEVAL, seed=SEED)[:3]
    else:
        teval, tcal = load_triviaqa_splits(SEED)
        assert len(teval) == N_TEVAL, f"T-eval size mismatch: {len(teval)}"
        assert len(tcal) == N_TCAL, f"T-cal size mismatch: {len(tcal)}"
        meval = load_mmlu_stratified(n_total=N_MEVAL, seed=SEED)

    print(f"[data] T-eval: {len(teval)}  T-cal: {len(tcal)}  M-eval: {len(meval)}")

    # ------------------------------------------------------------------
    # T-eval
    # ------------------------------------------------------------------
    print("\n=== Step 1: T-eval baseline ===")
    teval_records, teval_hs = run_eval_split(
        model, tokenizer, teval, kind="triviaqa",
        probe_layers=probe_layers, probe_positions=PROBE_POSITIONS,
        device=device, cache_hidden=True, label="T-eval",
    )
    teval_metrics = compute_split_metrics(teval_records, "T-eval")

    with open(RESULTS_DIR / "teval_responses.json", "w") as f:
        json.dump(teval_records, f, indent=2)
    torch.save(teval_hs, HIDDEN_DIR / "baseline_teval.pt")
    print(f"[out] T-eval metrics: AUROC₂={teval_metrics['auroc2_verbal']:.3f} "
          f"ceiling_L={teval_metrics['vrs']['L']:.3f} "
          f"tier={teval_metrics['vrs']['tier']} "
          f"entropy AUROC₂ (E5 baseline)={teval_metrics['auroc2_entropy_E5']:.3f}")

    # ------------------------------------------------------------------
    # M-eval
    # ------------------------------------------------------------------
    print("\n=== Step 1: M-eval baseline ===")
    meval_records, meval_hs = run_eval_split(
        model, tokenizer, meval, kind="mmlu",
        probe_layers=probe_layers, probe_positions=PROBE_POSITIONS,
        device=device, cache_hidden=True, label="M-eval",
    )
    meval_metrics = compute_split_metrics(meval_records, "M-eval")

    with open(RESULTS_DIR / "meval_responses.json", "w") as f:
        json.dump(meval_records, f, indent=2)
    torch.save(meval_hs, HIDDEN_DIR / "baseline_meval.pt")
    print(f"[out] M-eval metrics: AUROC₂={meval_metrics['auroc2_verbal']:.3f} "
          f"tier={meval_metrics['vrs']['tier']}")

    # ------------------------------------------------------------------
    # T-cal hidden-state pass (for Step 1b probe fit)
    # ------------------------------------------------------------------
    if not args.skip_tcal_hidden:
        print("\n=== Step 1: T-cal hidden-state pass (for Step 1b) ===")
        tcal_records, tcal_hs = run_eval_split(
            model, tokenizer, tcal, kind="triviaqa",
            probe_layers=probe_layers, probe_positions=PROBE_POSITIONS,
            device=device, cache_hidden=True, label="T-cal",
        )
        # We also need per-item correctness for T-cal to fit the probe.
        # This greedy-single-sample correctness is NOT the modal_correct used
        # downstream (that requires 10-sample consistency from Step 2). Step 1b
        # uses this greedy correctness as a first-pass label, and Step 4's
        # post-SFT probe re-fit uses modal_correct from Step 2 for consistency
        # with the training-set labels. Document this distinction.
        with open(RESULTS_DIR / "tcal_greedy_responses.json", "w") as f:
            json.dump(tcal_records, f, indent=2)
        torch.save(tcal_hs, HIDDEN_DIR / "baseline_tcal.pt")

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    all_metrics = {
        "model_id": MODEL_ID,
        "seed": SEED,
        "probe_layers": probe_layers,
        "probe_positions": PROBE_POSITIONS,
        "teval": teval_metrics,
        "meval": meval_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    print("\n=== Step 1 complete ===")
    print(f"[out] metrics -> {RESULTS_DIR / 'baseline_metrics.json'}")
    print(f"[out] T-eval: AUROC₂={teval_metrics['auroc2_verbal']:.3f}, "
          f"accuracy={teval_metrics['accuracy']:.3f}, "
          f"VRS={teval_metrics['vrs']['tier']}")
    print(f"[out] M-eval: AUROC₂={meval_metrics['auroc2_verbal']:.3f}, "
          f"accuracy={meval_metrics['accuracy']:.3f}, "
          f"VRS={meval_metrics['vrs']['tier']}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
