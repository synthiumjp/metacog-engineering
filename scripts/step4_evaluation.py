"""
Step 4: Post-SFT Evaluation & Decision-Rule Application
=========================================================

Phase 0 v4, pre-reg v2. Evaluates the fine-tuned (real-target) and
shuffled-target models on T-eval, M-eval, and the conflict set. Computes
all paired-bootstrap CIs, runs within-bin analysis (H3), conflict-set
evaluation (H5), probe re-fit (E2/E6), entropy comparison (E5), AURC
(E8), and applies the pre-registered decision tree.

Inputs required (from earlier steps):
    - Step 1: results/baseline/teval_responses.json
    - Step 1: results/baseline/meval_responses.json
    - Step 1: results/baseline/baseline_metrics.json
    - Step 1b: results/probe/probe_metrics.json
    - Step 1b: results/probe/probe_scores_teval.json
    - Step 2: data/step2_teval_difficulty.json
    - Step 2: data/step2_conflict_set.json
    - Step 3: results/finetune/lora_real/
    - Step 3b: results/finetune/lora_shuffled/

Outputs:
    D:\\metacog\\results\\evaluation\\step4_results.json    (all metrics)
    D:\\metacog\\results\\evaluation\\decision.json          (decision-tree)

Runtime: ~1 hour (T-eval × 2 models + M-eval × 1 + conflict × 1 +
         hidden-state passes + probe re-fits).
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from utils_phase0 import (
    ParsedResponse,
    auroc2,
    extract_answer_token_hidden_state,
    find_answer_token_positions,
    is_correct_mmlu,
    is_correct_triviaqa,
    paired_bootstrap_auroc2_delta,
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
BOOTSTRAP_N = 10_000
MAX_NEW_TOKENS = 64

PROJECT_ROOT = Path(r"D:\metacog")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = RESULTS_DIR / "evaluation"
HIDDEN_DIR = DATA_DIR / "hidden_states"

LORA_REAL_DIR = RESULTS_DIR / "finetune" / "lora_real"
LORA_SHUFFLED_DIR = RESULTS_DIR / "finetune" / "lora_shuffled"

# Pre-registered thresholds
H1_DELTA_FLOOR = 0.03
ACCURACY_DROP_THRESHOLD = 0.05
CEILING_CLEAR_PROCEED = 0.70

PROBE_POSITIONS = ["pre_answer_token", "last_answer_token"]
PROBE_CS = [0.001, 0.01, 0.1, 1, 10, 100]

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

MMLU_PROMPT = (
    "Answer the following multiple-choice question. "
    "Pick exactly one letter (A, B, C, or D). "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "{question}\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_with_lora(adapter_dir: Path):
    """Load base model + LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


def build_prompt_triviaqa(tokenizer, question: str) -> str:
    user_msg = TRIVIAQA_PROMPT.format(question=question)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_prompt_mmlu(tokenizer, item: dict) -> str:
    user_msg = MMLU_PROMPT.format(**item)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def evaluate_split(model, tokenizer, items, kind, label="eval",
                   cache_hidden=False, probe_layers=None):
    """Run greedy generation on a split. Returns records and optional HS."""
    records = []
    all_hs = {} if cache_hidden else None

    for idx, item in enumerate(items):
        if kind == "triviaqa":
            prompt = build_prompt_triviaqa(tokenizer, item["question"])
        else:
            prompt = build_prompt_mmlu(tokenizer, item)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        if cache_hidden:
            gen_kwargs["output_hidden_states"] = True

        out = model.generate(**gen_kwargs)
        gen_ids = out.sequences[0, prompt_len:].cpu().numpy()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        parsed = parse_response(text, prompt_format=kind)

        if kind == "triviaqa":
            correct = is_correct_triviaqa(parsed.answer, item["aliases"])
        else:
            correct = is_correct_mmlu(parsed.answer, item["gold"])

        record = {
            "question_id": item.get("question_id", str(idx)),
            "response": text,
            "answer": parsed.answer,
            "confidence": parsed.confidence,
            "correct": correct,
        }
        records.append(record)

        # Cache hidden states if needed
        if cache_hidden and probe_layers and hasattr(out, "hidden_states"):
            for layer_name, layer_idx in probe_layers.items():
                for pos_name in PROBE_POSITIONS:
                    key = f"{layer_name}_{pos_name}"
                    if key not in all_hs:
                        all_hs[key] = []
                    hs = extract_answer_token_hidden_state(
                        out.hidden_states, layer_idx, pos_name,
                        prompt_len, gen_ids,
                    )
                    if hs is not None:
                        all_hs[key].append(hs.cpu().numpy())
                    else:
                        all_hs[key].append(None)

        if (idx + 1) % 50 == 0 or idx == len(items) - 1:
            elapsed = time.time() - _split_start
            rate = (idx + 1) / elapsed
            eta = (len(items) - idx - 1) / rate if rate > 0 else 0
            print(f"[{label}] {idx+1}/{len(items)}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    return records, all_hs


def compute_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (E8).

    Sort items by descending confidence. At each coverage level k/n,
    risk = 1 - accuracy on top-k items. AURC = mean risk across all
    coverage levels.
    """
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    n = len(c)
    if n == 0:
        return float("nan")

    order = np.argsort(-c)  # descending confidence
    y_sorted = y[order]
    cum_correct = np.cumsum(y_sorted)
    coverage = np.arange(1, n + 1)
    risk = 1.0 - cum_correct / coverage
    return float(np.mean(risk))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _split_start

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate on 5 items per split.")
    args = parser.parse_args()

    print(f"[env] HSA_OVERRIDE_GFX_VERSION="
          f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'UNSET')}")
    assert torch.cuda.is_available(), "ROCm not detected"
    print(f"[env] device: {torch.cuda.get_device_name(0)}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load baseline data
    # ------------------------------------------------------------------
    print("\n=== Loading baseline data ===")

    with open(RESULTS_DIR / "baseline" / "teval_responses.json") as f:
        base_teval = json.load(f)
    with open(RESULTS_DIR / "baseline" / "meval_responses.json") as f:
        base_meval = json.load(f)
    with open(RESULTS_DIR / "baseline" / "baseline_metrics.json") as f:
        base_metrics = json.load(f)
    with open(RESULTS_DIR / "probe" / "probe_scores_teval.json") as f:
        base_probe_scores = json.load(f)

    # Difficulty bins for T-eval (from Step 2)
    with open(DATA_DIR / "step2_teval_difficulty.json") as f:
        teval_difficulty = json.load(f)

    # Conflict set (from Step 2)
    with open(DATA_DIR / "step2_conflict_set.json") as f:
        conflict_items = json.load(f)

    # Baseline arrays
    base_teval_conf = np.array([r["confidence"] for r in base_teval])
    base_teval_corr = np.array([int(r["correct"]) for r in base_teval])
    base_meval_conf = np.array([r["confidence"] for r in base_meval])
    base_meval_corr = np.array([int(r["correct"]) for r in base_meval])

    # Entropy AUROC2 from Step 1
    base_entropy_auroc2 = base_metrics["teval"].get("auroc2_entropy", None)

    print(f"[base] T-eval: n={len(base_teval)}, "
          f"AUROC2={base_metrics['teval']['auroc2_verbal']:.3f}")
    print(f"[base] M-eval: n={len(base_meval)}, "
          f"AUROC2={base_metrics['meval']['auroc2_verbal']:.3f}")
    print(f"[base] Conflict set: n={len(conflict_items)}")

    # ------------------------------------------------------------------
    # Load T-eval and M-eval item data for re-evaluation
    # ------------------------------------------------------------------
    partition = partition_triviaqa_pool(seed=SEED)
    from datasets import load_dataset
    ds_tqa = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    teval_items = []
    for i in partition["teval"]:
        ex = ds_tqa[i]
        aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        teval_items.append({
            "ds_index": i,
            "question_id": ex["question_id"],
            "question": ex["question"],
            "aliases": [a for a in aliases if a],
        })

    # Load MMLU items (same as Step 1)
    from step1_baseline import load_mmlu_meval
    meval_items = load_mmlu_meval()

    if args.dry_run:
        teval_items = teval_items[:5]
        meval_items = meval_items[:5]
        conflict_items = conflict_items[:5]

    # Probe layer indices
    text_config_layers = 34  # Gemma 3 4B
    probe_layers = {
        "first": 1,
        "middle": text_config_layers // 2,
        "last": text_config_layers,
    }

    # ==================================================================
    # A. Real-target model evaluation
    # ==================================================================
    print("\n=== Step 4A: Real-target model on T-eval ===")
    ft_model, tokenizer = load_model_with_lora(LORA_REAL_DIR)
    print(f"[load] VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    _split_start = time.time()
    ft_teval_records, ft_teval_hs = evaluate_split(
        ft_model, tokenizer, teval_items, "triviaqa",
        label="ft-T-eval", cache_hidden=True, probe_layers=probe_layers,
    )

    print("\n=== Step 4A: Real-target model on M-eval ===")
    _split_start = time.time()
    ft_meval_records, _ = evaluate_split(
        ft_model, tokenizer, meval_items, "mmlu",
        label="ft-M-eval",
    )

    print("\n=== Step 4A: Real-target model on conflict set ===")
    _split_start = time.time()
    ft_conflict_records, _ = evaluate_split(
        ft_model, tokenizer, conflict_items, "triviaqa",
        label="ft-conflict",
    )

    del ft_model
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # B. Shuffled-target model evaluation
    # ==================================================================
    print("\n=== Step 4B: Shuffled-target model on T-eval ===")
    shuf_model, tokenizer = load_model_with_lora(LORA_SHUFFLED_DIR)

    _split_start = time.time()
    shuf_teval_records, _ = evaluate_split(
        shuf_model, tokenizer, teval_items, "triviaqa",
        label="shuf-T-eval",
    )

    del shuf_model
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # C. Compute all metrics
    # ==================================================================
    print("\n=== Step 4C: Computing metrics ===")

    # Arrays for post-SFT
    ft_teval_conf = np.array([r["confidence"] for r in ft_teval_records])
    ft_teval_corr = np.array([int(r["correct"]) for r in ft_teval_records])
    ft_meval_conf = np.array([r["confidence"] for r in ft_meval_records])
    ft_meval_corr = np.array([int(r["correct"]) for r in ft_meval_records])
    shuf_teval_conf = np.array([r["confidence"] for r in shuf_teval_records])
    shuf_teval_corr = np.array([int(r["correct"]) for r in shuf_teval_records])
    ft_conflict_conf = np.array([r["confidence"] for r in ft_conflict_records])
    ft_conflict_corr = np.array([int(r["correct"]) for r in ft_conflict_records])

    # Use same correctness labels for paired comparisons on T-eval
    # (same items, same order, correctness may differ due to different answers)
    # Pre-reg uses per-condition correctness for AUROC2

    results = {}

    # --- Global AUROC2 ---
    results["auroc2_base_teval"] = float(auroc2(base_teval_conf, base_teval_corr))
    results["auroc2_ft_teval"] = float(auroc2(ft_teval_conf, ft_teval_corr))
    results["auroc2_shuf_teval"] = float(auroc2(shuf_teval_conf, shuf_teval_corr))
    results["auroc2_ft_meval"] = float(auroc2(ft_meval_conf, ft_meval_corr))
    results["auroc2_base_meval"] = float(auroc2(base_meval_conf, base_meval_corr))

    print(f"  AUROC2 base T-eval:    {results['auroc2_base_teval']:.3f}")
    print(f"  AUROC2 ft T-eval:      {results['auroc2_ft_teval']:.3f}")
    print(f"  AUROC2 shuf T-eval:    {results['auroc2_shuf_teval']:.3f}")
    print(f"  AUROC2 ft M-eval:      {results['auroc2_ft_meval']:.3f}")

    # --- H2: Shuffled-target adjustment ---
    shuf_vs_base = paired_bootstrap_auroc2_delta(
        shuf_teval_conf, base_teval_conf, base_teval_corr,
        n_resamples=BOOTSTRAP_N, seed=SEED,
    )
    results["h2_shuf_vs_base"] = shuf_vs_base
    shuf_moved = shuf_vs_base["lo"] > 0  # CI excludes zero and positive

    if shuf_moved:
        print(f"  H2: Shuffled moved (CI [{shuf_vs_base['lo']:.3f}, "
              f"{shuf_vs_base['hi']:.3f}]). Primary delta = ft - shuffled.")
        primary_delta = paired_bootstrap_auroc2_delta(
            ft_teval_conf, shuf_teval_conf, ft_teval_corr,
            n_resamples=BOOTSTRAP_N, seed=SEED,
        )
        results["primary_delta_comparator"] = "shuffled"
    else:
        print(f"  H2: Shuffled did not move (CI [{shuf_vs_base['lo']:.3f}, "
              f"{shuf_vs_base['hi']:.3f}]). Primary delta = ft - base.")
        primary_delta = paired_bootstrap_auroc2_delta(
            ft_teval_conf, base_teval_conf, base_teval_corr,
            n_resamples=BOOTSTRAP_N, seed=SEED,
        )
        results["primary_delta_comparator"] = "base"

    results["primary_delta"] = primary_delta
    h1_met = (primary_delta["lo"] > 0 and
              primary_delta["point_delta"] >= H1_DELTA_FLOOR)
    results["h1_met"] = h1_met
    print(f"  H1: delta={primary_delta['point_delta']:.3f} "
          f"CI=[{primary_delta['lo']:.3f}, {primary_delta['hi']:.3f}] "
          f"-> {'MET' if h1_met else 'NOT MET'}")

    # --- Accuracy ---
    acc_base = float(base_teval_corr.mean())
    acc_ft = float(ft_teval_corr.mean())
    acc_drop = acc_base - acc_ft
    results["accuracy_base"] = acc_base
    results["accuracy_ft"] = acc_ft
    results["accuracy_drop"] = acc_drop
    results["e4_accuracy_ok"] = acc_drop <= ACCURACY_DROP_THRESHOLD
    print(f"  E4: accuracy base={acc_base:.3f} ft={acc_ft:.3f} "
          f"drop={acc_drop:.3f} -> "
          f"{'OK' if results['e4_accuracy_ok'] else 'FAIL'}")

    # --- Ceiling rate ---
    ceil_base = float(np.mean(base_teval_conf >= 95))
    ceil_ft = float(np.mean(ft_teval_conf >= 95))
    results["ceiling_base"] = ceil_base
    results["ceiling_ft"] = ceil_ft
    print(f"  Ceiling: base={ceil_base:.3f} ft={ceil_ft:.3f}")

    # --- VRS ---
    vrs_base = vrs_screen(base_teval_conf, base_teval_corr)
    vrs_ft = vrs_screen(ft_teval_conf, ft_teval_corr)
    results["vrs_base"] = vrs_base
    results["vrs_ft"] = vrs_ft
    vrs_improved = (
        (vrs_base["tier"] == "Invalid" and vrs_ft["tier"] in ("Indeterminate", "Valid")) or
        (vrs_base["tier"] == "Indeterminate" and vrs_ft["tier"] == "Valid")
    )
    results["vrs_improved"] = vrs_improved
    print(f"  VRS: base={vrs_base['tier']} ft={vrs_ft['tier']} "
          f"-> {'improved' if vrs_improved else 'not improved'}")

    # --- H4: Cross-benchmark transfer (M-eval) ---
    h4_delta = paired_bootstrap_auroc2_delta(
        ft_meval_conf, base_meval_conf, base_meval_corr,
        n_resamples=BOOTSTRAP_N, seed=SEED,
    )
    results["h4_delta"] = h4_delta
    h4_met = h4_delta["lo"] > 0
    results["h4_met"] = h4_met
    print(f"  H4: M-eval delta={h4_delta['point_delta']:.3f} "
          f"CI=[{h4_delta['lo']:.3f}, {h4_delta['hi']:.3f}] "
          f"-> {'MET' if h4_met else 'NOT MET'}")

    # --- H3: Within-bin analysis ---
    print("\n  H3: Within-bin analysis")
    diff_map = {d["question_id"]: d for d in teval_difficulty}
    h3_bins = {}
    h3_any_bin_met = False
    h3_residual_any_met = False

    for bin_name in ["Easy", "Medium", "Hard"]:
        bin_mask = np.array([
            diff_map.get(r.get("question_id", ""), {}).get("difficulty_bin", "")
            == bin_name
            for r in base_teval[:len(ft_teval_records)]
        ])
        n_bin = int(bin_mask.sum())
        if n_bin < 10:
            print(f"    {bin_name}: n={n_bin}, too few items, skipping")
            h3_bins[bin_name] = {"n": n_bin, "skipped": True}
            continue

        bc = base_teval_conf[bin_mask]
        fc = ft_teval_conf[bin_mask]
        by = base_teval_corr[bin_mask]
        fy = ft_teval_corr[bin_mask]

        bin_delta = paired_bootstrap_auroc2_delta(
            fc, bc, by, n_resamples=BOOTSTRAP_N, seed=SEED,
        )
        bin_met = bin_delta["lo"] > 0

        # Residualisation: subtract bin mean from confidence
        bc_resid = bc - np.nanmean(bc)
        fc_resid = fc - np.nanmean(fc)
        resid_auroc2_base = auroc2(bc_resid, by)
        resid_auroc2_ft = auroc2(fc_resid, fy)
        resid_delta = paired_bootstrap_auroc2_delta(
            fc_resid, bc_resid, by, n_resamples=BOOTSTRAP_N, seed=SEED,
        )
        resid_met = resid_delta["lo"] > 0

        h3_bins[bin_name] = {
            "n": n_bin,
            "delta": bin_delta,
            "bin_met": bin_met,
            "resid_auroc2_base": resid_auroc2_base,
            "resid_auroc2_ft": resid_auroc2_ft,
            "resid_delta": resid_delta,
            "resid_met": resid_met,
        }

        if bin_met:
            h3_any_bin_met = True
        if resid_met:
            h3_residual_any_met = True

        print(f"    {bin_name} (n={n_bin}): delta={bin_delta['point_delta']:.3f} "
              f"CI=[{bin_delta['lo']:.3f}, {bin_delta['hi']:.3f}] "
              f"{'MET' if bin_met else 'not met'}")
        print(f"      residual: delta={resid_delta['point_delta']:.3f} "
              f"CI=[{resid_delta['lo']:.3f}, {resid_delta['hi']:.3f}] "
              f"{'MET' if resid_met else 'not met'}")

    results["h3_bins"] = h3_bins
    h3_met = h3_any_bin_met and h3_residual_any_met
    results["h3_met"] = h3_met
    print(f"  H3: any_bin={h3_any_bin_met} residual={h3_residual_any_met} "
          f"-> {'MET' if h3_met else 'NOT MET'}")

    # --- H5: Conflict set ---
    print("\n  H5: Conflict set evaluation")
    n_conflict = len(ft_conflict_records)
    results["conflict_n"] = n_conflict

    if n_conflict < 10:
        print(f"  H5: n={n_conflict}, too few items")
        results["h5_met"] = False
        results["h5_power_limited"] = True
    else:
        conflict_auroc2 = auroc2(ft_conflict_conf, ft_conflict_corr)
        # CI vs chance (0.5)
        # Use bootstrap on AUROC2 itself
        rng = np.random.default_rng(SEED)
        boot_auroc2s = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, n_conflict, size=n_conflict)
            a = auroc2(ft_conflict_conf[idx], ft_conflict_corr[idx])
            if not np.isnan(a):
                boot_auroc2s.append(a)
        boot_auroc2s = np.array(boot_auroc2s)
        h5_above_chance = float(np.quantile(boot_auroc2s, 0.025)) > 0.5

        # Compare to baseline on same items (need baseline conflict conf)
        # Get baseline responses for conflict items by question_id
        base_qid_map = {r.get("question_id", ""): r for r in base_teval}
        base_conflict_conf_list = []
        base_conflict_corr_list = []
        for ci in conflict_items[:n_conflict]:
            br = base_qid_map.get(ci["question_id"])
            if br:
                base_conflict_conf_list.append(br["confidence"])
                base_conflict_corr_list.append(int(br["correct"]))
            else:
                base_conflict_conf_list.append(float("nan"))
                base_conflict_corr_list.append(0)

        base_conflict_conf_arr = np.array(base_conflict_conf_list)
        base_conflict_corr_arr = np.array(base_conflict_corr_list)
        base_conflict_auroc2 = auroc2(base_conflict_conf_arr,
                                       base_conflict_corr_arr)
        h5_above_baseline = conflict_auroc2 > base_conflict_auroc2

        h5_met = h5_above_chance and h5_above_baseline
        results["h5_conflict_auroc2"] = float(conflict_auroc2)
        results["h5_base_conflict_auroc2"] = float(base_conflict_auroc2)
        results["h5_above_chance"] = h5_above_chance
        results["h5_above_baseline"] = h5_above_baseline
        results["h5_met"] = h5_met
        results["h5_power_limited"] = n_conflict < 200

        print(f"  H5: ft conflict AUROC2={conflict_auroc2:.3f}, "
              f"base conflict AUROC2={base_conflict_auroc2:.3f}")
        print(f"  H5: above_chance={h5_above_chance} "
              f"above_baseline={h5_above_baseline} "
              f"-> {'MET' if h5_met else 'NOT MET'}"
              f"{' (power-limited)' if n_conflict < 200 else ''}")

    # --- E5: Entropy comparison ---
    print("\n  E5: Entropy comparison")
    if base_entropy_auroc2 is not None:
        results["e5_entropy_auroc2"] = base_entropy_auroc2
        results["e5_ft_verbal_auroc2"] = results["auroc2_ft_teval"]
        e5_entropy_exceeds_ft = base_entropy_auroc2 >= results["auroc2_ft_teval"]
        results["e5_entropy_exceeds_ft"] = e5_entropy_exceeds_ft
        print(f"  E5: entropy AUROC2={base_entropy_auroc2:.3f} "
              f"ft verbal AUROC2={results['auroc2_ft_teval']:.3f}")
        print(f"  E5: entropy {'>=  ' if e5_entropy_exceeds_ft else '<'} "
              f"ft verbal -> "
              f"{'entropy-recapitulation' if e5_entropy_exceeds_ft else 'SFT adds beyond entropy'}")
    else:
        print("  E5: entropy AUROC2 not available from Step 1")
        results["e5_entropy_auroc2"] = None

    # --- E8: AURC ---
    print("\n  E8: AURC (selective prediction)")
    aurc_base = compute_aurc(base_teval_conf, base_teval_corr)
    aurc_ft = compute_aurc(ft_teval_conf, ft_teval_corr)
    results["e8_aurc_base"] = aurc_base
    results["e8_aurc_ft"] = aurc_ft
    print(f"  E8: AURC base={aurc_base:.3f} ft={aurc_ft:.3f}")

    # --- Probe re-fit (E2/E6) ---
    print("\n  E2/E6: Post-SFT probe re-fit")

    # Load baseline probe metrics
    with open(RESULTS_DIR / "probe" / "probe_metrics.json") as f:
        base_probe_metrics = json.load(f)

    probe_results = {}
    if ft_teval_hs:
        # Load baseline T-cal hidden states for training
        tcal_hs = torch.load(HIDDEN_DIR / "baseline_tcal.pt",
                              weights_only=True)
        with open(RESULTS_DIR / "baseline" / "tcal_greedy_responses.json") as f:
            tcal_records = json.load(f)
        tcal_labels = np.array([int(r["correct"]) for r in tcal_records])

        for layer_name in ["first", "middle", "last"]:
            for pos_name in PROBE_POSITIONS:
                config_key = f"{layer_name}_{pos_name}"
                print(f"    Fitting post-SFT probe: {config_key}")

                # Post-SFT eval hidden states
                ft_hs_list = ft_teval_hs.get(config_key, [])
                ft_hs_valid = [h for h in ft_hs_list if h is not None]
                if len(ft_hs_valid) < 20:
                    print(f"      Skipping: only {len(ft_hs_valid)} valid HS")
                    continue

                X_eval_post = np.stack(ft_hs_valid)
                y_eval_post = ft_teval_corr[:len(ft_hs_valid)]

                # Use baseline T-cal for training (same as Step 1b)
                tcal_key = config_key
                if tcal_key in tcal_hs:
                    X_train = tcal_hs[tcal_key]
                    if isinstance(X_train, torch.Tensor):
                        X_train = X_train.numpy()
                    y_train = tcal_labels[:len(X_train)]

                    clf = LogisticRegressionCV(
                        Cs=PROBE_CS, cv=5, penalty="l2",
                        max_iter=2000, random_state=SEED,
                    )
                    clf.fit(X_train, y_train)

                    proba_post = clf.predict_proba(X_eval_post)[:, 1]
                    auroc2_post = auroc2(proba_post, y_eval_post)

                    # Get baseline probe AUROC2 for this config
                    auroc2_pre = base_probe_metrics.get(
                        f"auroc2_{config_key}", None
                    )

                    probe_results[config_key] = {
                        "auroc2_pre": auroc2_pre,
                        "auroc2_post": float(auroc2_post),
                        "n_eval": len(X_eval_post),
                    }

                    if auroc2_pre is not None:
                        delta = auroc2_post - auroc2_pre
                        print(f"      pre={auroc2_pre:.3f} "
                              f"post={auroc2_post:.3f} delta={delta:.3f}")
                    else:
                        print(f"      post={auroc2_post:.3f} "
                              f"(no baseline for comparison)")

    results["probe_results"] = probe_results

    # --- Regime classification (§6.6) ---
    print("\n  Regime classification")
    primary_probe_key = "last_last_answer_token"
    if primary_probe_key in probe_results:
        pr = probe_results[primary_probe_key]
        probe_pre = pr.get("auroc2_pre")
        probe_post = pr["auroc2_post"]

        if probe_pre is not None:
            # Bootstrap CI on probe_post - probe_pre
            # Use the base probe scores from Step 1b
            base_probe_teval = base_probe_scores.get(primary_probe_key, {})
            base_probe_auroc2 = base_probe_teval.get("auroc2", probe_pre)

            # Simplified regime check using point estimates
            # (full bootstrap would require per-item probe scores)
            probe_above_base = probe_post > base_probe_auroc2 + 0.02
            probe_below_base = probe_post < base_probe_auroc2 - 0.02
            ft_above_probe = results["auroc2_ft_teval"] > probe_post + 0.02

            if not probe_above_base and not probe_below_base and h1_met:
                regime = 1
                regime_label = "No probe change + H1 met -> strongest result"
            elif probe_above_base and ft_above_probe:
                regime = 2
                regime_label = "Probe improved + ft exceeds probe -> engineering monitoring"
            elif probe_above_base and not ft_above_probe:
                regime = 3
                regime_label = "Probe improved + ft matches probe -> engineering verbalisation"
            elif probe_below_base and h1_met:
                regime = 4
                regime_label = "Probe degraded + H1 met -> representation collapse"
            else:
                regime = 0
                regime_label = "Unclassified"

            results["regime"] = regime
            results["regime_label"] = regime_label
            print(f"  Regime: {regime} - {regime_label}")
            print(f"    probe_pre={probe_pre:.3f} probe_post={probe_post:.3f} "
                  f"ft_verbal={results['auroc2_ft_teval']:.3f}")
        else:
            results["regime"] = None
            results["regime_label"] = "Cannot classify - no baseline probe"
            print("  Regime: cannot classify (no baseline probe)")
    else:
        results["regime"] = None
        results["regime_label"] = "Cannot classify - no post-SFT probe"
        print("  Regime: cannot classify (no post-SFT probe data)")

    # ==================================================================
    # D. Decision tree
    # ==================================================================
    print("\n" + "=" * 60)
    print("DECISION TREE APPLICATION")
    print("=" * 60)

    decision = {"checks": {}}

    # Clear Proceed checks
    decision["checks"]["h1"] = h1_met
    decision["checks"]["ceiling_below_70"] = ceil_ft < CEILING_CLEAR_PROCEED
    decision["checks"]["vrs_improved"] = vrs_improved
    decision["checks"]["accuracy_ok"] = results["e4_accuracy_ok"]
    decision["checks"]["h4"] = results.get("h4_met", False)
    decision["checks"]["h3"] = h3_met
    decision["checks"]["h5"] = results.get("h5_met", False)
    decision["checks"]["regime_1_or_2"] = results.get("regime") in (1, 2)

    clear_proceed = all(decision["checks"].values())

    # Determine terminal state
    if clear_proceed:
        decision["terminal"] = "Clear Proceed"
        decision["action"] = ("Full design: multi-intervention, multi-seed, "
                              "scale comparison across Gemma 3 4B/12B/27B")
    elif h1_met and results.get("regime") == 3:
        decision["terminal"] = "Regime 3 Decision Point"
        decision["action"] = ("Choose: Branch A (verbalisation paper) or "
                              "Branch B (representation-targeting intervention)")
    elif h1_met and results.get("regime") == 4:
        decision["terminal"] = "Regime 4 Pivot"
        decision["action"] = ("Reorganise design around characterising "
                              "representation collapse")
    elif not h1_met:
        # Check Stop criteria
        stop_reasons = []
        if primary_delta["point_delta"] < 0.02:
            stop_reasons.append("delta < 0.02")
        if acc_drop > 0.10:
            stop_reasons.append("accuracy collapse > 10pp")

        if stop_reasons:
            decision["terminal"] = "Stop"
            decision["action"] = f"Reasons: {'; '.join(stop_reasons)}"
        else:
            decision["terminal"] = "Proceed with Caution"
            decision["action"] = ("Phase 0b on larger model, or "
                                  "pivot intervention class")
    else:
        # H1 met but missing other criteria
        caution_reasons = []
        if not decision["checks"]["ceiling_below_70"]:
            caution_reasons.append("ceiling still >= 70%")
        if not vrs_improved:
            caution_reasons.append("VRS not improved")
        if not results["e4_accuracy_ok"]:
            caution_reasons.append(f"accuracy drop {acc_drop:.3f}")
        if not results.get("h4_met", False):
            caution_reasons.append("H4 (M-eval transfer) not met")
        if not h3_met:
            caution_reasons.append("H3 (within-bin) not met")
        if not results.get("h5_met", False):
            caution_reasons.append("H5 (conflict set) not met")

        decision["terminal"] = "Proceed with Caution"
        decision["action"] = f"Partial: {'; '.join(caution_reasons)}"

    print(f"\n>>> TERMINAL STATE: {decision['terminal']}")
    print(f">>> ACTION: {decision['action']}")
    print(f"\nChecks:")
    for k, v in decision["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # ==================================================================
    # E. Save everything
    # ==================================================================
    results["decision"] = decision
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(EVAL_DIR / "step4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open(EVAL_DIR / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # Save per-model responses
    with open(EVAL_DIR / "ft_teval_responses.json", "w") as f:
        json.dump(ft_teval_records, f, indent=2)
    with open(EVAL_DIR / "ft_meval_responses.json", "w") as f:
        json.dump(ft_meval_records, f, indent=2)
    with open(EVAL_DIR / "ft_conflict_responses.json", "w") as f:
        json.dump(ft_conflict_records, f, indent=2)
    with open(EVAL_DIR / "shuf_teval_responses.json", "w") as f:
        json.dump(shuf_teval_records, f, indent=2)

    print(f"\n[out] Results -> {EVAL_DIR / 'step4_results.json'}")
    print(f"[out] Decision -> {EVAL_DIR / 'decision.json'}")
    print("\n=== Step 4 complete ===")


if __name__ == "__main__":
    main()
