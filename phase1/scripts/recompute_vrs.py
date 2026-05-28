#!/usr/bin/env python3
"""
Complete VRS Recomputation for Paper Audit
==========================================
Definitive single-pass VRS verification for every condition in the paper.

Covers:
  - Baselines (7 models, step1 teval)
  - Primary PT-CSFT (step4 probe_target, last-layer)
  - Ablation configs (step4 ablation_*_responses)
  - Multi-seed replication (step4 ablation_multiseed_*)
  - Appendix E: Middle-layer PT-CSFT
  - Appendix F: MMLU meval
  - 70B curriculum (text + logit channels)
  - 70B balanced confonly (stored correctness)
  - ARC confonly PT-CSFT (Table 10)

File format handling:
  - TriviaQA: flex-matched correctness (bidirectional substring, min 2 chars)
  - MMLU: stored correctness (letter matching done at eval time)
  - Confonly: stored correctness (answer from separate generation step)
  - Curriculum: stored correctness, dual confidence channels
  - Answer field: checks parsed_answer, answer_text, answer (covers all formats)

Usage:
    cd ~/jpwork/metacog-engineering/phase1/scripts
    python3 recompute_vrs.py

Output:
    - Console: sectioned table + paper cross-check + seed stability
    - JSON: vrs_recomputation_complete.json
"""

import json, glob, os, sys
import numpy as np
import random
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Canonical VRS
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

try:
    from utils_phase0 import vrs_screen
    print("Using vrs_screen from utils_phase0.py")
except ImportError:
    print("WARNING: Could not import utils_phase0. Using inline copy.")

    def vrs_screen(confidence, correct, ceiling_threshold=95.0, floor_threshold=5.0):
        c = np.asarray(confidence, dtype=float)
        y = np.asarray(correct, dtype=int)
        mask = ~np.isnan(c)
        c, y = c[mask], y[mask]
        n = len(c)
        if n == 0:
            return {"tier": "undefined", "n": 0, "L": 0, "TRIN": 0, "r": 0}
        L = float(np.mean(c >= ceiling_threshold))
        Fp = float(np.mean(c <= floor_threshold))
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
        return {"L": L, "Fp": Fp, "RBS": RBS, "TRIN": TRIN, "r": r, "tier": tier, "n": n}


# ---------------------------------------------------------------------------
# TriviaQA aliases
# ---------------------------------------------------------------------------
print("Loading TriviaQA aliases...")
ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
indices = list(range(len(ds)))
random.seed(42)
random.shuffle(indices)
aliases_list = [ds[int(i)]["answer"]["aliases"] for i in indices[:1000]]


def flex_correct(response, aliases):
    """Bidirectional substring match, min 2 chars."""
    r = response.lower().strip()
    return any(
        (a.lower() in r or r in a.lower())
        for a in aliases
        if len(a) >= 2 and len(r) >= 2
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_why(vrs_result):
    """Human-readable reason for the VRS tier."""
    reasons = []
    if vrs_result["L"] >= 0.70:
        reasons.append("L>=0.70")
    if vrs_result["TRIN"] >= 0.80:
        reasons.append("TRIN>=0.80")
    if abs(vrs_result["r"]) < 0.05:
        reasons.append("|r|<0.05")
    if vrs_result["tier"] == "Indeterminate":
        if vrs_result["L"] >= 0.40:
            reasons.append("L>=0.40")
        if vrs_result["TRIN"] >= 0.60:
            reasons.append("TRIN>=0.60")
    if vrs_result["tier"] == "Valid":
        reasons.append("all clear")
    return ", ".join(reasons) if reasons else "—"


def make_result(label, confs, correct, n_skipped=0):
    """Compute VRS + AUROC2 and return a result dict."""
    c_arr = np.array(confs)
    y_arr = np.array(correct)

    # Filter NaN
    mask = ~np.isnan(c_arr)
    c_arr, y_arr = c_arr[mask], y_arr[mask]

    if len(c_arr) < 10:
        return None

    vrs = vrs_screen(c_arr, y_arr)
    auroc = roc_auc_score(y_arr, c_arr) if len(set(y_arr)) > 1 else float("nan")

    return {
        "label": label,
        "n": len(c_arr),
        "n_skipped": n_skipped,
        "acc": float(y_arr.mean()),
        "auroc2": auroc,
        "conf_mean": float(c_arr.mean()),
        "conf_std": float(c_arr.std()),
        "L": vrs["L"],
        "TRIN": vrs["TRIN"],
        "r": vrs["r"],
        "tier": vrs["tier"],
        "why": get_why(vrs),
    }


def load_triviaqa_file(filepath, label):
    """Load a TriviaQA response file, flex-match correctness."""
    with open(filepath) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        return None

    confs, correct = [], []
    n_skipped = 0
    for i, r in enumerate(data[:1000]):
        c = r.get("parsed_confidence", r.get("confidence", None))
        if c is None or (isinstance(c, float) and np.isnan(c)):
            n_skipped += 1
            continue
        confs.append(float(c))
        # Try all known answer field names
        ans = r.get("parsed_answer", r.get("answer_text", r.get("answer", "")))
        if i < len(aliases_list) and ans:
            correct.append(int(flex_correct(ans, aliases_list[i])))
        elif r.get("correct") is not None:
            correct.append(int(r["correct"]))
        else:
            correct.append(0)

    return make_result(label, confs, correct, n_skipped)


def load_stored_correct_file(filepath, label, conf_key="confidence"):
    """Load a file using stored correctness (confonly, MMLU, etc.)."""
    with open(filepath) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        return None

    confs, correct = [], []
    n_skipped = 0
    for r in data[:1000]:
        c = r.get("parsed_confidence", r.get(conf_key, None))
        if c is None or (isinstance(c, float) and np.isnan(c)):
            n_skipped += 1
            continue
        confs.append(float(c))
        correct.append(int(r.get("correct", False)))

    return make_result(label, confs, correct, n_skipped)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = os.path.expanduser("~/jpwork/metacog-engineering/phase1/results_raw")
STEP1 = f"{RESULTS}/step1"
STEP4 = f"{RESULTS}/step4"
DOMAIN = f"{RESULTS}/domain_gen"

results = []

# ===========================================================================
# 1. BASELINES (Table 5 baseline column)
# ===========================================================================
print("\n=== 1. Baselines ===")
baselines = [
    ("gemma-3-12b-it", "Baseline Gemma 12B"),
    ("gemma-3-27b-it", "Baseline Gemma 27B"),
    ("Qwen2.5-7B-Instruct-bf16", "Baseline Qwen 7B"),
    ("Qwen2.5-32B-Instruct-bf16", "Baseline Qwen 32B"),
    ("Meta-Llama-3.1-8B-Instruct-bf16", "Baseline Llama 8B"),
    ("Mistral-7B-Instruct-v0.3", "Baseline Mistral 7B"),
    ("Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED", "Baseline Llama 70B"),
]
for model, label in baselines:
    path = f"{STEP1}/teval_responses_{model}.json"
    if os.path.exists(path):
        r = load_triviaqa_file(path, label)
        if r:
            results.append(r)
            print(f"  {label}: {r['tier']}")

# ===========================================================================
# 2. PRIMARY PT-CSFT (Table 5, step4_probe_target files)
# ===========================================================================
print("\n=== 2. Primary PT-CSFT ===")
primaries = [
    ("step4_probe_target_gemma-3-12b-it.json", "Gemma 12B PT-CSFT"),
    ("step4_probe_target_gemma-3-27b-it.json", "Gemma 27B PT-CSFT"),
    ("step4_probe_target_Qwen2.5-7B-Instruct-bf16.json", "Qwen 7B PT-CSFT (primary)"),
    ("step4_probe_target_Meta-Llama-3.1-8B-Instruct-bf16.json", "Llama 8B PT-CSFT (primary, failed)"),
    # 70B primary excluded — known mis-joined correctness, not cited
]
for fname, label in primaries:
    path = f"{STEP4}/{fname}"
    if os.path.exists(path):
        r = load_triviaqa_file(path, label)
        if r:
            results.append(r)
            print(f"  {label}: {r['tier']} (acc={r['acc']:.3f})")

# Gemma 12B post-PT (alternative location)
path = f"{RESULTS}/step1_post_pt/teval_responses_gemma-3-12b-it.json"
if os.path.exists(path):
    r = load_triviaqa_file(path, "Gemma 12B PT-CSFT (post_pt)")
    if r:
        results.append(r)
        print(f"  Gemma 12B PT-CSFT (post_pt): {r['tier']}")

# ===========================================================================
# 3. ABLATION CONFIGS (§6, *_responses.json, excluding multiseed)
# ===========================================================================
print("\n=== 3. Ablation configs ===")
ablation_files = sorted(glob.glob(f"{STEP4}/ablation_*_responses.json"))
ablation_files = [f for f in ablation_files if "multiseed" not in os.path.basename(f)]
for filepath in ablation_files:
    label = os.path.basename(filepath).replace("_responses.json", "")
    r = load_triviaqa_file(filepath, label)
    if r:
        results.append(r)
        print(f"  {label}: {r['tier']}")

# ===========================================================================
# 4. MULTI-SEED (§5.3)
# ===========================================================================
print("\n=== 4. Multi-seed ===")
multiseed_files = sorted(glob.glob(f"{STEP4}/ablation_multiseed_*_responses.json"))
for filepath in multiseed_files:
    label = os.path.basename(filepath).replace("ablation_", "").replace("_responses.json", "")
    r = load_triviaqa_file(filepath, label)
    if r:
        results.append(r)

# Group for summary
seed_groups = defaultdict(list)
for r in results:
    if "multiseed" in r["label"]:
        if "llama8b" in r["label"]:
            seed_groups["Llama 8B gentle-lr"].append(r)
        elif "mistral" in r["label"]:
            seed_groups["Mistral 7B gentlest"].append(r)
        elif "qwen" in r["label"]:
            seed_groups["Qwen 7B primary"].append(r)

for group, seeds in seed_groups.items():
    tiers = [s["tier"] for s in seeds]
    aurocs = [s["auroc2"] for s in seeds]
    print(f"  {group} ({len(seeds)} seeds): "
          f"AUROC2={np.mean(aurocs):.3f}+/-{np.std(aurocs):.3f}  "
          f"Tiers: {', '.join(tiers)}")

# ===========================================================================
# 5. APPENDIX E: Middle-layer PT-CSFT
# ===========================================================================
print("\n=== 5. Appendix E (middle-layer) ===")
middle_files = [
    ("step4_probe_target_middle_teval_gemma-3-12b-it.json", "AppE: Gemma 12B middle"),
    ("step4_probe_target_middle_teval_Qwen2.5-7B-Instruct-bf16.json", "AppE: Qwen 7B middle"),
]
for fname, label in middle_files:
    path = f"{STEP4}/{fname}"
    if os.path.exists(path):
        r = load_triviaqa_file(path, label)
        if r:
            results.append(r)
            print(f"  {label}: {r['tier']}")

# ===========================================================================
# 6. APPENDIX F: MMLU (meval)
# ===========================================================================
print("\n=== 6. Appendix F (MMLU) ===")
meval_files = [
    ("step4_probe_target_meval_gemma-3-12b-it.json", "AppF: Gemma 12B MMLU"),
    ("step4_probe_target_meval_gemma-3-27b-it.json", "AppF: Gemma 27B MMLU"),
    ("step4_probe_target_meval_Qwen2.5-7B-Instruct-bf16.json", "AppF: Qwen 7B MMLU"),
    ("step4_probe_target_meval_Meta-Llama-3.1-8B-Instruct-bf16.json", "AppF: Llama 8B MMLU"),
    ("step4_probe_target_meval_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED.json", "AppF: Llama 70B MMLU"),
]
for fname, label in meval_files:
    path = f"{STEP4}/{fname}"
    if os.path.exists(path):
        r = load_stored_correct_file(path, label)
        if r:
            results.append(r)
            print(f"  {label}: {r['tier']}")

# ===========================================================================
# 7. 70B CURRICULUM (§7.3.4 — text + logit channels)
# ===========================================================================
print("\n=== 7. 70B Curriculum ===")
CUR_PATH = f"{DOMAIN}/curriculum_70b/curriculum_responses.json"
if os.path.exists(CUR_PATH):
    with open(CUR_PATH) as f:
        cur = json.load(f)
    correct_cur = np.array([int(r["correct"]) for r in cur])

    for conf_key, label in [
        ("text_confidence", "70B curriculum TEXT"),
        ("logit_confidence", "70B curriculum LOGIT"),
    ]:
        confs = np.array([float(r[conf_key]) for r in cur])
        r = make_result(label, confs.tolist(), correct_cur.tolist())
        if r:
            results.append(r)
            print(f"  {label}: {r['tier']}")
else:
    print("  WARNING: Curriculum file not found")

# ===========================================================================
# 8. 70B BALANCED CONFONLY (§7.3.3 — stored correctness)
# ===========================================================================
print("\n=== 8. 70B Balanced confonly ===")
for name, label in [
    ("balanced_confonly_v1_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED_responses.json",
     "70B balanced confonly v1"),
    ("balanced_confonly_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED_responses.json",
     "70B balanced confonly"),
]:
    path = f"{STEP4}/{name}"
    if os.path.exists(path):
        r = load_stored_correct_file(path, label)
        if r:
            results.append(r)
            print(f"  {label}: {r['tier']}")

# ===========================================================================
# 9. ARC CONFONLY (Table 10)
# ===========================================================================
print("\n=== 9. ARC confonly ===")
ARC_PATH = f"{DOMAIN}/arc_confonly/confonly_eval.json"
if os.path.exists(ARC_PATH):
    with open(ARC_PATH) as f:
        arc = json.load(f)
    correct_arc = np.array([int(r["correct"]) for r in arc])

    # Baseline channel
    bl_confs = []
    for r in arc:
        c = r.get("baseline_confidence", None)
        bl_confs.append(float(c) if c is not None else float("nan"))
    bl_r = make_result("ARC baseline (Qwen 7B)", bl_confs, correct_arc.tolist())
    if bl_r:
        results.append(bl_r)
        print(f"  ARC baseline: {bl_r['tier']}")

    # PT-CSFT confonly channel
    pt_confs = []
    for r in arc:
        c = r.get("ptcsft_confidence", None)
        pt_confs.append(float(c) if c is not None else float("nan"))
    pt_r = make_result("ARC confonly PT-CSFT (Qwen 7B)", pt_confs, correct_arc.tolist())
    if pt_r:
        results.append(pt_r)
        print(f"  ARC confonly PT-CSFT: {pt_r['tier']}")

    # Also load ARC baseline from separate file if it exists
    arc_bl_path = f"{DOMAIN}/arc_confonly/baseline_eval.json"
    if os.path.exists(arc_bl_path):
        r = load_stored_correct_file(arc_bl_path, "ARC baseline (from baseline_eval)")
        if r:
            results.append(r)
            print(f"  ARC baseline (baseline_eval): {r['tier']}")
else:
    print("  WARNING: ARC confonly file not found")


# ===========================================================================
# OUTPUT TABLE
# ===========================================================================
print("\n" + "=" * 135)
print("COMPLETE VRS RECOMPUTATION TABLE")
print("=" * 135)
print(f"  Invalid:       L >= 0.70 OR TRIN >= 0.80 OR |r| < 0.05")
print(f"  Indeterminate: L >= 0.40 OR TRIN >= 0.60")
print(f"  Valid:         otherwise (L < 0.40 AND TRIN < 0.60 AND |r| >= 0.05)")
print("=" * 135)

header = (f"{'Label':<55} {'N':>4} {'Acc':>6} {'AUROC2':>7} "
          f"{'Conf':>5} {'L':>6} {'TRIN':>6} {'r':>7} {'Tier':<14} {'Why'}")
print(header)
print("-" * 135)

sections = [
    ("BASELINES", lambda r: r["label"].startswith("Baseline")),
    ("PRIMARY PT-CSFT", lambda r: "PT-CSFT" in r["label"] and "middle" not in r["label"]
     and "AppE" not in r["label"] and "ARC" not in r["label"]
     and "curriculum" not in r["label"] and "confonly" not in r["label"]),
    ("ABLATION CONFIGS", lambda r: r["label"].startswith("ablation_") and "multiseed" not in r["label"]),
    ("MULTI-SEED", lambda r: "multiseed" in r["label"]),
    ("APPENDIX E: MIDDLE-LAYER", lambda r: "AppE" in r["label"]),
    ("APPENDIX F: MMLU", lambda r: "AppF" in r["label"]),
    ("70B CURRICULUM", lambda r: "curriculum" in r["label"]),
    ("70B CONFONLY", lambda r: "confonly" in r["label"] and "ARC" not in r["label"]),
    ("ARC CONFONLY (Table 10)", lambda r: "ARC" in r["label"]),
]

for section_name, filter_fn in sections:
    section_results = [r for r in results if filter_fn(r)]
    if not section_results:
        continue
    print(f"\n  --- {section_name} ---")
    for r in section_results:
        auroc_str = f"{r['auroc2']:>7.3f}" if not np.isnan(r['auroc2']) else "    nan"
        print(f"  {r['label']:<53} {r['n']:>4} {r['acc']:>6.3f} {auroc_str} "
              f"{r['conf_mean']:>5.1f} {r['L']:>6.3f} {r['TRIN']:>6.3f} {r['r']:>+7.3f} "
              f"{r['tier']:<14} {r['why']}")


# ===========================================================================
# PAPER CROSS-CHECK
# ===========================================================================
print("\n" + "=" * 105)
print("PAPER CROSS-CHECK: Every VRS claim")
print("=" * 105)

checks = [
    ("Table 5: Gemma 12B", "Gemma 12B PT-CSFT"),
    ("Table 5: Gemma 27B", "Gemma 27B PT-CSFT"),
    ("Table 5: Qwen 7B primary", "Qwen 7B PT-CSFT (primary)"),
    ("Table 5: Qwen 32B primary", "ablation_qwen32b_primary"),
    ("Table 5: Qwen 32B gentle-lr", "ablation_qwen32b_gentle_lr"),
    ("Table 5: Llama 8B low-rank", "ablation_low_rank"),
    ("Table 5: Llama 8B gentle-lr", "ablation_gentle_lr"),
    ("Table 5: Llama 8B gentlest", "ablation_gentlest"),
    ("Table 5: Mistral gentlest", "ablation_mistral_gentlest"),
    ("Table 5: Mistral gentle-lr", "ablation_mistral_gentle_lr"),
    ("Table 5: Mistral low-rank", "ablation_mistral_low_rank"),
    ("§7.3.3: 70B confonly", "70B balanced confonly v1"),
    ("§7.3.4: 70B curriculum TEXT", "70B curriculum TEXT"),
    ("§7.3.4: 70B curriculum LOGIT", "70B curriculum LOGIT"),
    ("Table 10: ARC confonly", "ARC confonly PT-CSFT"),
    ("App E: Gemma 12B middle", "AppE: Gemma 12B"),
    ("App E: Qwen 7B middle", "AppE: Qwen 7B"),
    ("App F: Gemma 12B MMLU", "AppF: Gemma 12B"),
    ("App F: Gemma 27B MMLU", "AppF: Gemma 27B"),
    ("App F: Qwen 7B MMLU", "AppF: Qwen 7B"),
    ("App F: Llama 8B MMLU", "AppF: Llama 8B"),
    ("App F: Llama 70B MMLU", "AppF: Llama 70B"),
]

for desc, pattern in checks:
    match = [r for r in results if pattern in r["label"]]
    # Prefer non-post_pt version
    if len(match) > 1:
        preferred = [m for m in match if "post_pt" not in m["label"]]
        if preferred:
            match = preferred
    if match:
        r = match[0]
        print(f"  {desc:<40} L={r['L']:.3f}  TRIN={r['TRIN']:.3f}  r={r['r']:+.3f}  "
              f"AUROC2={r['auroc2']:.3f}  -> {r['tier']:<14} ({r['why']})")
    else:
        print(f"  {desc:<40} NOT FOUND")

# ===========================================================================
# SEED STABILITY
# ===========================================================================
print("\n" + "=" * 80)
print("MULTI-SEED VRS STABILITY")
print("=" * 80)

for group, seeds in seed_groups.items():
    tier_counts = defaultdict(int)
    for s in seeds:
        tier_counts[s["tier"]] += 1
    tier_str = ", ".join(f"{t}: {c}" for t, c in sorted(tier_counts.items()))
    aurocs = [s["auroc2"] for s in seeds]
    print(f"\n  {group} ({len(seeds)} seeds):")
    print(f"    AUROC2 = {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f} "
          f"[{np.min(aurocs):.3f}, {np.max(aurocs):.3f}]")
    print(f"    VRS tiers: {tier_str}")
    for s in seeds:
        seed_id = s["label"].split("seed")[-1] if "seed" in s["label"] else "?"
        print(f"      seed {seed_id:<6} AUROC2={s['auroc2']:.3f}  "
              f"L={s['L']:.3f}  TRIN={s['TRIN']:.3f}  r={s['r']:+.3f}  -> {s['tier']}")


# ===========================================================================
# SAVE
# ===========================================================================
output = {
    "thresholds": {
        "Invalid": "L >= 0.70 OR TRIN >= 0.80 OR |r| < 0.05",
        "Indeterminate": "L >= 0.40 OR TRIN >= 0.60",
        "Valid": "otherwise (L < 0.40 AND TRIN < 0.60 AND |r| >= 0.05)",
    },
    "results": results,
    "seed_stability": {
        group: {
            "n_seeds": len(seeds),
            "auroc2_mean": float(np.mean([s["auroc2"] for s in seeds])),
            "auroc2_std": float(np.std([s["auroc2"] for s in seeds])),
            "tiers": {s["label"].split("seed")[-1]: s["tier"] for s in seeds},
        }
        for group, seeds in seed_groups.items()
    },
}

out_path = f"{STEP4}/vrs_recomputation_complete.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"\nTotal conditions verified: {len(results)}")
