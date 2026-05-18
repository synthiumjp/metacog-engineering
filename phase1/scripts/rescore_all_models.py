"""
rescore_all_models.py — Unified rescoring of ALL models with identical flex matching.
Resolves the scoring discrepancy between Gemma/Qwen (original pipeline) and
Llama/Mistral (ablation pipeline) by applying one matching function to all.

Run from: ~/jpwork/metacog-engineering/phase1/scripts/
"""
import json, os, random, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

SEED = 42
N_TEVAL = 1000
BASE = os.path.expanduser("~/jpwork/metacog-engineering/phase1/results_raw")

# Probe AUROC₂ values (primary config, last layer) from original paper
PROBE = {
    "gemma-3-12b-it": 0.857,
    "gemma-3-27b-it": 0.807,
    "Qwen2.5-7B-Instruct-bf16": 0.762,
    "Meta-Llama-3.1-8B-Instruct-bf16": 0.843,
    "Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED": 0.803,
    "Mistral-7B-Instruct-v0.3": 0.762,
}

# Model configurations: (model_name, baseline_path, [(label, pt_csft_path), ...])
MODELS = [
    ("gemma-3-12b-it",
     f"{BASE}/step1/teval_responses_gemma-3-12b-it.json",
     [("pt-csft r16/lr2e-4", f"{BASE}/step4/step4_probe_target_gemma-3-12b-it.json")]),
    ("gemma-3-27b-it",
     f"{BASE}/step1/teval_responses_gemma-3-27b-it.json",
     [("pt-csft r16/lr2e-4", f"{BASE}/step4/step4_probe_target_gemma-3-27b-it.json")]),
    ("Qwen2.5-7B-Instruct-bf16",
     f"{BASE}/step1/teval_responses_Qwen2.5-7B-Instruct-bf16.json",
     [("pt-csft r16/lr2e-4", f"{BASE}/step4/step4_probe_target_Qwen2.5-7B-Instruct-bf16.json")]),
    ("Meta-Llama-3.1-8B-Instruct-bf16",
     f"{BASE}/step1/teval_responses_Meta-Llama-3.1-8B-Instruct-bf16.json",
     [("pt-csft r16/lr2e-4", f"{BASE}/step4/step4_probe_target_Meta-Llama-3.1-8B-Instruct-bf16.json"),
      ("gentle-lr r16/5e-5", f"{BASE}/step4/ablation_gentle_lr_responses.json"),
      ("low-rank r4/2e-4", f"{BASE}/step4/ablation_low_rank_responses.json"),
      ("gentlest r4/5e-5", f"{BASE}/step4/ablation_gentlest_responses.json")]),
    ("Mistral-7B-Instruct-v0.3",
     f"{BASE}/step1/teval_responses_Mistral-7B-Instruct-v0.3.json",
     [("gentle-lr r16/5e-5", f"{BASE}/step4/ablation_mistral_gentle_lr_responses.json"),
      ("low-rank r4/2e-4", f"{BASE}/step4/ablation_mistral_low_rank_responses.json"),
      ("gentlest r4/5e-5", f"{BASE}/step4/ablation_mistral_gentlest_responses.json")]),
    ("Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED",
     f"{BASE}/step1/teval_responses_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED.json",
     [("gentle-lr r16/5e-5", f"{BASE}/step4/ablation_llama70b_gentle_lr_responses.json"),
      ("low-rank r4/2e-4", f"{BASE}/step4/ablation_llama70b_low_rank_responses.json"),
      ("gentlest r4/5e-5", f"{BASE}/step4/ablation_llama70b_gentlest_responses.json"),
      ("mid-lr r16/1e-4", f"{BASE}/step4/ablation_llama70b_mid_lr_responses.json"),
      ("r32/1e-4", f"{BASE}/step4/ablation_llama70b_r32_mid_lr_responses.json")]),
]

# ---------------------------------------------------------------------------
# Flex matching
# ---------------------------------------------------------------------------
def is_correct_flex(answer, aliases):
    if not answer:
        return False
    pred = answer.lower().strip().rstrip(".")
    if len(pred) < 2:
        return False  # avoid single-char false positives
    for a in aliases:
        a_low = a.lower().strip()
        if len(a_low) < 2:
            continue
        if a_low == pred or a_low in pred or pred in a_low:
            return True
    return False

def is_correct_exact(answer, aliases):
    if not answer:
        return False
    pred = answer.lower().strip().rstrip(".")
    return any(pred == a.lower().strip() for a in aliases)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auroc2(conf, correct):
    c, y = np.asarray(conf, float), np.asarray(correct, int)
    mask = ~np.isnan(c)
    if mask.sum() < 2 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))

def vrs_screen(conf, correct):
    c = np.asarray(conf, float)
    y = np.asarray(correct, int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    if len(c) < 10:
        return "n/a"
    L = float(np.mean(c >= 90))
    modal = float(np.bincount(np.clip(c.astype(int), 0, 100)).argmax())
    TRIN = float(np.mean(c == modal))
    r = float(np.corrcoef(c, y)[0, 1]) if np.std(c) > 0 and np.std(y) > 0 else 0.0
    invalid = (L > 0.90) or (TRIN > 0.80) or (abs(r) < 0.10)
    valid = (L < 0.50) and (TRIN < 0.50) and (abs(r) > 0.20)
    return "Valid" if valid else ("Invalid" if invalid else "Indeterm.")

def ece(conf, correct, n_bins=10):
    c = np.asarray(conf, float) / 100.0
    y = np.asarray(correct, int)
    mask = ~np.isnan(c * 100)
    c, y = c[mask], y[mask]
    if len(c) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if in_bin.sum() == 0:
            continue
        total += in_bin.sum() * abs(y[in_bin].mean() - c[in_bin].mean())
    return float(total / len(c))

def brier(conf, correct):
    c = np.asarray(conf, float) / 100.0
    y = np.asarray(correct, int)
    mask = ~np.isnan(c * 100)
    return float(np.mean((c[mask] - y[mask]) ** 2))

# ---------------------------------------------------------------------------
# Normalize response format
# ---------------------------------------------------------------------------
def get_qid(r):
    return r.get("question_id", r.get("qid", ""))

def get_answer(r):
    return r.get("parsed_answer", r.get("answer", ""))

def get_conf(r):
    for k in ["confidence", "parsed_confidence"]:
        v = r.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return float("nan")

# ---------------------------------------------------------------------------
# Score a response file
# ---------------------------------------------------------------------------
def score_file(path, aliases_map, label=""):
    with open(path) as f:
        data = json.load(f)

    confs, corrects_flex, corrects_exact, answers = [], [], [], []
    for r in data:
        qid = get_qid(r)
        ans = get_answer(r)
        conf = get_conf(r)
        als = aliases_map.get(qid, [])
        corrects_flex.append(int(is_correct_flex(ans, als)))
        corrects_exact.append(int(is_correct_exact(ans, als)))
        confs.append(conf)
        answers.append(ans.lower().strip().rstrip("."))

    c = np.array(confs)
    yf = np.array(corrects_flex)
    ye = np.array(corrects_exact)
    mask = ~np.isnan(c)

    return {
        "label": label,
        "n": len(data),
        "parse_rate": round(float(mask.mean()), 3),
        "acc_exact": round(float(ye.mean()), 3),
        "acc_flex": round(float(yf.mean()), 3),
        "auroc2_flex": round(auroc2(c, yf), 3),
        "auroc2_exact": round(auroc2(c, ye), 3),
        "vrs_flex": vrs_screen(c, yf),
        "ece_flex": round(ece(c, yf), 3),
        "brier_flex": round(brier(c, yf), 3),
        "conf_mean": round(float(np.nanmean(c)), 1) if mask.any() else None,
        "conf_std": round(float(np.nanstd(c)), 1) if mask.any() else None,
        "_confs": c,
        "_corrects": yf,
        "_answers": answers,
    }

# ---------------------------------------------------------------------------
# E10 diagnostic
# ---------------------------------------------------------------------------
def compute_e10(base_result, ft_result):
    bc, fc = base_result["_confs"], ft_result["_confs"]
    by, fy = base_result["_corrects"], ft_result["_corrects"]
    unchanged = np.where(by == fy)[0]
    if len(unchanged) < 20:
        return {"n": len(unchanged), "delta": "n/a"}
    mask = ~np.isnan(bc[unchanged]) & ~np.isnan(fc[unchanged])
    idx = unchanged[mask]
    y = by[idx]
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": len(idx), "delta": "degenerate"}
    auc_b = roc_auc_score(y, bc[idx])
    auc_f = roc_auc_score(y, fc[idx])
    return {"n": int(len(idx)), "delta": round(auc_f - auc_b, 3)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[data] Loading TriviaQA aliases...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    random.seed(SEED)
    random.shuffle(indices)
    aliases_map = {}
    for i in indices[:N_TEVAL]:
        row = ds[int(i)]
        aliases_map[row["question_id"]] = row["answer"]["aliases"]

    all_results = []

    for model_name, baseline_path, pt_configs in MODELS:
        probe_auc = PROBE.get(model_name, 0.80)
        short = model_name.split("/")[-1][:20]

        print(f"\n{'='*70}")
        print(f"  {model_name}")
        print(f"{'='*70}")

        # Baseline
        if not os.path.exists(baseline_path):
            print(f"  [skip] baseline not found: {baseline_path}")
            continue
        base = score_file(baseline_path, aliases_map, f"{short} baseline")
        base["recovery"] = round((base["auroc2_flex"] - 0.5) / (probe_auc - 0.5) * 100, 1)
        print(f"  Baseline:  acc_flex={base['acc_flex']:.3f}  auroc2={base['auroc2_flex']:.3f}  VRS={base['vrs_flex']}")
        all_results.append({"model": model_name, **{k: v for k, v in base.items() if not k.startswith("_")}})

        # PT-CSFT configs
        best_auc = 0
        best_label = ""
        for label, path in pt_configs:
            if not os.path.exists(path):
                print(f"  [skip] {label}: {path}")
                continue
            r = score_file(path, aliases_map, f"{short} {label}")
            r["recovery"] = round((r["auroc2_flex"] - 0.5) / (probe_auc - 0.5) * 100, 1)
            e10 = compute_e10(base, r)
            r["e10_n"] = e10["n"]
            r["e10_delta"] = e10["delta"]
            drop = r["acc_flex"] - base["acc_flex"]

            print(f"  {label:<22} acc={r['acc_flex']:.3f} (Δ{drop:+.3f})  "
                  f"auroc2={r['auroc2_flex']:.3f}  rec={r['recovery']:.0f}%  "
                  f"ECE={r['ece_flex']:.3f}  VRS={r['vrs_flex']}  "
                  f"E10={r['e10_delta']}  conf={r['conf_mean']:.1f}±{r['conf_std']:.1f}")

            all_results.append({"model": model_name, **{k: v for k, v in r.items() if not k.startswith("_")}})

            if r["auroc2_flex"] > best_auc:
                best_auc = r["auroc2_flex"]
                best_label = label

        if best_label:
            print(f"  >>> Best: {best_label} (AUROC₂={best_auc:.3f})")

    # Summary table
    print(f"\n{'='*110}")
    print("UNIFIED SUMMARY TABLE (flex matching, all models)")
    print(f"{'='*110}")
    print(f"{'Model':<20} {'Config':<22} {'Acc':>6} {'AUROC₂':>7} {'Rec%':>6} "
          f"{'ECE':>6} {'VRS':<11} {'E10':>6} {'Conf':>10}")
    print("-" * 110)

    for r in all_results:
        m = r["model"][:20]
        lab = r["label"].split(" ", 1)[-1] if " " in r["label"] else r["label"]
        lab = lab[:22]
        e10 = r.get("e10_delta", "—")
        e10_s = f"{e10:+.3f}" if isinstance(e10, float) else str(e10)
        conf_s = f"{r['conf_mean']:.0f}±{r['conf_std']:.0f}" if r.get("conf_mean") else "—"
        rec = r.get("recovery", "—")
        rec_s = f"{rec:.0f}" if isinstance(rec, (int, float)) else rec
        print(f"{m:<20} {lab:<22} {r['acc_flex']:>6.3f} {r['auroc2_flex']:>7.3f} {rec_s:>6} "
              f"{r['ece_flex']:>6.3f} {r.get('vrs_flex','—'):<11} {e10_s:>6} {conf_s:>10}")

    # Save
    outpath = os.path.join(BASE, "step4", "unified_rescore_all_models.json")
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    with open(outpath, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
