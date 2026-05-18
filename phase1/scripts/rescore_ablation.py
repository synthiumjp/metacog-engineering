"""
rescore_ablation.py — Comprehensive rescoring of Llama 8B ablation results.
Computes all metrics with flex matching:
  - Accuracy, AUROC₂, VRS
  - ECE (10 bins), Brier score
  - E10 (answer-unchanged diagnostic)
  - Probe recovery
  - Comparison with baseline

Run from: ~/jpwork/metacog-engineering/phase1/scripts/
"""
import json, os, random, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_TEVAL = 1000
BASELINE_ACC = None  # computed from data
PROBE_AUROC2 = 0.843  # Llama 8B probe (primary, last layer)

RESULTS_DIR = os.path.expanduser(
    "~/jpwork/metacog-engineering/phase1/results_raw/step4")
BASELINE_PATH = os.path.expanduser(
    "~/jpwork/metacog-engineering/phase1/results_raw/step1/teval_responses_Meta-Llama-3.1-8B-Instruct-bf16.json")
ORIGINAL_PATH = os.path.join(RESULTS_DIR,
    "step4_probe_target_Meta-Llama-3.1-8B-Instruct-bf16.json")

ABLATION_LABELS = ["gentle_lr", "low_rank", "gentlest"]

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def is_correct_exact(answer, aliases):
    if not answer:
        return False
    pred = answer.lower().strip().rstrip(".")
    return any(pred == a.lower().strip() for a in aliases)

def is_correct_flex(answer, aliases):
    """Substring match: alias in answer OR answer in alias."""
    if not answer:
        return False
    pred = answer.lower().strip().rstrip(".")
    for a in aliases:
        a_low = a.lower().strip()
        if a_low == pred or a_low in pred or pred in a_low:
            return True
    return False

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auroc2(confidence, correct):
    c = np.asarray(confidence, float)
    y = np.asarray(correct, int)
    mask = ~np.isnan(c)
    if mask.sum() < 2 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))

def vrs_screen(confidence, correct):
    c = np.asarray(confidence, float)
    y = np.asarray(correct, int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    n = len(c)
    if n < 10:
        return {"status": "insufficient_data"}
    L = float(np.mean(c >= 90))
    modal = float(np.bincount(np.clip(c.astype(int), 0, 100)).argmax())
    top_mask = c == modal
    TRIN = float(np.mean(top_mask))
    if np.std(c) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(c, y)[0, 1])
    else:
        r = 0.0
    invalid = (L > 0.90) or (TRIN > 0.80) or (abs(r) < 0.10)
    valid = (L < 0.50) and (TRIN < 0.50) and (abs(r) > 0.20)
    status = "Valid" if valid else ("Invalid" if invalid else "Indeterminate")
    return {"status": status, "L": round(L, 4), "TRIN": round(TRIN, 4), "r": round(r, 4)}

def ece(confidence, correct, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    c = np.asarray(confidence, float) / 100.0  # scale to [0,1]
    y = np.asarray(correct, int)
    mask = ~np.isnan(c * 100)  # original scale NaN check
    c, y = c[mask], y[mask]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if in_bin.sum() == 0:
            continue
        avg_conf = c[in_bin].mean()
        avg_acc = y[in_bin].mean()
        total += in_bin.sum() * abs(avg_acc - avg_conf)
    return float(total / len(c))

def brier_score(confidence, correct):
    c = np.asarray(confidence, float) / 100.0
    y = np.asarray(correct, int)
    mask = ~np.isnan(c * 100)
    c, y = c[mask], y[mask]
    return float(np.mean((c - y) ** 2))

def paired_bootstrap_ci(conf_a, conf_b, correct, n_boot=10000, seed=42):
    """Paired bootstrap CI for AUROC₂(A) - AUROC₂(B)."""
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(conf_a) & ~np.isnan(conf_b)
    a, b, y = conf_a[mask], conf_b[mask], correct[mask]
    n = len(y)
    if n < 10 or y.sum() == 0 or y.sum() == n:
        return {"delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ya, yb, yy = a[idx], b[idx], y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        deltas.append(roc_auc_score(yy, ya) - roc_auc_score(yy, yb))
    deltas = np.array(deltas)
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    return {
        "delta": round(obs, 4),
        "ci_lo": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_hi": round(float(np.percentile(deltas, 97.5)), 4),
        "sig": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
    }

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_aliases():
    print("[data] Loading TriviaQA aliases...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    random.seed(SEED)
    random.shuffle(indices)
    aliases = {}
    for i in indices[:N_TEVAL]:
        row = ds[int(i)]
        aliases[row["question_id"]] = row["answer"]["aliases"]
    return aliases

def load_responses(path):
    with open(path) as f:
        return json.load(f)

def get_confidence(r):
    """Handle different key names across result files."""
    for key in ["confidence", "parsed_confidence"]:
        v = r.get(key)
        if v is not None:
            return float(v)
    return float("nan")

# ---------------------------------------------------------------------------
# Score a result set
# ---------------------------------------------------------------------------
def score(responses, aliases, label=""):
    confs, corrects_exact, corrects_flex = [], [], []
    answers = []
    for r in responses:
        qid = r["question_id"]
        ans = r.get("parsed_answer", "")
        conf = get_confidence(r)
        als = aliases.get(qid, [])
        corrects_exact.append(int(is_correct_exact(ans, als)))
        corrects_flex.append(int(is_correct_flex(ans, als)))
        confs.append(conf)
        answers.append(ans.lower().strip().rstrip("."))

    c = np.array(confs)
    ye = np.array(corrects_exact)
    yf = np.array(corrects_flex)

    mask = ~np.isnan(c)
    parse_rate = mask.mean()

    result = {
        "label": label,
        "n": len(responses),
        "parse_rate": round(float(parse_rate), 3),
        "acc_exact": round(float(ye.mean()), 3),
        "acc_flex": round(float(yf.mean()), 3),
        "auroc2_exact": round(auroc2(c, ye), 3),
        "auroc2_flex": round(auroc2(c, yf), 3),
        "vrs_flex": vrs_screen(c, yf),
        "ece_flex": round(ece(c, yf), 3),
        "brier_flex": round(brier_score(c, yf), 3),
        "conf_mean": round(float(np.nanmean(c)), 1) if mask.any() else None,
        "conf_std": round(float(np.nanstd(c)), 1) if mask.any() else None,
    }

    # Recovery
    auc = result["auroc2_flex"]
    if not np.isnan(auc):
        result["recovery_pct"] = round((auc - 0.5) / (PROBE_AUROC2 - 0.5) * 100, 1)

    return result, c, yf, answers

# ---------------------------------------------------------------------------
# E10 diagnostic
# ---------------------------------------------------------------------------
def compute_e10(baseline_answers, baseline_correct, ft_answers, ft_correct,
                baseline_conf, ft_conf):
    """AUROC₂ on items where correctness is unchanged."""
    unchanged = []
    for i in range(len(baseline_answers)):
        if baseline_correct[i] == ft_correct[i]:
            unchanged.append(i)
    unchanged = np.array(unchanged)
    if len(unchanged) < 10:
        return {"n": len(unchanged), "status": "too_few"}

    bc = baseline_conf[unchanged]
    fc = ft_conf[unchanged]
    y = baseline_correct[unchanged]

    mask = ~np.isnan(bc) & ~np.isnan(fc)
    if mask.sum() < 10 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return {"n": len(unchanged), "status": "degenerate"}

    auc_base = float(roc_auc_score(y[mask], bc[mask]))
    auc_ft = float(roc_auc_score(y[mask], fc[mask]))

    return {
        "n": int(len(unchanged)),
        "auroc2_baseline": round(auc_base, 3),
        "auroc2_ft": round(auc_ft, 3),
        "delta": round(auc_ft - auc_base, 3),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    aliases = load_aliases()

    # --- Baseline ---
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    baseline_data = load_responses(BASELINE_PATH)
    baseline_result, baseline_conf, baseline_correct, baseline_answers = \
        score(baseline_data, aliases, "baseline")
    print_result(baseline_result)

    # --- Original PT-CSFT (rank 16, lr 2e-4) ---
    print("\n" + "=" * 60)
    print("ORIGINAL PT-CSFT (rank 16, lr 2e-4)")
    print("=" * 60)
    if os.path.exists(ORIGINAL_PATH):
        orig_data = load_responses(ORIGINAL_PATH)
        orig_result, orig_conf, orig_correct, orig_answers = \
            score(orig_data, aliases, "original_r16_lr2e4")
        print_result(orig_result)

        e10 = compute_e10(baseline_answers, baseline_correct,
                          orig_answers, orig_correct,
                          baseline_conf, orig_conf)
        print(f"  E10: {e10}")

        boot = paired_bootstrap_ci(orig_conf, baseline_conf, baseline_correct)
        print(f"  Bootstrap vs baseline: {boot}")
    else:
        print(f"  [skip] Not found: {ORIGINAL_PATH}")

    # --- Ablation configs ---
    for label in ABLATION_LABELS:
        print("\n" + "=" * 60)
        print(f"ABLATION: {label}")
        print("=" * 60)
        path = os.path.join(RESULTS_DIR, f"ablation_{label}_responses.json")
        if not os.path.exists(path):
            print(f"  [skip] Not found: {path}")
            continue

        abl_data = load_responses(path)
        abl_result, abl_conf, abl_correct, abl_answers = \
            score(abl_data, aliases, label)
        print_result(abl_result)

        # E10 vs baseline
        e10 = compute_e10(baseline_answers, baseline_correct,
                          abl_answers, abl_correct,
                          baseline_conf, abl_conf)
        print(f"  E10: {e10}")

        # Bootstrap vs baseline
        boot = paired_bootstrap_ci(abl_conf, baseline_conf, baseline_correct)
        print(f"  Bootstrap vs baseline: {boot}")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (flex matching)")
    print("=" * 60)
    print(f"{'Config':<20} {'Acc':>6} {'Drop':>7} {'AUROC₂':>7} {'Recov%':>7} "
          f"{'ECE':>6} {'Brier':>6} {'VRS':<15} {'Conf μ':>6} {'Conf σ':>6}")
    print("-" * 100)

    all_results = [baseline_result]
    if os.path.exists(ORIGINAL_PATH):
        all_results.append(orig_result)
    for label in ABLATION_LABELS:
        path = os.path.join(RESULTS_DIR, f"ablation_{label}_responses.json")
        if os.path.exists(path):
            abl_data = load_responses(path)
            r, _, _, _ = score(abl_data, aliases, label)
            all_results.append(r)

    base_acc = baseline_result["acc_flex"]
    for r in all_results:
        drop = r["acc_flex"] - base_acc
        rec = r.get("recovery_pct", "—")
        rec_s = f"{rec:.1f}" if isinstance(rec, float) else rec
        vrs_s = r["vrs_flex"]["status"]
        print(f"{r['label']:<20} {r['acc_flex']:>6.3f} {drop:>+7.3f} "
              f"{r['auroc2_flex']:>7.3f} {rec_s:>7} "
              f"{r['ece_flex']:>6.3f} {r['brier_flex']:>6.3f} "
              f"{vrs_s:<15} {r['conf_mean']:>6.1f} {r['conf_std']:>6.1f}")

    # Save full results
    outpath = os.path.join(RESULTS_DIR, "ablation_comprehensive_rescore.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {outpath}")


def print_result(r):
    print(f"  Accuracy:  exact={r['acc_exact']:.3f}  flex={r['acc_flex']:.3f}")
    print(f"  AUROC₂:    exact={r['auroc2_exact']:.3f}  flex={r['auroc2_flex']:.3f}")
    print(f"  VRS:       {r['vrs_flex']}")
    print(f"  ECE:       {r['ece_flex']:.3f}")
    print(f"  Brier:     {r['brier_flex']:.3f}")
    print(f"  Conf:      mean={r['conf_mean']}, std={r['conf_std']}")
    print(f"  Parse:     {r['parse_rate']:.3f}")
    rec = r.get("recovery_pct")
    if rec is not None:
        print(f"  Recovery:  {rec:.1f}%")


if __name__ == "__main__":
    main()
