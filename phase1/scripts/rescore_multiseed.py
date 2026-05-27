#!/usr/bin/env python3
"""
Rescore multiseed PT-CSFT response files with flex matching.

eval_ablation.py uses exact string matching, which gives near-zero accuracy
on PT-CSFT models that produce full-sentence answers. This script applies
bidirectional substring matching (flex) and recomputes AUROC2.

Usage:
    python3 rescore_multiseed.py

Outputs per-seed and aggregate statistics per model.
"""

import json, glob, os, numpy as np, random
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

# Load TriviaQA aliases for T-eval (first 1000 items, seed 42 shuffle)
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


# Find all multiseed response files
results_dir = os.path.expanduser(
    "~/jpwork/metacog-engineering/phase1/results_raw/step4"
)
files = sorted(glob.glob(f"{results_dir}/ablation_multiseed_*_responses.json"))

if not files:
    print("No multiseed response files found.")
    print(f"Looked in: {results_dir}")
    exit(1)

print(f"Found {len(files)} response files\n")

# Known baselines for recovery calculation
BASELINES = {
    "llama8b": {"auroc2": 0.704, "probe": 0.843, "acc": 0.775},
    "mistral7b": {"auroc2": 0.583, "probe": 0.762, "acc": 0.729},
    "qwen7b": {"auroc2": 0.577, "probe": 0.762, "acc": 0.655},
}

# Group by model
by_model = {}

header = f"{'Label':<48} {'Acc':>6} {'Drop':>7} {'AUROC2':>8} {'Recov%':>7} {'Conf_m':>7} {'Conf_s':>7} {'N':>5}"
print(header)
print("-" * len(header))

for f in files:
    label = os.path.basename(f).replace("ablation_", "").replace("_responses.json", "")
    with open(f) as fh:
        responses = json.load(fh)

    correct, confs = [], []
    for i, resp in enumerate(responses[:1000]):
        if i >= len(aliases_list):
            break
        c = resp.get("confidence", None)
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        correct.append(int(flex_correct(resp.get("parsed_answer", ""), aliases_list[i])))
        confs.append(float(c))

    correct, confs = np.array(correct), np.array(confs)
    acc = correct.mean()
    auroc = roc_auc_score(correct, confs) if len(set(correct)) > 1 else 0.5

    # Determine model key
    if "llama8b" in label:
        model_key = "llama8b"
    elif "mistral7b" in label:
        model_key = "mistral7b"
    elif "qwen7b" in label:
        model_key = "qwen7b"
    else:
        model_key = "unknown"

    bl = BASELINES.get(model_key, {})
    probe = bl.get("probe", 0.8)
    recovery = (auroc - 0.5) / (probe - 0.5) * 100 if probe > 0.5 else 0
    acc_drop = acc - bl.get("acc", acc)

    print(
        f"{label:<48} {acc:>6.3f} {acc_drop:>+7.3f} {auroc:>8.3f} {recovery:>6.1f}% "
        f"{confs.mean():>7.1f} {confs.std():>7.1f} {len(correct):>5d}"
    )

    if model_key not in by_model:
        by_model[model_key] = []
    by_model[model_key].append({
        "seed": label.split("seed")[-1] if "seed" in label else "42",
        "acc": float(acc),
        "auroc2": float(auroc),
        "conf_mean": float(confs.mean()),
        "conf_std": float(confs.std()),
        "n": len(correct),
    })

# Aggregate per model
print(f"\n{'='*60}")
print("AGGREGATE (for paper Table 5)")
print(f"{'='*60}\n")

for model_key in sorted(by_model.keys()):
    seeds = by_model[model_key]
    aurocs = np.array([s["auroc2"] for s in seeds])
    accs = np.array([s["acc"] for s in seeds])
    bl = BASELINES.get(model_key, {})

    print(f"{model_key} ({len(seeds)} seeds):")
    print(f"  AUROC2 = {aurocs.mean():.3f} +/- {aurocs.std():.3f}  "
          f"[{aurocs.min():.3f}, {aurocs.max():.3f}]")
    print(f"  Acc    = {accs.mean():.3f} +/- {accs.std():.3f}  "
          f"(drop: {accs.mean() - bl.get('acc', accs.mean()):+.3f})")
    recovery = (aurocs.mean() - 0.5) / (bl.get("probe", 0.8) - 0.5) * 100
    print(f"  Recovery = {recovery:.1f}%")
    print()

# Save
output = {
    "per_model": by_model,
    "aggregates": {},
}
for model_key in by_model:
    aurocs = [s["auroc2"] for s in by_model[model_key]]
    accs = [s["acc"] for s in by_model[model_key]]
    output["aggregates"][model_key] = {
        "auroc2_mean": float(np.mean(aurocs)),
        "auroc2_std": float(np.std(aurocs)),
        "auroc2_min": float(np.min(aurocs)),
        "auroc2_max": float(np.max(aurocs)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "n_seeds": len(aurocs),
    }

out_path = os.path.join(results_dir, "multiseed_rescore_summary.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: {out_path}")
