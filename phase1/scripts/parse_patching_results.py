#!/usr/bin/env python3
"""
Parse activation patching sweep results and compute per-layer statistics.
Produces the summary table and gradient analysis for §8.5.

Usage:
    python3 parse_patching_results.py [--json-path PATH]

Defaults to the Llama 8B gentle-lr all_sweep output.
"""

import argparse, json, os
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--json-path", default=os.path.expanduser(
    "~/jpwork/metacog-engineering/phase1/results_raw/finetune/"
    "Meta-Llama-3.1-8B-Instruct-bf16/ablation_gentle_lr/"
    "activation_patching_all_sweep.json"
))
args = parser.parse_args()

with open(args.json_path) as f:
    results = json.load(f)

print(f"Loaded {len(results)} results from {os.path.basename(args.json_path)}")

# Group by layer
by_layer = {}
for r in results:
    layer = r["layer"]
    if layer not in by_layer:
        by_layer[layer] = []
    by_layer[layer].append(r)

# Per-layer stats
print(f"\n{'Layer':<8} {'N':<6} {'Mean':>8} {'Median':>8} {'Std':>8} "
      f"{'Frac>0':>8} {'Frac>.5':>8} {'p':>10}")
print("-" * 70)

layer_ids, layer_means, layer_frac_pos = [], [], []

for layer in sorted(by_layer.keys()):
    shifts = [r["shift"] for r in by_layer[layer] if r["shift"] is not None]
    s = np.clip(shifts, -5, 5)
    n_pos = int((s > 0).sum())
    p = stats.binomtest(n_pos, len(s), 0.5, alternative="greater").pvalue
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    layer_ids.append(layer)
    layer_means.append(np.mean(s))
    layer_frac_pos.append((s > 0).mean())

    print(f"{layer:<8} {len(s):<6} {np.mean(s):>+8.3f} {np.median(s):>+8.3f} "
          f"{np.std(s):>8.3f} {(s>0).mean():>8.1%} {(s>0.5).mean():>8.1%} "
          f"{p:>9.4f} {sig}")

# Gradient
rho, p_rho = stats.spearmanr(layer_ids, layer_means)
rho_f, p_f = stats.spearmanr(layer_ids, layer_frac_pos)

early = np.clip([r["shift"] for l in sorted(by_layer) if l <= max(layer_ids)//4
                 for r in by_layer[l] if r["shift"] is not None], -5, 5)
late = np.clip([r["shift"] for l in sorted(by_layer) if l >= 3*max(layer_ids)//4
                for r in by_layer[l] if r["shift"] is not None], -5, 5)

u_stat, u_p = stats.mannwhitneyu(late, early, alternative="greater")

print(f"\nSpearman rho (layer vs mean):   {rho:.3f} (p={p_rho:.6f})")
print(f"Spearman rho (layer vs frac>0): {rho_f:.3f} (p={p_f:.6f})")
print(f"Early (0-{max(layer_ids)//4}):  mean={np.mean(early):.3f}, frac>0={(early>0).mean():.1%}")
print(f"Late ({3*max(layer_ids)//4}-{max(layer_ids)}): mean={np.mean(late):.3f}, frac>0={(late>0).mean():.1%}")
print(f"Mann-Whitney (late > early): U={u_stat:.0f}, p={u_p:.6f}")

# Save
summary = {
    "per_layer": {str(l): {"mean": float(np.mean(np.clip([r["shift"] for r in by_layer[l] if r["shift"] is not None], -5, 5))),
                           "frac_pos": float((np.clip([r["shift"] for r in by_layer[l] if r["shift"] is not None], -5, 5) > 0).mean()),
                           "n": len([r for r in by_layer[l] if r["shift"] is not None])}
                  for l in sorted(by_layer)},
    "gradient": {"spearman_rho": float(rho), "spearman_p": float(p_rho),
                 "mann_whitney_p": float(u_p)},
}
out = args.json_path.replace(".json", "_summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out}")
