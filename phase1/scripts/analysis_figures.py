"""
analysis_figures.py — Three key analyses for the PT-CSFT paper.

1. Selective prediction curves: coverage vs accuracy for each confidence method
2. Reliability diagram: logit EV vs actual P(correct), with smECE
3. Token entropy comparison: AUROC₂ of entropy vs logit EV vs text confidence

Reads per-item logit response data from all 10 gentle E10 seeds.
Generates publication-quality matplotlib figures.

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/analysis_figures.py

Output:
    results_raw/domain_gen/e10_gentle/figures/
        fig_selective_prediction.pdf
        fig_reliability_diagram.pdf
        fig_entropy_comparison.pdf
        fig_bar_progression.pdf
        analysis_summary.json
"""

import json, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1234, 5678, 9012, 3456, 7890, 2468]
RESULTS_DIR = Path("results_raw/domain_gen/e10_gentle")
BASELINE_PATH = Path("results_raw/domain_gen/responses_gsm8k_gemma-3-12b-it.json")
FIGURE_DIR = RESULTS_DIR / "figures"

# Known values from prior experiments
PROBE_AUROC2 = 0.769
BASELINE_VERBAL_AUROC2 = 0.546

# Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'baseline': '#999999',
    'text': '#e07b39',
    'logit': '#2166ac',
    'entropy': '#4daf4a',
    'probe': '#984ea3',
    'random': '#cccccc',
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_seeds():
    """Load per-item data from all seed response files."""
    all_items = []
    seed_items = {}

    for seed in SEEDS:
        logit_path = RESULTS_DIR / f"e2e_ce_gentle_logit_seed{seed}_responses.json"
        if not logit_path.exists():
            print(f"  Warning: {logit_path} not found")
            continue

        with open(logit_path) as f:
            items = json.load(f)

        for item in items:
            item['seed'] = seed
        seed_items[seed] = items
        all_items.extend(items)
        print(f"  Seed {seed}: {len(items)} items, "
              f"acc={np.mean([i['correct'] for i in items]):.3f}, "
              f"logit_mean={np.mean([i['logit_confidence'] for i in items]):.1f}")

    print(f"\n  Total: {len(all_items)} items across {len(seed_items)} seeds")
    return all_items, seed_items


def load_baseline():
    """Load baseline (no adapter) responses if available."""
    if not BASELINE_PATH.exists():
        print(f"  Baseline not found at {BASELINE_PATH}")
        return None

    with open(BASELINE_PATH) as f:
        items = json.load(f)

    print(f"  Baseline: {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auroc2(confidence, correct):
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c) & ~np.isinf(c)
    if mask.sum() < 2 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))


def token_entropy(digit_probs):
    """Compute Shannon entropy from digit probability distribution."""
    p = np.asarray(digit_probs, dtype=float)
    p = p[p > 0]  # avoid log(0)
    return -np.sum(p * np.log2(p))


def selective_prediction(confidence, correct, n_thresholds=200):
    """Compute selective prediction curve.

    Returns:
        coverages: fraction of items retained at each threshold
        accuracies: accuracy on retained items at each threshold
        thresholds: the thresholds used
    """
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)

    thresholds = np.linspace(np.min(c) - 0.01, np.max(c) + 0.01, n_thresholds)
    coverages = []
    accuracies = []
    valid_thresholds = []

    for t in thresholds:
        mask = c >= t
        if mask.sum() == 0:
            continue
        coverage = mask.sum() / len(y)
        acc = y[mask].mean()
        coverages.append(coverage)
        accuracies.append(acc)
        valid_thresholds.append(t)

    return np.array(coverages), np.array(accuracies), np.array(valid_thresholds)


def compute_ece(confidence, correct, n_bins=10):
    """Expected Calibration Error."""
    c = np.asarray(confidence, dtype=float) / 100.0  # normalise to 0-1
    y = np.asarray(correct, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (c >= bin_edges[i]) & (c < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = (c >= bin_edges[i]) & (c <= bin_edges[i + 1])
        if mask.sum() == 0:
            bin_data.append(None)
            continue
        bin_conf = c[mask].mean()
        bin_acc = y[mask].mean()
        bin_n = mask.sum()
        ece += (bin_n / len(y)) * abs(bin_acc - bin_conf)
        bin_data.append({
            'conf_mean': float(bin_conf),
            'acc_mean': float(bin_acc),
            'n': int(bin_n),
            'gap': float(bin_acc - bin_conf),
        })

    return float(ece), bin_data


def compute_smece(confidence, correct, n_bins=15):
    """Smooth ECE (adaptive binning)."""
    c = np.asarray(confidence, dtype=float) / 100.0
    y = np.asarray(correct, dtype=float)
    n = len(y)

    # Sort by confidence
    order = np.argsort(c)
    c_sorted = c[order]
    y_sorted = y[order]

    # Equal-count bins
    bin_size = n // n_bins
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        bc = c_sorted[start:end]
        by = y_sorted[start:end]
        if len(bc) == 0:
            continue
        bin_conf = bc.mean()
        bin_acc = by.mean()
        ece += (len(bc) / n) * abs(bin_acc - bin_conf)
        bin_data.append({
            'conf_mean': float(bin_conf),
            'acc_mean': float(bin_acc),
            'n': len(bc),
        })

    return float(ece), bin_data


# ---------------------------------------------------------------------------
# Figure 1: Selective Prediction Curves
# ---------------------------------------------------------------------------
def fig_selective_prediction(all_items, baseline_items=None):
    """Coverage vs accuracy for each confidence method."""
    print("\n--- Figure 1: Selective Prediction ---")

    correct = np.array([int(i['correct']) for i in all_items])
    logit_conf = np.array([i['logit_confidence'] for i in all_items])
    text_conf = np.array([i['text_confidence'] for i in all_items])

    # Token entropy (negated so higher = more confident)
    entropies = np.array([token_entropy(i['digit_probs']) for i in all_items])
    neg_entropy = -entropies  # higher = more confident

    # Compute curves
    cov_logit, acc_logit, _ = selective_prediction(logit_conf, correct)
    cov_text, acc_text, _ = selective_prediction(text_conf, correct)
    cov_ent, acc_ent, _ = selective_prediction(neg_entropy, correct)

    # Overall accuracy (no abstention)
    overall_acc = correct.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left panel: Coverage vs Accuracy ---
    ax1.plot(cov_logit, acc_logit, color=COLORS['logit'], linewidth=2,
             label=f'Logit EV (AUROC₂={auroc2(logit_conf, correct):.3f})')
    ax1.plot(cov_text, acc_text, color=COLORS['text'], linewidth=2,
             label=f'Argmax text (AUROC₂={auroc2(text_conf, correct):.3f})')
    ax1.plot(cov_ent, acc_ent, color=COLORS['entropy'], linewidth=2, linestyle='--',
             label=f'Neg. entropy (AUROC₂={auroc2(neg_entropy, correct):.3f})')

    ax1.axhline(y=overall_acc, color=COLORS['random'], linestyle=':', linewidth=1,
                label=f'No abstention ({overall_acc:.3f})')
    ax1.axhline(y=0.95, color='#cc0000', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.text(0.15, 0.953, '95% target', fontsize=8, color='#cc0000', alpha=0.7)

    ax1.set_xlabel('Coverage (fraction of items retained)')
    ax1.set_ylabel('Accuracy on retained items')
    ax1.set_title('Selective Prediction: Coverage vs Accuracy')
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(max(0.5, overall_acc - 0.15), 1.02)
    ax1.grid(True, alpha=0.2)

    # --- Right panel: Utility Tax ---
    # At each target accuracy, what fraction of correct items must we discard?
    targets = np.arange(0.80, 1.001, 0.01)

    def utility_at_target(coverages, accuracies, correct_arr, target):
        """At a given accuracy target, what coverage is needed? What fraction of
        correct items are discarded?"""
        # Find the highest coverage that achieves the target accuracy
        for i in range(len(coverages)):
            if accuracies[i] >= target:
                cov = coverages[i]
                # Utility = fraction of correct items retained
                # At this coverage, we retain cov * len items
                n_retained = int(cov * len(correct_arr))
                # Approximate: retained items have accuracy = accuracies[i]
                # Total correct in full set
                total_correct = correct_arr.sum()
                # Correct items retained ≈ n_retained * accuracies[i]
                retained_correct = n_retained * accuracies[i]
                utility = retained_correct / total_correct if total_correct > 0 else 0
                return cov, utility
        return 0.0, 0.0

    util_logit = [utility_at_target(cov_logit, acc_logit, correct, t) for t in targets]
    util_text = [utility_at_target(cov_text, acc_text, correct, t) for t in targets]
    util_ent = [utility_at_target(cov_ent, acc_ent, correct, t) for t in targets]

    error_targets = 1 - targets  # convert accuracy to error rate

    ax2.plot(error_targets * 100, [1 - u[1] for u in util_logit],
             color=COLORS['logit'], linewidth=2, label='Logit EV')
    ax2.plot(error_targets * 100, [1 - u[1] for u in util_text],
             color=COLORS['text'], linewidth=2, label='Argmax text')
    ax2.plot(error_targets * 100, [1 - u[1] for u in util_ent],
             color=COLORS['entropy'], linewidth=2, linestyle='--', label='Neg. entropy')

    ax2.set_xlabel('Target error rate (%)')
    ax2.set_ylabel('Utility tax (fraction of correct answers lost)')
    ax2.set_title('Utility-Error Trade-off')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig_selective_prediction.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {out_path}")

    # Print key numbers
    for name, conf in [("Logit EV", logit_conf), ("Text", text_conf), ("Neg entropy", neg_entropy)]:
        cov, acc, _ = selective_prediction(conf, correct)
        # Find coverage at 95% accuracy
        idx_95 = np.where(acc >= 0.95)[0]
        if len(idx_95) > 0:
            cov_95 = cov[idx_95[0]]
            tax_95 = 1 - cov_95
            print(f"  {name}: at 95% accuracy, coverage={cov_95:.3f}, tax={tax_95:.3f}")
        else:
            print(f"  {name}: cannot reach 95% accuracy")

    return {
        "logit_auroc2": auroc2(logit_conf, correct),
        "text_auroc2": auroc2(text_conf, correct),
        "entropy_auroc2": auroc2(neg_entropy, correct),
    }


# ---------------------------------------------------------------------------
# Figure 2: Reliability Diagram
# ---------------------------------------------------------------------------
def fig_reliability_diagram(all_items):
    """Logit EV vs actual P(correct) with ECE."""
    print("\n--- Figure 2: Reliability Diagram ---")

    correct = np.array([int(i['correct']) for i in all_items])
    logit_conf = np.array([i['logit_confidence'] for i in all_items])

    # Compute ECE (equal-width bins)
    ece, bin_data_ew = compute_ece(logit_conf, correct, n_bins=10)
    smece, bin_data_sm = compute_smece(logit_conf, correct, n_bins=15)

    print(f"  ECE (equal-width, 10 bins): {ece:.4f}")
    print(f"  smECE (equal-count, 15 bins): {smece:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: Standard reliability diagram ---
    # Equal-width bins
    conf_bins = []
    acc_bins = []
    n_bins_actual = []
    for bd in bin_data_ew:
        if bd is not None:
            conf_bins.append(bd['conf_mean'])
            acc_bins.append(bd['acc_mean'])
            n_bins_actual.append(bd['n'])

    conf_bins = np.array(conf_bins)
    acc_bins = np.array(acc_bins)
    n_bins_actual = np.array(n_bins_actual)

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect calibration')

    # Bar chart for bin counts (background)
    bin_width = 0.1
    for cb, ab, n in zip(conf_bins, acc_bins, n_bins_actual):
        gap_color = '#2166ac' if ab >= cb else '#e07b39'
        ax1.bar(cb, ab, width=bin_width * 0.8, alpha=0.6, color=gap_color,
                edgecolor='white', linewidth=0.5)

    # Line connecting bin centers
    ax1.plot(conf_bins, acc_bins, 'o-', color=COLORS['logit'], linewidth=1.5,
             markersize=5, label=f'Logit EV (ECE={ece:.3f})')

    ax1.set_xlabel('Mean predicted confidence')
    ax1.set_ylabel('Fraction correct')
    ax1.set_title('Reliability Diagram (Logit Expected Value)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # Add ECE annotation
    ax1.text(0.95, 0.05, f'ECE = {ece:.4f}\nsmECE = {smece:.4f}\nn = {len(correct)}',
             transform=ax1.transAxes, fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # --- Right: Histogram of logit confidence by correctness ---
    correct_confs = logit_conf[correct == 1] / 100.0
    incorrect_confs = logit_conf[correct == 0] / 100.0

    bins = np.linspace(0, 1, 25)
    ax2.hist(correct_confs, bins=bins, alpha=0.6, color='#2166ac',
             label=f'Correct (n={len(correct_confs)})', density=True)
    ax2.hist(incorrect_confs, bins=bins, alpha=0.6, color='#e07b39',
             label=f'Incorrect (n={len(incorrect_confs)})', density=True)

    ax2.set_xlabel('Logit confidence (normalised)')
    ax2.set_ylabel('Density')
    ax2.set_title('Confidence Distribution by Correctness')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig_reliability_diagram.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "ece": ece,
        "smece": smece,
        "bin_data": bin_data_ew,
    }


# ---------------------------------------------------------------------------
# Figure 3: Entropy Comparison
# ---------------------------------------------------------------------------
def fig_entropy_comparison(all_items):
    """Compare AUROC₂ of different confidence signals."""
    print("\n--- Figure 3: Entropy vs Logit Comparison ---")

    correct = np.array([int(i['correct']) for i in all_items])
    logit_conf = np.array([i['logit_confidence'] for i in all_items])
    text_conf = np.array([i['text_confidence'] for i in all_items])

    # Compute entropies
    entropies = np.array([token_entropy(i['digit_probs']) for i in all_items])
    neg_entropy = -entropies

    # Max probability (argmax softmax value)
    max_probs = np.array([max(i['digit_probs']) for i in all_items])

    # Margin (top1 - top2 probability)
    margins = []
    for item in all_items:
        probs = sorted(item['digit_probs'], reverse=True)
        margins.append(probs[0] - probs[1])
    margins = np.array(margins)

    # Compute AUROC₂ for each method
    methods = {
        'Logit EV': logit_conf,
        'Argmax digit (text)': text_conf,
        'Neg. entropy': neg_entropy,
        'Max probability': max_probs,
        'Softmax margin': margins,
    }

    print("\n  Method AUROC₂ comparison:")
    aurocs = {}
    for name, conf in methods.items():
        auc = auroc2(conf, correct)
        aurocs[name] = auc
        print(f"    {name:25s}: {auc:.4f}")

    # Also compute per-seed AUROC₂ for logit and entropy
    seed_logit = []
    seed_entropy = []
    seed_text = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"e2e_ce_gentle_logit_seed{seed}_responses.json"
        if not path.exists():
            continue
        with open(path) as f:
            items = json.load(f)
        c = np.array([int(i['correct']) for i in items])
        lc = np.array([i['logit_confidence'] for i in items])
        tc = np.array([i['text_confidence'] for i in items])
        ent = np.array([-token_entropy(i['digit_probs']) for i in items])
        seed_logit.append(auroc2(lc, c))
        seed_entropy.append(auroc2(ent, c))
        seed_text.append(auroc2(tc, c))

    print(f"\n  Per-seed AUROC₂:")
    print(f"    Logit EV:    {np.mean(seed_logit):.3f} ± {np.std(seed_logit, ddof=1):.3f}")
    print(f"    Neg entropy: {np.mean(seed_entropy):.3f} ± {np.std(seed_entropy, ddof=1):.3f}")
    print(f"    Text:        {np.mean(seed_text):.3f} ± {np.std(seed_text, ddof=1):.3f}")

    # --- Bar chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Bar chart of AUROC₂ by method
    method_names = list(aurocs.keys())
    method_vals = [aurocs[m] for m in method_names]
    method_colors = [COLORS['logit'], COLORS['text'], COLORS['entropy'],
                     '#666666', '#888888']

    bars = ax1.barh(range(len(method_names)), method_vals,
                    color=method_colors, edgecolor='white', height=0.6)

    # Add probe and baseline reference lines
    ax1.axvline(x=PROBE_AUROC2, color=COLORS['probe'], linestyle='--',
                linewidth=1.5, label=f'Probe ({PROBE_AUROC2:.3f})')
    ax1.axvline(x=0.5, color=COLORS['random'], linestyle=':', linewidth=1,
                label='Random (0.500)')

    ax1.set_yticks(range(len(method_names)))
    ax1.set_yticklabels(method_names)
    ax1.set_xlabel('AUROC₂')
    ax1.set_title('Discrimination by Confidence Method')
    ax1.set_xlim(0.4, 1.0)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, axis='x', alpha=0.2)

    # Add value labels
    for bar, val in zip(bars, method_vals):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=9)

    # Right: Scatter of logit EV vs entropy per item
    ax2.scatter(entropies[correct == 1], logit_conf[correct == 1] / 100,
                alpha=0.15, s=8, color='#2166ac', label='Correct')
    ax2.scatter(entropies[correct == 0], logit_conf[correct == 0] / 100,
                alpha=0.3, s=8, color='#e07b39', label='Incorrect')

    ax2.set_xlabel('Token entropy (bits)')
    ax2.set_ylabel('Logit expected value')
    ax2.set_title('Entropy vs Logit Confidence')
    ax2.legend(loc='upper right', framealpha=0.9, markerscale=3)
    ax2.grid(True, alpha=0.2)

    # Compute correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(-entropies, logit_conf)
    ax2.text(0.05, 0.05, f'r = {r:.3f}', transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig_entropy_comparison.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "aurocs": aurocs,
        "per_seed_logit": seed_logit,
        "per_seed_entropy": seed_entropy,
        "per_seed_text": seed_text,
        "entropy_logit_correlation": {"r": float(r), "p": float(p)},
    }


# ---------------------------------------------------------------------------
# Figure 4: Bar Chart Progression
# ---------------------------------------------------------------------------
def fig_bar_progression(all_items):
    """Baseline → confonly → E2E argmax → E2E logit → probe progression."""
    print("\n--- Figure 4: Progression Bar Chart ---")

    correct = np.array([int(i['correct']) for i in all_items])
    logit_conf = np.array([i['logit_confidence'] for i in all_items])
    text_conf = np.array([i['text_confidence'] for i in all_items])
    neg_ent = np.array([-token_entropy(i['digit_probs']) for i in all_items])

    # Known values
    stages = {
        'Baseline\nverbal': BASELINE_VERBAL_AUROC2,
        'E2E\nargmax': auroc2(text_conf, correct),
        'E2E\nentropy': auroc2(neg_ent, correct),
        'Probe\n(hidden)': PROBE_AUROC2,
        'E2E\nlogit': auroc2(logit_conf, correct),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = list(stages.keys())
    vals = list(stages.values())
    colors = [COLORS['baseline'], COLORS['text'], COLORS['entropy'],
              COLORS['probe'], COLORS['logit']]

    bars = ax.bar(range(len(names)), vals, color=colors,
                  edgecolor='white', width=0.65)

    # Value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Reference lines
    ax.axhline(y=0.5, color=COLORS['random'], linestyle=':', linewidth=1, alpha=0.5)
    ax.text(len(names) - 0.5, 0.51, 'chance', fontsize=8, color='#999', alpha=0.7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('AUROC₂')
    ax.set_title('Metacognitive Discrimination: Method Progression (GSM8K, Gemma 12B)')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig_bar_progression.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Diagnostic digit distributions
# ---------------------------------------------------------------------------
def fig_digit_distributions(all_items):
    """P(digit|correct) vs P(digit|incorrect) showing graded logit signal."""
    print("\n--- Figure 5: Digit Probability Distributions ---")

    correct_items = [i for i in all_items if i['correct']]
    incorrect_items = [i for i in all_items if not i['correct']]

    correct_probs = np.array([i['digit_probs'] for i in correct_items])
    incorrect_probs = np.array([i['digit_probs'] for i in incorrect_items])

    mean_correct = correct_probs.mean(axis=0)
    mean_incorrect = incorrect_probs.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    digits = np.arange(10)
    width = 0.35

    # Left: Mean probability per digit
    ax1.bar(digits - width / 2, mean_correct, width, color='#2166ac',
            alpha=0.8, label=f'Correct (n={len(correct_items)})')
    ax1.bar(digits + width / 2, mean_incorrect, width, color='#e07b39',
            alpha=0.8, label=f'Incorrect (n={len(incorrect_items)})')

    ax1.set_xlabel('Confidence digit')
    ax1.set_ylabel('Mean probability')
    ax1.set_title('Mean Digit Probability by Correctness')
    ax1.set_xticks(digits)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, axis='y', alpha=0.2)

    # Right: Log scale to see the tails
    ax2.bar(digits - width / 2, mean_correct, width, color='#2166ac',
            alpha=0.8, label='Correct')
    ax2.bar(digits + width / 2, mean_incorrect, width, color='#e07b39',
            alpha=0.8, label='Incorrect')

    ax2.set_xlabel('Confidence digit')
    ax2.set_ylabel('Mean probability (log scale)')
    ax2.set_title('Digit Probabilities (Log Scale)')
    ax2.set_xticks(digits)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-6, 1.0)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.2)

    # Print the key numbers
    print(f"  P(digit=9 | correct):   {mean_correct[9]:.4f}")
    print(f"  P(digit=9 | incorrect): {mean_incorrect[9]:.4f}")
    print(f"  P(digit=0 | correct):   {mean_correct[0]:.4f}")
    print(f"  P(digit=0 | incorrect): {mean_incorrect[0]:.4f}")
    print(f"  Ratio (digit 9): {mean_correct[9] / mean_incorrect[9]:.2f}x")
    print(f"  Ratio (digit 0): {mean_incorrect[0] / mean_correct[0]:.2f}x")

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig_digit_distributions.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PT-CSFT Analysis Suite")
    print("=" * 60)

    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Load data
    print("\n--- Loading Data ---")
    all_items, seed_items = load_all_seeds()
    baseline_items = load_baseline()

    if not all_items:
        print("ERROR: No per-item data found")
        sys.exit(1)

    # Run all analyses
    results = {}

    results['selective_prediction'] = fig_selective_prediction(all_items, baseline_items)
    results['reliability'] = fig_reliability_diagram(all_items)
    results['entropy'] = fig_entropy_comparison(all_items)
    fig_bar_progression(all_items)
    fig_digit_distributions(all_items)

    # Save summary
    summary_path = FIGURE_DIR / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # Paper-ready summary
    print("\n" + "=" * 60)
    print("Paper-Ready Numbers:")
    print("=" * 60)

    r = results['reliability']
    print(f"  Logit EV ECE:  {r['ece']:.4f}")
    print(f"  Logit EV smECE: {r['smece']:.4f}")

    e = results['entropy']
    for name, auc in e['aurocs'].items():
        print(f"  {name:25s} AUROC₂ = {auc:.4f}")
    print(f"  Logit > entropy by: {e['aurocs']['Logit EV'] - e['aurocs']['Neg. entropy']:+.4f}")
    print(f"  Logit > max prob by: {e['aurocs']['Logit EV'] - e['aurocs']['Max probability']:+.4f}")


if __name__ == "__main__":
    main()
