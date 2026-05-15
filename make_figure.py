"""Generate figures for the Phase 1 paper.

Run on the Mac:
    pip install matplotlib
    python3 make_figures.py

Outputs to ~/jpwork/repo/figures/phase1/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.expanduser("~/jpwork/repo/figures/phase1")
os.makedirs(OUT, exist_ok=True)

# Use a clean style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

# ================================================================
# Figure 1: Main AUROC₂ results (baseline vs PT-CSFT)
# ================================================================
def fig1_auroc_comparison():
    models = ['Gemma\n12B', 'Gemma\n27B', 'Qwen\n7B', 'Llama\n8B', 'Llama\n70B', 'Mistral\n7B']
    baseline = [0.684, 0.717, 0.581, 0.711, 0.760, 0.584]
    ptcsft =   [0.842, 0.785, 0.754, 0.513, 0.582, None]  # Mistral diverged
    probe =    [0.857, 0.807, 0.762, 0.843, 0.803, 0.762]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.25

    ax.bar(x - w, baseline, w, label='Baseline verbal', color='#bdbdbd', edgecolor='white')
    ptcsft_plot = [v if v is not None else 0 for v in ptcsft]
    colors = ['#2196F3' if i < 3 else '#f44336' for i in range(len(models))]
    bars = ax.bar(x, ptcsft_plot, w, label='PT-CSFT verbal', color=colors, edgecolor='white')
    ax.bar(x + w, probe, w, label='Probe (upper ref.)', color='#e0e0e0', edgecolor='#999',
           linestyle='--', linewidth=0.8)

    # Mark Mistral as diverged
    ax.text(5, 0.52, '×\ndiverged', ha='center', va='bottom', fontsize=8, color='#f44336')

    ax.axhline(y=0.5, color='#999', linestyle=':', linewidth=0.8, label='Chance')
    ax.set_ylabel('AUROC₂')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.4, 0.95)
    ax.legend(loc='upper left', frameon=False, fontsize=9)

    # Architecture class labels
    ax.annotate('GemmaForCausalLM', xy=(0.5, 0.42), fontsize=8, color='#2196F3',
                ha='center', style='italic')
    ax.annotate('Qwen2ForCausalLM', xy=(2, 0.42), fontsize=8, color='#2196F3',
                ha='center', style='italic')
    ax.annotate('LlamaForCausalLM', xy=(4, 0.42), fontsize=8, color='#f44336',
                ha='center', style='italic')

    ax.set_title('PT-CSFT verbal confidence discrimination by model', fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_auroc_comparison.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig1_auroc_comparison.pdf', bbox_inches='tight')
    print(f'  [save] fig1_auroc_comparison')
    plt.close()


# ================================================================
# Figure 2: ECE + Brier (before/after)
# ================================================================
def fig2_calibration_metrics():
    models = ['Gemma 12B', 'Qwen 7B', 'Llama 8B']

    ece_base = [0.230, 0.312, 0.154]
    ece_pt =   [0.108, 0.068, 0.276]
    brier_base = [0.234, 0.312, 0.166]
    brier_pt =   [0.174, 0.193, 0.292]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(models))
    w = 0.3

    # ECE
    ax1.bar(x - w/2, ece_base, w, label='Baseline', color='#bdbdbd', edgecolor='white')
    colors = ['#2196F3', '#2196F3', '#f44336']
    ax1.bar(x + w/2, ece_pt, w, label='PT-CSFT', color=colors, edgecolor='white')
    ax1.set_ylabel('ECE (lower = better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title('Expected Calibration Error', fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_ylim(0, 0.38)

    # Add delta labels
    for i in range(len(models)):
        delta = ece_pt[i] - ece_base[i]
        color = '#2196F3' if delta < 0 else '#f44336'
        label = f'{delta:+.0%}' if abs(delta) > 0.01 else ''
        pct = (ece_pt[i] - ece_base[i]) / ece_base[i] * 100
        ax1.text(i + w/2, ece_pt[i] + 0.01, f'{pct:+.0f}%', ha='center', fontsize=8, color=color)

    # Brier
    ax2.bar(x - w/2, brier_base, w, label='Baseline', color='#bdbdbd', edgecolor='white')
    ax2.bar(x + w/2, brier_pt, w, label='PT-CSFT', color=colors, edgecolor='white')
    ax2.set_ylabel('Brier Score (lower = better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_title('Brier Score', fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_ylim(0, 0.38)

    for i in range(len(models)):
        pct = (brier_pt[i] - brier_base[i]) / brier_base[i] * 100
        color = '#2196F3' if pct < 0 else '#f44336'
        ax2.text(i + w/2, brier_pt[i] + 0.01, f'{pct:+.0f}%', ha='center', fontsize=8, color=color)

    fig.suptitle('Calibration metrics: baseline vs PT-CSFT', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_calibration_metrics.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig2_calibration_metrics.pdf', bbox_inches='tight')
    print(f'  [save] fig2_calibration_metrics')
    plt.close()


# ================================================================
# Figure 3: Retrained probes (baseline CV vs post-PT CV)
# ================================================================
def fig3_retrained_probes():
    configs = ['first\npre', 'first\nlast', 'middle\npre', 'middle\nlast', 'last\npre', 'last\nlast']

    # Gemma 12B
    gemma_base = [0.691, 0.708, 0.822, 0.820, 0.775, 0.807]
    gemma_post = [0.660, 0.686, 0.843, 0.844, 0.819, 0.844]

    # Qwen 7B
    qwen_base = [0.640, 0.732, 0.714, 0.685, 0.718, 0.728]
    qwen_post = [0.618, 0.645, 0.679, 0.692, 0.706, 0.658]

    # Llama 8B
    llama_base = [0.680, 0.646, 0.831, 0.828, 0.765, 0.806]
    llama_post = [0.645, 0.568, 0.828, 0.778, 0.780, 0.749]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, name, base, post, color in [
        (axes[0], 'Gemma 12B (success)', gemma_base, gemma_post, '#2196F3'),
        (axes[1], 'Qwen 7B (success)', qwen_base, qwen_post, '#2196F3'),
        (axes[2], 'Llama 8B (failure)', llama_base, llama_post, '#f44336'),
    ]:
        x = np.arange(len(configs))
        w = 0.3
        ax.bar(x - w/2, base, w, label='Baseline CV', color='#bdbdbd', edgecolor='white')
        ax.bar(x + w/2, post, w, label='Post-PT CV', color=color, edgecolor='white', alpha=0.8)

        # Delta annotations
        for i in range(len(configs)):
            delta = post[i] - base[i]
            c = '#2196F3' if delta >= -0.02 else '#f44336'
            ax.text(i + w/2, post[i] + 0.008, f'{delta:+.3f}', ha='center', fontsize=6.5, color=c)

        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0.5, 0.9)
        ax.axhline(y=0.5, color='#999', linestyle=':', linewidth=0.5)
        if ax == axes[0]:
            ax.set_ylabel('AUROC₂ (5-fold CV)')
            ax.legend(frameon=False, fontsize=8, loc='lower left')

    fig.suptitle('Retrained probes on post-PT-CSFT hidden states', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_retrained_probes.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig3_retrained_probes.pdf', bbox_inches='tight')
    print(f'  [save] fig3_retrained_probes')
    plt.close()


# ================================================================
# Figure 4: Steering sweep
# ================================================================
def fig4_steering():
    # Gemma 12B
    alphas = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
    gemma_auroc = [0.680, 0.684, 0.684, 0.684, 0.684, 0.684, 0.684, 0.684, 0.684, 0.673]
    gemma_conf = [97.8, 97.8, 97.8, 97.8, 97.8, 97.8, 97.8, 97.8, 97.8, 97.8]

    # Qwen 7B
    qwen_auroc = [0.635, 0.586, 0.583, 0.582, 0.581, 0.581, 0.581, 0.581, 0.580, 0.577]
    qwen_conf = [97.3, 97.5, 97.5, 97.6, 97.6, 97.6, 97.6, 97.6, 97.7, 97.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(alphas, gemma_auroc, 'o-', color='#2196F3', label='Gemma 12B', markersize=5)
    ax1.plot(alphas, qwen_auroc, 's-', color='#FF9800', label='Qwen 7B', markersize=5)
    ax1.axhline(y=0.5, color='#999', linestyle=':', linewidth=0.5)
    ax1.set_xlabel('Steering strength (α)')
    ax1.set_ylabel('AUROC₂')
    ax1.set_title('Verbal discrimination vs steering', fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_ylim(0.5, 0.72)

    ax2.plot(alphas, gemma_conf, 'o-', color='#2196F3', label='Gemma 12B', markersize=5)
    ax2.plot(alphas, qwen_conf, 's-', color='#FF9800', label='Qwen 7B', markersize=5)
    ax2.set_xlabel('Steering strength (α)')
    ax2.set_ylabel('Mean confidence')
    ax2.set_title('Confidence mean vs steering', fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_ylim(95, 100)

    fig.suptitle('Inference-time steering along probe direction: null result', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_steering.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig4_steering.pdf', bbox_inches='tight')
    print(f'  [save] fig4_steering')
    plt.close()


# ================================================================
# Figure 5: Phase 0 bimodality scaling
# ================================================================
def fig5_bimodality():
    scales = ['4B', '12B', '27B']
    extreme_pct = [84.6, 91.4, 94.5]
    csft_auroc = [0.774, 0.540, 0.499]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.bar(scales, extreme_pct, color=['#4CAF50', '#FF9800', '#f44336'], edgecolor='white')
    ax1.set_ylabel('% items at n_correct ∈ {0, 10}')
    ax1.set_xlabel('Model scale')
    ax1.set_title('Self-consistency bimodality', fontsize=11)
    ax1.set_ylim(0, 100)
    for i, v in enumerate(extreme_pct):
        ax1.text(i, v + 1.5, f'{v}%', ha='center', fontsize=10)

    ax2.bar(scales, csft_auroc, color=['#4CAF50', '#FF9800', '#f44336'], edgecolor='white')
    ax2.axhline(y=0.5, color='#999', linestyle=':', linewidth=0.8, label='Chance')
    ax2.set_ylabel('AUROC₂ after SC-CSFT')
    ax2.set_xlabel('Model scale')
    ax2.set_title('SC-CSFT outcome', fontsize=11)
    ax2.set_ylim(0.3, 0.85)
    ax2.legend(frameon=False, fontsize=9)
    for i, v in enumerate(csft_auroc):
        ax2.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=10)

    fig.suptitle('Phase 0: Self-consistency targets fail at scale', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_bimodality.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig5_bimodality.pdf', bbox_inches='tight')
    print(f'  [save] fig5_bimodality')
    plt.close()


# ================================================================
# Figure 6: E10 answer-unchanged deltas
# ================================================================
def fig6_e10():
    models = ['Gemma 12B', 'Gemma 27B', 'Qwen 7B', 'Llama 8B']
    deltas = [0.184, 0.077, 0.211, -0.225]
    colors = ['#2196F3', '#2196F3', '#2196F3', '#f44336']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, deltas, color=colors, edgecolor='white')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('AUROC₂ delta (answer-unchanged items)')
    ax.set_title('E10 diagnostic: confidence improvement on unchanged answers', fontsize=11)

    for i, (bar, d) in enumerate(zip(bars, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2, d + (0.01 if d > 0 else -0.02),
                f'{d:+.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylim(-0.3, 0.3)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_e10.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig6_e10.pdf', bbox_inches='tight')
    print(f'  [save] fig6_e10')
    plt.close()


if __name__ == '__main__':
    print("Generating Phase 1 paper figures...")
    fig1_auroc_comparison()
    fig2_calibration_metrics()
    fig3_retrained_probes()
    fig4_steering()
    fig5_bimodality()
    fig6_e10()
    print(f"\nAll figures saved to {OUT}/")
