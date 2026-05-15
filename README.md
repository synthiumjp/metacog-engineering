# metacog-engineering

Closing the metacognitive gap in LLMs: from diagnosis to intervention.

**Phase 0 paper:** [arXiv:2604.24070](https://arxiv.org/abs/2604.24070)
**Phase 1 paper:** *In preparation for JAIR*
**Pre-registration:** [OSF (osf.io/mpcr5)](https://osf.io/mpcr5)

## The Problem

Instruct-tuned LLMs report 95–100% confidence regardless of correctness. Linear probes on hidden states achieve AUROC₂ of 0.76–0.88 on the same items. The information is there internally — the verbal channel just doesn't transmit it. This project builds methods to close that gap.

## What We Found

**Phase 0 (Gemma 4B):** Self-consistency-derived confidence targets produce a binary verbal correctness discriminator (AUROC₂ = 0.774) at 4B scale, but the self-consistency signal becomes bimodally degenerate at 12B and 27B (84.6% → 91.4% → 94.5% extreme items). SC-CSFT does not scale.

**Phase 1 (6 models, 3 architecture classes):** Probe-targeted CSFT (PT-CSFT) replaces the self-consistency signal with the output of a linear probe trained on the model's own hidden states. Results:

| Model | Baseline AUROC₂ | PT-CSFT AUROC₂ | ECE | VRS |
|-------|----------------|----------------|-----|-----|
| Gemma 12B | 0.684 | **0.842** | 0.108 | Invalid → Invalid |
| Gemma 27B | 0.717 | **0.785** | — | Invalid → Indeterminate |
| Qwen 7B | 0.581 | **0.754** | 0.068 | Invalid → **Valid** |
| Llama 8B | 0.711 | 0.513 | 0.276 | Invalid → Invalid |
| Llama 70B | 0.760 | 0.582 | — | Invalid → Invalid |
| Mistral 7B | 0.584 | diverged | — | n/a |

PT-CSFT succeeds on GemmaForCausalLM and Qwen2ForCausalLM. All LlamaForCausalLM models fail under the same configuration. The failure is not scale-dependent (8B and 70B both fail).

**Key controls:**
- Shuffled-target control: no improvement (AUROC₂ ≈ 0.50)
- E10 answer-unchanged diagnostic: AUROC₂ improves +0.08 to +0.21 on items where the answer didn't change
- Cross-benchmark: confidence calibration transfers from TriviaQA to MMLU
- Retrained probes: correctness information preserved in Llama after PT-CSFT (middle-layer δ = −0.004) but inaccessible to verbal output
- Steering: additive inference-time steering along the probe direction has no effect on verbal confidence

## Repository Structure

```
metacog-engineering/
├── scripts/                        # Phase 0 pipeline (Windows/ROCm)
│   ├── utils_phase0.py
│   ├── step0_substrate_check.py
│   ├── step1_baseline.py
│   ├── step1b_probe.py
│   ├── step2_calibration.py
│   ├── step2b_no_filter.py
│   ├── step3_finetune.py
│   └── step4_evaluation.py
├── phase1/
│   ├── scripts/                    # Phase 1 pipeline (Mac/MLX)
│   │   ├── model_config.py
│   │   ├── gen_helpers.py
│   │   ├── step0_spike_phase1.py
│   │   ├── step1_baseline_phase1.py
│   │   ├── step1b_probe_phase1.py
│   │   ├── step2_calibration_phase1.py
│   │   ├── step3_finetune_phase1.py
│   │   ├── step4_eval_phase1.py
│   │   ├── step_probe_target.py
│   │   ├── step_steering.py
│   │   ├── step_exp4_post_pt_probes.py
│   │   ├── step_prereg_compliance.py
│   │   └── step_reviewer_experiments.py
│   └── results/                    # Phase 1 results (JSONs)
│       ├── step4/                  # PT-CSFT evaluation metrics
│       ├── steering/               # Steering sweep results
│       ├── prereg_compliance/      # E10 + meta-d' compliance
│       ├── reviewer_experiments/   # Retrained probes + ECE/Brier
│       └── step1_post_pt/          # Post-PT baseline metrics
├── data/                           # Phase 0 data splits
├── results/                        # Phase 0 results
├── figures/                        # Phase 0 figures
└── .gitignore
```

## Data

All items from TriviaQA rc.nocontext validation and MMLU. Shuffled with seed 42, partitioned:

- **T-eval:** 1,000 items (held-out evaluation — never used for training or probe fitting)
- **T-cal:** 2,000 items (probe training + CSFT fine-tuning)
- **M-eval:** 498 MMLU items (cross-benchmark transfer)

Partitions identical across Phase 0 and Phase 1. Disjointness verified programmatically.

## Hardware

- **Phase 0:** AMD Radeon RX 7900 GRE (16GB VRAM), ROCm PyTorch 2.8.0, Windows 11
- **Phase 1:** Apple M3 Ultra (512GB unified memory), MLX framework, macOS

## Method: PT-CSFT Pipeline

1. **Baseline** → greedy generation, extract hidden states at 3 layers × 2 token positions
2. **Probe** → L2-regularised logistic regression on hidden states, 5-fold CV
3. **Target** → probe P(correct) scaled to 0–100 as confidence target
4. **Fine-tune** → LoRA (rank 16, α=32) with probe-derived targets
5. **Evaluate** → AUROC₂, VRS, ECE, Brier, E10, paired-bootstrap CIs

## Related Work

Part of the CMM (Classical Minds, Modern Machines) programme applying Type-2 signal detection theory to LLM metacognition:

- [Do LLMs Know What They Know?](https://arxiv.org/abs/2603.25112) — M1: domain-specific metacognitive profiles
- [The Metacognitive Monitoring Battery](https://arxiv.org/abs/2604.15702) — M2: cross-domain benchmark (submitted NeurIPS 2026)
- [Quantisation Reshapes Metacognitive Geometry](https://arxiv.org/abs/2604.08976) — M-ratio profiles restructure across formats
- [Screen Before You Interpret](https://arxiv.org/abs/2604.17714) — VRS validity protocol
- [Domain-Level Metacognitive Monitoring](https://arxiv.org/abs/2605.06673) — 33-model atlas

## Citation

```bibtex
@article{cacioli2026ptcsft,
  title={Bridging the Metacognitive Gap: Probe-Targeted Fine-Tuning
         Improves Verbal Confidence Calibration Across {LLM} Architectures},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={In preparation}
}
```

## License

Code: MIT. Data splits are deterministic from publicly available benchmarks (TriviaQA, MMLU).
