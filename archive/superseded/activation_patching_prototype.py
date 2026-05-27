#!/usr/bin/env python3
"""
Activation Patching: Confidence-Position Hidden State Swap
===========================================================
PROTOTYPE. Tests routing thesis: inject PT-CSFT hidden state at the
confidence position into a baseline forward pass. If verbal confidence
shifts, routing is causal.

Usage (smoke test first):
    python3 activation_patching_prototype.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16 \
        --adapter-path ~/jpwork/metacog-engineering/phase1/results_raw/finetune/Meta-Llama-3.1-8B-Instruct-bf16/ablation_gentle_lr/adapters \
        --n-items 5 --patch-layer middle

Full run (~1 hour):
    Same command with --n-items 100
"""

import argparse, json, os, sys, time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument(
        "--patch-layer",
        choices=["first", "middle", "last", "all_sweep"],
        default="middle",
    )
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from datasets import load_dataset
    import random

    # Load T-eval
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    test_items = [ds[int(i)] for i in indices[: args.n_items]]
    print(f"Testing on {len(test_items)} items")

    # ---------------------------------------------------------------
    # Helpers to find model internals
    # ---------------------------------------------------------------
    def get_internals(model):
        """Return (layers, embed, norm, lm_head) regardless of architecture."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            m = model.model
            lm_head = getattr(model, "lm_head", None)
            return m.layers, m.embed_tokens, m.norm, lm_head
        if hasattr(model, "language_model"):
            m = model.language_model.model
            lm_head = getattr(model.language_model, "lm_head", None)
            return m.layers, m.embed_tokens, m.norm, lm_head
        raise ValueError("Cannot find transformer layers")

    def forward_capture(layers, embed, norm, lm_head, input_ids, capture_layer, capture_pos):
        """Forward pass capturing hidden state at one layer/position."""
        h = embed(input_ids)
        seq_len = h.shape[1]
        mask = mx.full((seq_len, seq_len), -1e9, dtype=mx.bfloat16)
        mask = mx.triu(mask, k=1)
        captured = None
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            if i == capture_layer:
                captured = h[:, capture_pos : capture_pos + 1, :]
        h = norm(h)
        if lm_head is not None:
            logits = lm_head(h)
        else:
            logits = h @ embed.weight.T
        return logits, captured

    def forward_inject(layers, embed, norm, lm_head, input_ids, inject_layer, inject_pos, inject_state):
        """Forward pass injecting a hidden state at one layer/position."""
        h = embed(input_ids)
        seq_len = h.shape[1]
        mask = mx.full((seq_len, seq_len), -1e9, dtype=mx.bfloat16)
        mask = mx.triu(mask, k=1)
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            if i == inject_layer:
                # Build replacement: keep everything except inject_pos
                before = h[:, :inject_pos, :]
                after = h[:, inject_pos + 1 :, :]
                h = mx.concatenate([before, inject_state, after], axis=1)
        h = norm(h)
        if lm_head is not None:
            logits = lm_head(h)
        else:
            logits = h @ embed.weight.T
        return logits

    # ---------------------------------------------------------------
    # Build digit-token map for confidence readout
    # ---------------------------------------------------------------
    print("Loading baseline model (for tokenizer + digit map)...")
    model_tmp, tokenizer = load(args.model_path)
    digit_ids = {}
    for d in range(101):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if len(toks) == 1:
            digit_ids[d] = toks[0]
    print(f"Single-token digits: {len(digit_ids)}/101")
    digits_sorted = sorted(digit_ids.keys())
    digit_token_ids = [digit_ids[d] for d in digits_sorted]
    digits_arr = np.array(digits_sorted, dtype=np.float64)

    layers_tmp, _, _, _ = get_internals(model_tmp)
    n_layers = len(layers_tmp)
    del model_tmp
    if hasattr(mx, "metal"):
        mx.metal.clear_cache()

    mid = n_layers // 2
    layer_map = {
        "first": [0],
        "middle": [mid],
        "last": [n_layers - 1],
        "all_sweep": list(range(0, n_layers, max(1, n_layers // 8))),
    }
    patch_layers = layer_map[args.patch_layer]
    print(f"Model: {n_layers} layers. Patching at: {patch_layers}")

    # ---------------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------------
    PROMPT = (
        "You are answering trivia questions. "
        "After your answer, state your confidence as a percentage from 0 to 100.\n"
        "Question: {question}\n"
    )

    def extract_conf(logits_tensor):
        """Expected confidence from logits at last position."""
        last_logits = logits_tensor[0, -1, :]
        digit_logits = mx.array([float(last_logits[tid]) for tid in digit_token_ids])
        probs = mx.softmax(digit_logits)
        probs_np = np.array(probs.tolist())
        expected = float(np.sum(digits_arr * probs_np) / max(digits_arr))
        return expected

    # ---------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------
    all_results = []

    for patch_layer in patch_layers:
        print(f"\n{'='*60}")
        print(f"Patch layer {patch_layer}/{n_layers}")
        print(f"{'='*60}")

        print("  Loading PT-CSFT model...")
        model_pt, _ = load(args.model_path, adapter_path=args.adapter_path)
        pt_layers, pt_embed, pt_norm, pt_head = get_internals(model_pt)

        print("  Loading baseline model...")
        model_bl, _ = load(args.model_path)
        bl_layers, bl_embed, bl_norm, bl_head = get_internals(model_bl)

        for idx, item in enumerate(test_items):
            question = item["question"]
            messages = [{"role": "user", "content": PROMPT.format(question=question)}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = mx.array(tokenizer.encode(prompt_text)).reshape(1, -1)
            conf_pos = input_ids.shape[1] - 1

            try:
                # PT-CSFT forward: capture hidden state
                pt_logits, pt_hidden = forward_capture(
                    pt_layers, pt_embed, pt_norm, pt_head,
                    input_ids, patch_layer, conf_pos
                )

                # Baseline forward: unpatched
                bl_logits, _ = forward_capture(
                    bl_layers, bl_embed, bl_norm, bl_head,
                    input_ids, patch_layer, conf_pos
                )

                # Baseline forward: inject PT-CSFT hidden state
                patched_logits = forward_inject(
                    bl_layers, bl_embed, bl_norm, bl_head,
                    input_ids, patch_layer, conf_pos, pt_hidden
                )

                bl_conf = extract_conf(bl_logits)
                pt_conf = extract_conf(pt_logits)
                pa_conf = extract_conf(patched_logits)

                gap = pt_conf - bl_conf
                shift = (pa_conf - bl_conf) / (gap + 1e-10) if abs(gap) > 0.01 else float("nan")

                result = {
                    "idx": idx,
                    "layer": patch_layer,
                    "baseline_conf": round(bl_conf, 4),
                    "ptcsft_conf": round(pt_conf, 4),
                    "patched_conf": round(pa_conf, 4),
                    "shift": round(shift, 4) if not np.isnan(shift) else None,
                }
                all_results.append(result)

                if idx < 5 or idx % 20 == 0:
                    print(
                        f"  [{idx:3d}] bl={bl_conf:.3f} pt={pt_conf:.3f} "
                        f"patched={pa_conf:.3f} shift={shift:.2f}"
                    )

                mx.eval(patched_logits)

            except Exception as e:
                print(f"  [{idx:3d}] ERROR: {e}")
                if idx == 0:
                    import traceback
                    traceback.print_exc()
                    print("\n  >>> Forward pass hooks need adjustment for this architecture.")
                    print("  >>> Kill and defer to R&R if this doesn't resolve quickly.")
                continue

        del model_pt, model_bl
        if hasattr(mx, "metal"):
            mx.metal.clear_cache()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    if not all_results:
        print("\nNo results. Forward pass hooks need debugging.")
        sys.exit(1)

    shifts = [r["shift"] for r in all_results if r["shift"] is not None]

    print(f"\n{'='*60}")
    print(f"RESULTS ({len(shifts)} items with measurable gap)")
    print(f"{'='*60}")

    if shifts:
        s = np.clip(shifts, -5, 5)
        print(f"Mean shift toward PT-CSFT: {np.mean(s):.3f} +/- {np.std(s):.3f}")
        print(f"Median: {np.median(s):.3f}")
        print(f"Fraction > 0: {(np.array(s) > 0).mean():.1%}")
        print(f"Fraction > 0.5: {(np.array(s) > 0.5).mean():.1%}")

        if np.mean(s) > 0.3:
            print("\n>>> STRONG EVIDENCE for routing thesis.")
        elif np.mean(s) > 0.1:
            print("\n>>> MODERATE EVIDENCE — effect may be distributed across layers.")
        else:
            print("\n>>> WEAK/NO EVIDENCE at this layer. Try --patch-layer all_sweep")

    output_path = args.output_path or os.path.join(
        os.path.dirname(args.adapter_path),
        f"activation_patching_layer{patch_layers[0]}.json",
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
