#!/usr/bin/env python3
"""
Activation Patching: Confidence-Position Hidden State Swap
===========================================================
Tests the routing thesis directly by injecting PT-CSFT hidden states
at the confidence-token position into a baseline forward pass.

Supports single-layer and all_sweep (8 evenly-spaced layers).

Usage:
    # Smoke test (5 items, ~2 min):
    python3 activation_patching.py --n-items 5 --patch-layer middle

    # Full single layer (100 items, ~1 hour):
    python3 activation_patching.py --n-items 100 --patch-layer middle

    # Layer sweep (100 items × 8 layers, ~3-4 hours):
    python3 activation_patching.py --n-items 100 --patch-layer all_sweep

All paths default to Llama 8B gentle-lr. Override with --model-path and --adapter-path.
"""

import argparse, json, os, sys, time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.expanduser(
        "~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16"))
    parser.add_argument("--adapter-path", default=os.path.expanduser(
        "~/jpwork/metacog-engineering/phase1/results_raw/finetune/"
        "Meta-Llama-3.1-8B-Instruct-bf16/ablation_gentle_lr/adapters"))
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--patch-layer", choices=["first", "middle", "last", "all_sweep"], default="middle")
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
    test_items = [ds[int(i)] for i in indices[:args.n_items]]

    # ---------------------------------------------------------------
    # Model internals
    # ---------------------------------------------------------------
    def get_internals(model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            m = model.model
            return m.layers, m.embed_tokens, m.norm, getattr(model, "lm_head", None)
        if hasattr(model, "language_model"):
            m = model.language_model.model
            return m.layers, m.embed_tokens, m.norm, getattr(model.language_model, "lm_head", None)
        raise ValueError("Cannot find layers")

    def forward_capture(layers, embed, norm, lm_head, input_ids, capture_layer, capture_pos):
        h = embed(input_ids)
        seq_len = h.shape[1]
        mask = mx.full((seq_len, seq_len), -1e9, dtype=mx.bfloat16)
        mask = mx.triu(mask, k=1)
        captured = None
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            if i == capture_layer:
                captured = h[:, capture_pos:capture_pos + 1, :]
        h = norm(h)
        logits = lm_head(h) if lm_head else h @ embed.weight.T
        return logits, captured

    def forward_inject(layers, embed, norm, lm_head, input_ids, inject_layer, inject_pos, inject_state):
        h = embed(input_ids)
        seq_len = h.shape[1]
        mask = mx.full((seq_len, seq_len), -1e9, dtype=mx.bfloat16)
        mask = mx.triu(mask, k=1)
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            if i == inject_layer:
                before = h[:, :inject_pos, :]
                after = h[:, inject_pos + 1:, :]
                h = mx.concatenate([before, inject_state, after], axis=1)
        h = norm(h)
        logits = lm_head(h) if lm_head else h @ embed.weight.T
        return logits

    # Build digit map
    print("Loading model for tokenizer...")
    model_tmp, tokenizer = load(args.model_path)
    digit_ids = {}
    for d in range(101):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if len(toks) == 1:
            digit_ids[d] = toks[0]
    digits_sorted = sorted(digit_ids.keys())
    digit_token_ids = [digit_ids[d] for d in digits_sorted]
    digits_arr = np.array(digits_sorted, dtype=np.float64)
    print(f"Single-token digits: {len(digit_ids)}/101")

    tmp_layers, _, _, _ = get_internals(model_tmp)
    n_layers = len(tmp_layers)
    del model_tmp
    mx.clear_cache()

    mid = n_layers // 2
    layer_map = {
        "first": [0],
        "middle": [mid],
        "last": [n_layers - 1],
        "all_sweep": list(range(0, n_layers, max(1, n_layers // 8))),
    }
    patch_layers = layer_map[args.patch_layer]
    print(f"Layers: {n_layers}. Patching at: {patch_layers}")

    PROMPT = (
        "You are answering trivia questions. "
        "After your answer, state your confidence as a percentage from 0 to 100.\n"
        "Question: {question}\n"
    )

    def extract_conf(logits_tensor):
        last_logits = logits_tensor[0, -1, :]
        digit_logits = mx.array([float(last_logits[tid]) for tid in digit_token_ids])
        probs = mx.softmax(digit_logits)
        probs_np = np.array(probs.tolist())
        return float(np.sum(digits_arr * probs_np) / max(digits_arr))

    # ---------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------
    all_results = []

    for patch_layer in patch_layers:
        print(f"\n{'='*60}")
        print(f"Patch layer {patch_layer}/{n_layers}")
        print(f"{'='*60}")

        model_pt, _ = load(args.model_path, adapter_path=args.adapter_path)
        pt_layers, pt_embed, pt_norm, pt_head = get_internals(model_pt)
        model_bl, _ = load(args.model_path)
        bl_layers, bl_embed, bl_norm, bl_head = get_internals(model_bl)

        for idx, item in enumerate(test_items):
            messages = [{"role": "user", "content": PROMPT.format(question=item["question"])}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = mx.array(tokenizer.encode(prompt_text)).reshape(1, -1)
            conf_pos = input_ids.shape[1] - 1

            try:
                pt_logits, pt_hidden = forward_capture(pt_layers, pt_embed, pt_norm, pt_head, input_ids, patch_layer, conf_pos)
                bl_logits, _ = forward_capture(bl_layers, bl_embed, bl_norm, bl_head, input_ids, patch_layer, conf_pos)
                patched_logits = forward_inject(bl_layers, bl_embed, bl_norm, bl_head, input_ids, patch_layer, conf_pos, pt_hidden)

                bl_conf = extract_conf(bl_logits)
                pt_conf = extract_conf(pt_logits)
                pa_conf = extract_conf(patched_logits)
                gap = pt_conf - bl_conf
                shift = (pa_conf - bl_conf) / (gap + 1e-10) if abs(gap) > 0.01 else None

                all_results.append({
                    "idx": idx, "layer": patch_layer,
                    "baseline_conf": round(bl_conf, 4), "ptcsft_conf": round(pt_conf, 4),
                    "patched_conf": round(pa_conf, 4),
                    "shift": round(shift, 4) if shift else None,
                })

                if idx < 5 or idx % 20 == 0:
                    s = f"{shift:.2f}" if shift else "nan"
                    print(f"  [{idx:3d}] bl={bl_conf:.3f} pt={pt_conf:.3f} patched={pa_conf:.3f} shift={s}")
                mx.eval(patched_logits)
            except Exception as e:
                print(f"  [{idx:3d}] ERROR: {e}")
                if idx == 0:
                    import traceback; traceback.print_exc()

        del model_pt, model_bl
        mx.clear_cache()

    # Summary
    shifts = [r["shift"] for r in all_results if r["shift"] is not None]
    if shifts:
        s = np.clip(shifts, -5, 5)
        print(f"\n{'='*60}")
        print(f"RESULTS ({len(shifts)} items with measurable gap)")
        print(f"{'='*60}")
        print(f"Mean shift: {np.mean(s):.3f} +/- {np.std(s):.3f}")
        print(f"Median: {np.median(s):.3f}")
        print(f"Frac > 0: {(np.array(s) > 0).mean():.1%}")
        print(f"Frac > 0.5: {(np.array(s) > 0.5).mean():.1%}")

    output_path = args.output_path or os.path.join(
        os.path.dirname(args.adapter_path),
        f"activation_patching_{'sweep' if len(patch_layers) > 1 else f'layer{patch_layers[0]}'}.json",
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
