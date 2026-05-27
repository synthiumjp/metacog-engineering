#!/usr/bin/env python3
"""
Activation Patching Controls
=============================
Three controls for the routing thesis:
1. REVERSE: Patch baseline states into PT-CSFT (bidirectional test)
2. CONTROL POSITION: Patch at mid-question instead of confidence position
3. ANSWER CHECK: Does patching change the answer or just confidence?

Usage:
    python3 patching_controls.py --n-items 100
"""

import argparse, json, os, sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.expanduser(
        "~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16"))
    parser.add_argument("--adapter-path", default=os.path.expanduser(
        "~/jpwork/metacog-engineering/phase1/results_raw/finetune/"
        "Meta-Llama-3.1-8B-Instruct-bf16/ablation_gentle_lr/adapters"))
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from datasets import load_dataset
    import random

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    test_items = [ds[int(i)] for i in indices[:args.n_items]]

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.adapter_path), "patching_controls"
    )
    os.makedirs(output_dir, exist_ok=True)

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

    # Digit map
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

    tmp_layers, _, _, _ = get_internals(model_tmp)
    n_layers = len(tmp_layers)
    del model_tmp
    mx.clear_cache()

    PATCH_LAYER = n_layers - 4  # layer 28 for 32-layer model
    print(f"Layers: {n_layers}, patch layer: {PATCH_LAYER}")

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

    def get_argmax_token(logits_tensor):
        return int(mx.argmax(logits_tensor[0, -1, :]))

    # ===============================================================
    # EXPERIMENT 1: REVERSE
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"EXP 1: REVERSE PATCHING (baseline → PT-CSFT, layer {PATCH_LAYER})")
    print(f"{'='*60}")

    model_pt, _ = load(args.model_path, adapter_path=args.adapter_path)
    pt_l, pt_e, pt_n, pt_h = get_internals(model_pt)
    model_bl, _ = load(args.model_path)
    bl_l, bl_e, bl_n, bl_h = get_internals(model_bl)

    reverse_results = []
    for idx, item in enumerate(test_items):
        messages = [{"role": "user", "content": PROMPT.format(question=item["question"])}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = mx.array(tokenizer.encode(prompt_text)).reshape(1, -1)
        conf_pos = input_ids.shape[1] - 1
        try:
            bl_logits, bl_hidden = forward_capture(bl_l, bl_e, bl_n, bl_h, input_ids, PATCH_LAYER, conf_pos)
            pt_logits, _ = forward_capture(pt_l, pt_e, pt_n, pt_h, input_ids, PATCH_LAYER, conf_pos)
            rev_logits = forward_inject(pt_l, pt_e, pt_n, pt_h, input_ids, PATCH_LAYER, conf_pos, bl_hidden)
            bl_c, pt_c, rev_c = extract_conf(bl_logits), extract_conf(pt_logits), extract_conf(rev_logits)
            gap = bl_c - pt_c
            shift = (rev_c - pt_c) / (gap + 1e-10) if abs(gap) > 0.01 else None
            reverse_results.append({"idx": idx, "bl": round(bl_c,4), "pt": round(pt_c,4), "rev": round(rev_c,4), "shift": round(shift,4) if shift else None})
            if idx < 5 or idx % 20 == 0:
                print(f"  [{idx:3d}] pt={pt_c:.3f} bl={bl_c:.3f} rev={rev_c:.3f} shift→bl={f'{shift:.2f}' if shift else 'nan'}")
            mx.eval(rev_logits)
        except Exception as e:
            print(f"  [{idx:3d}] ERROR: {e}")

    del model_pt, model_bl; mx.clear_cache()

    # ===============================================================
    # EXPERIMENT 2: CONTROL POSITION
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"EXP 2: CONTROL POSITION (mid-question, layer {PATCH_LAYER})")
    print(f"{'='*60}")

    model_pt, _ = load(args.model_path, adapter_path=args.adapter_path)
    pt_l, pt_e, pt_n, pt_h = get_internals(model_pt)
    model_bl, _ = load(args.model_path)
    bl_l, bl_e, bl_n, bl_h = get_internals(model_bl)

    control_results = []
    for idx, item in enumerate(test_items):
        messages = [{"role": "user", "content": PROMPT.format(question=item["question"])}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = mx.array(tokenizer.encode(prompt_text)).reshape(1, -1)
        conf_pos = input_ids.shape[1] - 1
        ctrl_pos = input_ids.shape[1] // 2
        try:
            _, pt_hidden_ctrl = forward_capture(pt_l, pt_e, pt_n, pt_h, input_ids, PATCH_LAYER, ctrl_pos)
            bl_logits, _ = forward_capture(bl_l, bl_e, bl_n, bl_h, input_ids, PATCH_LAYER, ctrl_pos)
            ctrl_logits = forward_inject(bl_l, bl_e, bl_n, bl_h, input_ids, PATCH_LAYER, ctrl_pos, pt_hidden_ctrl)
            _, pt_hidden_conf = forward_capture(pt_l, pt_e, pt_n, pt_h, input_ids, PATCH_LAYER, conf_pos)
            conf_logits = forward_inject(bl_l, bl_e, bl_n, bl_h, input_ids, PATCH_LAYER, conf_pos, pt_hidden_conf)

            bl_c = extract_conf(bl_logits)
            pt_c = extract_conf(forward_capture(pt_l, pt_e, pt_n, pt_h, input_ids, PATCH_LAYER, conf_pos)[0])
            ctrl_c = extract_conf(ctrl_logits)
            conf_c = extract_conf(conf_logits)
            gap = pt_c - bl_c
            ctrl_shift = (ctrl_c - bl_c) / (gap + 1e-10) if abs(gap) > 0.01 else None
            conf_shift = (conf_c - bl_c) / (gap + 1e-10) if abs(gap) > 0.01 else None
            control_results.append({"idx": idx, "ctrl_shift": round(ctrl_shift,4) if ctrl_shift else None, "conf_shift": round(conf_shift,4) if conf_shift else None})
            if idx < 5 or idx % 20 == 0:
                print(f"  [{idx:3d}] ctrl={f'{ctrl_shift:.2f}' if ctrl_shift else 'nan'}  conf={f'{conf_shift:.2f}' if conf_shift else 'nan'}")
            mx.eval(ctrl_logits, conf_logits)
        except Exception as e:
            print(f"  [{idx:3d}] ERROR: {e}")

    del model_pt, model_bl; mx.clear_cache()

    # ===============================================================
    # EXPERIMENT 3: ANSWER CHECK
    # ===============================================================
    print(f"\n{'='*60}")
    print(f"EXP 3: ANSWER ACCURACY CHECK (layer {PATCH_LAYER})")
    print(f"{'='*60}")

    model_pt, _ = load(args.model_path, adapter_path=args.adapter_path)
    pt_l, pt_e, pt_n, pt_h = get_internals(model_pt)
    model_bl, _ = load(args.model_path)
    bl_l, bl_e, bl_n, bl_h = get_internals(model_bl)

    answer_results = []
    for idx, item in enumerate(test_items):
        messages = [{"role": "user", "content": PROMPT.format(question=item["question"])}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = mx.array(tokenizer.encode(prompt_text)).reshape(1, -1)
        conf_pos = input_ids.shape[1] - 1
        try:
            bl_logits, _ = forward_capture(bl_l, bl_e, bl_n, bl_h, input_ids, PATCH_LAYER, conf_pos)
            pt_logits, pt_hidden = forward_capture(pt_l, pt_e, pt_n, pt_h, input_ids, PATCH_LAYER, conf_pos)
            pa_logits = forward_inject(bl_l, bl_e, bl_n, bl_h, input_ids, PATCH_LAYER, conf_pos, pt_hidden)
            bl_tok, pa_tok = get_argmax_token(bl_logits), get_argmax_token(pa_logits)
            bl_c, pa_c = extract_conf(bl_logits), extract_conf(pa_logits)
            answer_results.append({"idx": idx, "same": bl_tok == pa_tok, "conf_shifted": abs(pa_c - bl_c) > 0.02, "bl_conf": round(bl_c,4), "pa_conf": round(pa_c,4)})
            if idx < 5 or idx % 20 == 0:
                s = "SAME" if bl_tok == pa_tok else "DIFF"
                print(f"  [{idx:3d}] [{s}] conf: {bl_c:.3f}→{pa_c:.3f}")
            mx.eval(pa_logits)
        except Exception as e:
            print(f"  [{idx:3d}] ERROR: {e}")

    del model_pt, model_bl; mx.clear_cache()

    # ===============================================================
    # SUMMARY
    # ===============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    rev_shifts = [r["shift"] for r in reverse_results if r["shift"] is not None]
    if rev_shifts:
        s = np.clip(rev_shifts, -5, 5)
        print(f"\n1. REVERSE (layer {PATCH_LAYER}):")
        print(f"   Mean shift toward baseline: {np.mean(s):.3f} +/- {np.std(s):.3f}")
        print(f"   Frac shifting toward baseline: {(np.array(s) > 0).mean():.1%}  (N={len(s)})")
        print(f"   Bidirectional? {'YES' if np.mean(s) > 0.2 and (np.array(s)>0).mean() > 0.6 else 'NO'}")

    ctrl = [r["ctrl_shift"] for r in control_results if r["ctrl_shift"] is not None]
    conf = [r["conf_shift"] for r in control_results if r["conf_shift"] is not None]
    if ctrl and conf:
        cs, fs = np.clip(ctrl, -5, 5), np.clip(conf, -5, 5)
        print(f"\n2. CONTROL POSITION (layer {PATCH_LAYER}):")
        print(f"   Control (mid-question): mean={np.mean(cs):.3f}, frac>0={( np.array(cs)>0).mean():.1%}")
        print(f"   Confidence position:    mean={np.mean(fs):.3f}, frac>0={(np.array(fs)>0).mean():.1%}")
        print(f"   Position-specific? {'YES' if np.mean(fs) > 2*np.mean(np.abs(cs)) else 'NO'}")

    if answer_results:
        n_same = sum(1 for r in answer_results if r["same"])
        n_shifted = sum(1 for r in answer_results if r["conf_shifted"])
        n_both = sum(1 for r in answer_results if r["same"] and r["conf_shifted"])
        n = len(answer_results)
        print(f"\n3. ANSWER CHECK (layer {PATCH_LAYER}):")
        print(f"   Answers unchanged: {n_same}/{n} ({n_same/n:.1%})")
        print(f"   Confidence shifted: {n_shifted}/{n} ({n_shifted/n:.1%})")
        print(f"   Both: {n_both}/{n} ({n_both/n:.1%})")
        print(f"   Selective routing? {'YES' if n_same/n > 0.8 and n_shifted/n > 0.3 else 'NO'}")

    out_path = os.path.join(output_dir, "patching_controls.json")
    with open(out_path, "w") as f:
        json.dump({"reverse": reverse_results, "control": control_results, "answer": answer_results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
