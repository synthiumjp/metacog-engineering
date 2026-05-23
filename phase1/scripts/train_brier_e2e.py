#!/usr/bin/env python3
"""
train_brier_e2e.py — Minimal custom MLX training loop for Brier-score PT-CSFT.

Hybrid loss: standard CE on answer tokens + tokenized Brier score on confidence token.
Binary correctness labels (0/1) — the proper scoring rule incentivises P(correct).

Usage:
    python3 train_brier_e2e.py \
        --model-path ~/mnt/models-lan/.../gemma-3-12b-it \
        --train-data .../train.jsonl \
        --val-data .../valid.jsonl \
        --output-dir .../adapter_brier \
        --lr 2e-4 --epochs 3 --grad-accum 16 --brier-weight 1.0
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def load_dataset(path, tokenizer, max_seq_len=1024):
    """Load JSONL, tokenize, find confidence token position."""
    # Get digit token IDs (0-9)
    digit_ids = []
    for d in range(10):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        assert len(toks) == 1, f"Digit {d} is multi-token"
        digit_ids.append(toks[0])
    digit_set = set(digit_ids)

    items = []
    with open(path) as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            meta = data.get("metadata", {})

            # Tokenize via chat template
            text = tokenizer.apply_chat_template(
                data["messages"], tokenize=False, add_generation_prompt=False,
            )
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]

            # Find confidence token position: last digit token in the sequence
            conf_pos = None
            for i in range(len(tokens) - 1, max(0, len(tokens) - 15), -1):
                if tokens[i] in digit_set:
                    conf_pos = i
                    break

            if conf_pos is None:
                print(f"WARNING: no confidence digit found in item {line_num}, skipping")
                continue

            # Find prompt/assistant boundary for mask_prompt
            # Tokenize just the user turn to find where assistant starts
            user_only = tokenizer.apply_chat_template(
                [data["messages"][0]], tokenize=False, add_generation_prompt=True,
            )
            prompt_len = len(tokenizer.encode(user_only, add_special_tokens=True))

            items.append({
                "tokens": tokens,
                "conf_pos": conf_pos,
                "correct": float(meta.get("correct", False)),
                "prompt_len": prompt_len,
            })

    print(f"Loaded {len(items)} items from {path}")
    print(f"  Digit token IDs: {digit_ids}")
    return items, digit_ids


# ──────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────

def compute_loss(model, tokens_mx, conf_pos, correct, prompt_len,
                 digit_ids_mx, brier_weight):
    """Compute CE (masked, excluding conf token) + Brier (on conf token).

    Args:
        model: the model
        tokens_mx: (1, seq_len) input token IDs
        conf_pos: int, position of the confidence digit token
        correct: float, 0.0 or 1.0
        prompt_len: int, where assistant tokens start
        digit_ids_mx: mx.array of shape (10,), digit token IDs
        brier_weight: float
    """
    seq_len = tokens_mx.shape[1]

    # Forward pass: input is tokens[:-1], targets are tokens[1:]
    input_ids = tokens_mx[:, :-1]
    targets = tokens_mx[:, 1:]
    logits = model(input_ids)  # (1, seq_len-1, vocab_size)

    # ── CE loss on assistant tokens, excluding confidence position ──
    # Build mask: 1 for assistant tokens, 0 for prompt and conf position
    # In shifted indexing: position i in logits predicts tokens[i+1]
    # So conf_pos in tokens → conf_pos-1 in logits/targets
    conf_logit_pos = conf_pos - 1

    mask = mx.zeros(targets.shape)  # (1, seq_len-1)
    # Set assistant tokens to 1 (prompt_len onwards, shifted by 1)
    # targets[0, i] corresponds to tokens[i+1], predicted by logits[0, i]
    # We want to mask prompt tokens (positions 0 to prompt_len-2 in targets)
    # and unmask assistant tokens (positions prompt_len-1 onwards)
    mask_np = np.zeros((1, seq_len - 1), dtype=np.float32)
    if prompt_len - 1 < seq_len - 1:
        mask_np[0, prompt_len - 1:] = 1.0
    # Zero out the confidence position
    if 0 <= conf_logit_pos < seq_len - 1:
        mask_np[0, conf_logit_pos] = 0.0
    mask = mx.array(mask_np)

    # CE loss
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Gather log probs for target tokens using take_along_axis
    targets_expanded = targets[:, :, None]  # (1, seq_len-1, 1)
    gathered = mx.take_along_axis(log_probs, targets_expanded, axis=-1)  # (1, seq_len-1, 1)
    nll = -gathered.squeeze(-1)  # (1, seq_len-1)
    ce_loss = mx.sum(nll * mask) / mx.maximum(mx.sum(mask), 1.0)

    # ── Brier loss on confidence token ──
    # logits at conf_logit_pos predict the confidence digit
    conf_logits = logits[0, conf_logit_pos, :]  # (vocab_size,)

    # Extract logits for digit tokens only
    digit_logits = conf_logits[digit_ids_mx]  # (10,)

    # Softmax over digits
    digit_probs = mx.softmax(digit_logits)  # (10,)

    # Brier: E_p[(y - c/9)^2] where c in {0,...,9}, y in {0,1}
    scores = mx.arange(10).astype(mx.float32) / 9.0  # [0, 1/9, ..., 1]
    y = mx.array(correct)
    squared_diff = (y - scores) ** 2  # (10,)
    brier_loss = mx.sum(digit_probs * squared_diff)

    total = ce_loss + brier_weight * brier_loss
    return total, ce_loss, brier_loss


# ──────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=None,
                        help="Number of layers to apply LoRA to (default: all)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--brier-weight", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    random.seed(args.seed)

    # ── Load model ──
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load(args.model_path)

    # ── Apply LoRA ──
    # Detect model structure (Gemma wraps in language_model)
    if hasattr(model, 'language_model'):
        inner_layers = model.language_model.model.layers
    elif hasattr(model, 'model'):
        inner_layers = model.model.layers
    else:
        raise ValueError("Cannot find model layers")

    num_layers = args.lora_layers or len(inner_layers)
    lora_cfg = {
        "rank": args.lora_rank,
        "scale": 2.0,
        "dropout": 0.05,
        "keys": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"],
    }
    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_cfg)

    import mlx.utils
    n_train = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
    n_total = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"LoRA: {n_train:,} trainable / {n_total:,} total ({100*n_train/n_total:.3f}%)")

    # ── Load data ──
    train_data, digit_ids = load_dataset(args.train_data, tokenizer, args.max_seq_len)
    val_data = None
    if args.val_data:
        val_data, _ = load_dataset(args.val_data, tokenizer, args.max_seq_len)

    digit_ids_mx = mx.array(digit_ids)

    # ── Optimizer ──
    n_items = len(train_data)
    total_iters = n_items * args.epochs
    n_steps = total_iters // args.grad_accum
    schedule = optim.cosine_decay(args.lr, n_steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    print(f"\nTraining: {n_items} items × {args.epochs} epochs = {total_iters} iters")
    print(f"Grad accum: {args.grad_accum}, effective steps: {n_steps}")
    print(f"Brier weight: {args.brier_weight}")

    # ── Loss function for value_and_grad ──
    def loss_fn(model, tokens_mx, conf_pos, correct, prompt_len):
        total, ce, brier = compute_loss(
            model, tokens_mx, conf_pos, correct, prompt_len,
            digit_ids_mx, args.brier_weight
        )
        return total

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    # ── Training loop ──
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    accum_total = 0.0
    accumulated_grads = None
    t_start = time.time()

    for epoch in range(args.epochs):
        # Shuffle training data each epoch
        random.shuffle(train_data)

        for i, item in enumerate(train_data):
            tokens_mx = mx.array([item["tokens"]])
            conf_pos = item["conf_pos"]
            correct = item["correct"]
            prompt_len = item["prompt_len"]

            # Forward + backward
            loss_val, grads = loss_grad_fn(model, tokens_mx, conf_pos, correct, prompt_len)
            mx.eval(loss_val, grads)

            accum_total += loss_val.item()

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = mlx.utils.tree_map(
                    lambda a, b: a + b, accumulated_grads, grads
                )
                mx.eval(accumulated_grads)

            # Step
            global_iter = epoch * n_items + i + 1
            if global_iter % args.grad_accum == 0:
                # Average gradients
                avg_grads = mlx.utils.tree_map(
                    lambda g: g / args.grad_accum, accumulated_grads
                )
                optimizer.update(model, avg_grads)
                mx.eval(model.parameters(), optimizer.state)

                step += 1
                avg_t = accum_total / args.grad_accum

                if step % args.log_interval == 0 or step == 1:
                    elapsed = time.time() - t_start
                    print(f"Step {step}/{n_steps} | "
                          f"Loss: {avg_t:.4f} | "
                          f"{elapsed:.0f}s")

                if step % args.save_interval == 0:
                    ckpt = args.output_dir / f"step_{step}"
                    ckpt.mkdir(exist_ok=True)
                    weights = dict(model.trainable_parameters())
                    try:
                        from mlx.utils import save_safetensors
                        save_safetensors(str(ckpt / "adapters.safetensors"), weights)
                    except (ImportError, AttributeError):
                        mx.savez(str(ckpt / "adapters.npz"), **weights)
                    print(f"  Saved checkpoint: {ckpt}")

                accum_total = 0.0
                accumulated_grads = None

            # Clear cache periodically
            if global_iter % 100 == 0:
                mx.clear_cache()

    # ── Save final adapter ──
    # Save in safetensors format for mlx_lm compatibility
    weights = dict(model.trainable_parameters())
    # Try safetensors first, fall back to npz
    try:
        from mlx.utils import save_safetensors
        save_safetensors(str(args.output_dir / "adapters.safetensors"), weights)
        print(f"Saved as safetensors")
    except (ImportError, AttributeError):
        try:
            mx.save_safetensors(str(args.output_dir / "adapters.safetensors"), weights)
            print(f"Saved as safetensors")
        except (AttributeError, TypeError):
            mx.savez(str(args.output_dir / "adapters.npz"), **weights)
            print(f"Saved as npz (convert to safetensors for mlx_lm load)")

    # Save adapter_config.json for mlx_lm compatibility
    config = {
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": 2.0,
            "dropout": 0.05,
            "keys": ["self_attn.q_proj", "self_attn.v_proj",
                     "self_attn.k_proj", "self_attn.o_proj"],
        },
    }
    with open(args.output_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Final adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
