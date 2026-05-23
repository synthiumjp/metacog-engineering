#!/usr/bin/env python3
"""
train_brier_patch.py — Run stock mlx_lm LoRA training with Brier loss patched in.

Monkey-patches the default_loss function to add tokenized Brier score
on the confidence token position. All memory management, LoRA, data loading,
and gradient accumulation stays stock.

Usage:
    python3 train_brier_patch.py \
        --config /path/to/config.yaml \
        --brier-weight 1.0
"""

import argparse
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.tuner.trainer as trainer

# Gemma 3 digit token IDs (0-9)
DIGIT_TOKEN_IDS = [236771, 236770, 236778, 236800, 236812, 236810,
                   236825, 236832, 236828, 236819]
# Map digit token ID → digit value
DIGIT_TOKEN_TO_VALUE = {tid: d for d, tid in enumerate(DIGIT_TOKEN_IDS)}
DIGIT_IDS_SET = set(DIGIT_TOKEN_IDS)

BRIER_WEIGHT = 1.0  # set from CLI


def brier_loss_for_item(logits_at_pos, correctness):
    """Compute tokenized Brier score for a single position.

    Args:
        logits_at_pos: (vocab_size,) logits at the confidence token position
        correctness: float, 0.0 or 1.0
    Returns:
        scalar Brier loss
    """
    digit_ids = mx.array(DIGIT_TOKEN_IDS)
    digit_logits = logits_at_pos[digit_ids]  # (10,)
    digit_probs = mx.softmax(digit_logits)  # (10,)

    # Scores: [0/9, 1/9, ..., 9/9] = [0.0, 0.111, ..., 1.0]
    scores = mx.arange(10).astype(mx.float32) / 9.0
    y = mx.array(correctness)
    squared_diff = (y - scores) ** 2
    return mx.sum(digit_probs * squared_diff)


def find_conf_position_and_correctness(targets_row):
    """Find confidence digit position and correctness from target tokens.

    Searches backward for the last digit token (0-9).
    For binary targets: digit 9 → correct=1.0, digit 0 → correct=0.0.
    For any digit d: correctness = d/9.

    Args:
        targets_row: 1D array of target token IDs (already shifted)
    Returns:
        (position, correctness) or (None, None) if no digit found
    """
    targets_list = targets_row.tolist()
    for i in range(len(targets_list) - 1, max(0, len(targets_list) - 15), -1):
        tid = targets_list[i]
        if tid in DIGIT_IDS_SET:
            digit_val = DIGIT_TOKEN_TO_VALUE[tid]
            correctness = digit_val / 9.0
            return i, correctness
    return None, None


def patched_loss(model, batch, lengths):
    """Stock CE loss + tokenized Brier score on confidence token (pure MLX ops)."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)

    # ── Standard CE (same as stock) ──
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce_scalar = ce.astype(mx.float32).sum() / ntoks

    # ── Brier loss on confidence tokens (pure MLX, no .tolist()) ──
    digit_ids = mx.array(DIGIT_TOKEN_IDS)  # (10,)
    batch_size = targets.shape[0]
    seq_len = targets.shape[1]

    # Find which positions have digit tokens as targets
    is_digit = mx.zeros(targets.shape, dtype=mx.float32)
    for did in DIGIT_TOKEN_IDS:
        is_digit = is_digit + (targets == did).astype(mx.float32)

    # Last digit position per item (multiply by position index, argmax)
    positions = mx.arange(seq_len).astype(mx.float32)[None, :]
    masked_pos = positions * is_digit + (-1.0) * (1.0 - is_digit)
    conf_positions = mx.argmax(masked_pos, axis=1)  # (batch,)

    # Get target token at confidence position
    conf_pos_1d = conf_positions[:, None]  # (batch, 1)
    target_at_conf = mx.take_along_axis(targets, conf_pos_1d.astype(mx.int32), axis=1).squeeze(1)

    # Map target token → correctness (digit_value / 9.0)
    correctness = mx.zeros(batch_size)
    for d in range(10):
        correctness = mx.where(target_at_conf == DIGIT_TOKEN_IDS[d], d / 9.0, correctness)

    # Gather logits at confidence positions
    conf_pos_3d = mx.broadcast_to(
        conf_positions[:, None, None].astype(mx.int32),
        (batch_size, 1, logits.shape[-1])
    )
    conf_logits = mx.take_along_axis(logits, conf_pos_3d, axis=1).squeeze(1)  # (batch, vocab)

    # Extract digit logits, softmax, Brier
    digit_logits = conf_logits[:, digit_ids]  # (batch, 10)
    digit_probs = mx.softmax(digit_logits, axis=1)
    scores = mx.arange(10).astype(mx.float32) / 9.0
    squared_diff = (correctness[:, None] - scores[None, :]) ** 2
    brier_avg = mx.mean(mx.sum(digit_probs * squared_diff, axis=1))

    total_loss = ce_scalar + BRIER_WEIGHT * brier_avg
    return total_loss, ntoks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to LoRA config YAML")
    parser.add_argument("--brier-weight", type=float, default=1.0)
    args = parser.parse_args()

    global BRIER_WEIGHT
    BRIER_WEIGHT = args.brier_weight
    print(f"Brier weight: {BRIER_WEIGHT}")

    # Monkey-patch: wrap trainer.train to inject our loss function
    original_train = trainer.train
    def wrapped_train(*args, **kwargs):
        kwargs['loss'] = patched_loss
        return original_train(*args, **kwargs)
    trainer.train = wrapped_train
    # Also patch the module-level reference for evaluate
    trainer.default_loss = patched_loss
    print("Patched train() to inject Brier loss")

    # Run stock mlx_lm lora
    sys.argv = ["mlx_lm", "-c", args.config]
    from mlx_lm.lora import main as lora_main
    lora_main()


if __name__ == "__main__":
    main()
