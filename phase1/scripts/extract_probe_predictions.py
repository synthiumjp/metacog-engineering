#!/usr/bin/env python3
"""
extract_probe_predictions.py — Re-extract hidden states from saved responses,
train a probe, and dump per-item P(correct) predictions.

Uses the same pipeline as probe_check_domain.py but saves per-item predictions
instead of just summary AUROC₂.

Usage:
    python3 extract_probe_predictions.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --model-name gemma-3-12b-it \
        --responses ~/jpwork/metacog-engineering/phase1/results_raw/domain_gen/responses_gsm8k_gemma-3-12b-it.json \
        --benchmark gsm8k \
        --output ~/jpwork/metacog-engineering/phase1/results_raw/domain_gen/probe_predictions_gsm8k_gemma-3-12b-it.json
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Import model config from existing scripts
sys.path.insert(0, str(Path(__file__).parent))
from model_config import MODEL_LAYERS, get_model_config

SEED = 42


# ──────────────────────────────────────────────────────────────────
# Data loading (mirrors probe_check_domain.py exactly)
# ──────────────────────────────────────────────────────────────────

GSM8K_PROMPT = (
    "Solve this math problem step by step, then state your confidence "
    "as an integer from 0 to 100 on the last line in the format "
    "'Confidence: X%'.\n\nQuestion: {question}"
)


def load_gsm8k_items(seed=SEED):
    """Load GSM8K, shuffle, split. Returns (cal_items, eval_items)."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        match = re.search(r"####\s*(.+)", row["answer"])
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({
            "id": f"gsm8k_{len(items)}",
            "question": row["question"],
            "gold_answer": gold,
            "prompt_template": GSM8K_PROMPT,
        })
    random.seed(seed)
    random.shuffle(items)
    n_cal = 800
    return items[:n_cal], items[n_cal:]


# ──────────────────────────────────────────────────────────────────
# Hidden state extraction (from probe_check_domain.py)
# ──────────────────────────────────────────────────────────────────

def extract_hidden_states(model, tokenizer, prompt, response_text,
                          layer_indices, model_cfg):
    """Extract hidden states at specified layers for the last token."""
    full_text = prompt + response_text
    tokens = tokenizer.encode(full_text)

    x = mx.array([tokens])
    lm = model_cfg["lm"]
    layers = model_cfg["layers"]

    h = lm.model.embed_tokens(x)
    if model_cfg["scale_embeddings"]:
        hidden_size = h.shape[-1]
        h = h * (hidden_size ** 0.5)

    hidden_states = {}
    for i, layer in enumerate(layers):
        h = layer(h, cache=None)
        for label, idx in layer_indices.items():
            if i == idx:
                last_pos = len(tokens) - 1
                hidden_states[f"{label}_last"] = np.array(
                    h[0, last_pos].astype(mx.float32)
                )
    mx.eval(h)
    return hidden_states


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load(args.model_path)
    model_cfg = get_model_config(model, tokenizer, args.model_name)

    # Layer indices (same as probe_check_domain.py)
    n_layers = len(model_cfg["layers"])
    layer_indices = {
        "first": 0,
        "middle": n_layers // 2,
        "last": n_layers - 1,
    }
    print(f"Layer indices: {layer_indices} (of {n_layers} total)")

    # Load responses
    with open(args.responses) as f:
        responses = json.load(f)
    print(f"Loaded {len(responses)} responses")

    # Build response lookup by ID
    resp_by_id = {r["id"]: r for r in responses}

    # Load questions (to reconstruct prompts)
    cal_items, eval_items = load_gsm8k_items()
    all_items = cal_items + eval_items

    # Build cal ID set for split detection
    cal_ids = {it["id"] for it in cal_items}

    # ── Extract hidden states ──
    print(f"\nExtracting hidden states for {len(all_items)} items...")
    t0 = time.time()
    records = []

    for i, item in enumerate(all_items):
        rid = item["id"]
        resp = resp_by_id.get(rid)
        if resp is None:
            continue

        # Reconstruct prompt
        user_msg = GSM8K_PROMPT.format(question=item["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        try:
            hs = extract_hidden_states(
                model, tokenizer, prompt, resp["raw_output"],
                layer_indices, model_cfg
            )
        except Exception as e:
            print(f"  WARNING: failed on {rid}: {e}")
            continue

        if hs is None:
            continue

        records.append({
            "id": rid,
            "correct": bool(resp["correct"]),
            "hidden_states": hs,
            "split": "cal" if rid in cal_ids else "eval",
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(all_items)}] {elapsed:.0f}s")

        # Clear cache periodically
        if (i + 1) % 200 == 0:
            mx.clear_cache()

    elapsed = time.time() - t0
    print(f"Extracted {len(records)} hidden states in {elapsed:.0f}s")

    # ── Train probe on cal split, predict on all ──
    cal_records = [r for r in records if r["split"] == "cal"]
    eval_records = [r for r in records if r["split"] == "eval"]

    # Use middle layer (best from probe_check: AUROC₂ = 0.769)
    probe_key = "middle_last"
    print(f"\nTraining probe on {len(cal_records)} cal items, key={probe_key}")

    X_cal = np.array([r["hidden_states"][probe_key] for r in cal_records])
    y_cal = np.array([int(r["correct"]) for r in cal_records])

    X_eval = np.array([r["hidden_states"][probe_key] for r in eval_records])
    y_eval = np.array([int(r["correct"]) for r in eval_records])

    # Fit probe (same as probe_check_domain.py)
    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            cv=5, random_state=SEED, max_iter=1000,
            scoring="roc_auc",
        )),
    ])
    probe.fit(X_cal, y_cal)

    # Predict P(correct) for ALL items
    X_all = np.array([r["hidden_states"][probe_key] for r in records])
    p_correct_all = probe.predict_proba(X_all)[:, 1]

    # Eval AUROC₂ (sanity check)
    p_eval = probe.predict_proba(X_eval)[:, 1]
    eval_auroc = roc_auc_score(y_eval, p_eval)
    print(f"Probe AUROC₂ on eval: {eval_auroc:.3f} (expected ~0.769)")

    # ── Save per-item predictions ──
    output = []
    for r, p in zip(records, p_correct_all):
        output.append({
            "id": r["id"],
            "correct": r["correct"],
            "probe_pcorrect": round(float(p), 4),
            "split": r["split"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(output)} per-item predictions to {args.output}")

    # Stats
    p_arr = np.array([o["probe_pcorrect"] for o in output])
    print(f"P(correct) stats: mean={p_arr.mean():.3f}, std={p_arr.std():.3f}, "
          f"min={p_arr.min():.3f}, max={p_arr.max():.3f}")


if __name__ == "__main__":
    main()
