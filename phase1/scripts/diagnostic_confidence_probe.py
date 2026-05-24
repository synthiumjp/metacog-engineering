#!/usr/bin/env python3
"""diagnostic_confidence_probe.py — Mechanistic analysis of logit readout.

Extracts hidden states at the confidence token position (the position where
the model is about to predict the confidence digit), trains a logistic
regression probe, and compares probe AUROC₂ to logit AUROC₂.

This diagnoses WHY the logit readout works for some models and not others:
- Probe good + logit good → model encodes AND routes correctness info
- Probe good + logit bad → model encodes but fails to route (LoRA bottleneck)  
- Probe bad + logit bad → model doesn't encode correctness at this position

Usage:
    python3 diagnostic_confidence_probe.py \
        --model-path ~/models/gemma-3-12b-it \
        --adapter-path ~/adapters/adapter_e2e_ce_gentle_seed42 \
        --digit-token-ids "236771,236770,236778,236800,236812,236810,236825,236832,236828,236819" \
        --output results_raw/domain_gen/diagnostic_gemma12b.json

    python3 diagnostic_confidence_probe.py \
        --model-path ~/models/Meta-Llama-3.1-8B-Instruct-bf16 \
        --adapter-path ~/adapters/adapter_e2e_ce_v2 \
        --digit-token-ids "15,16,17,18,19,20,21,22,23,24" \
        --output results_raw/domain_gen/diagnostic_llama8b.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate

SEED = 42


# ── GSM8K loading (same as other scripts) ──

def load_gsm8k_eval(seed=SEED):
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
        })
    random.seed(seed)
    random.shuffle(items)
    return items


def is_correct_gsm8k(predicted, gold):
    try:
        return abs(float(predicted.replace(",", "")) - float(gold)) < 1e-5
    except (ValueError, TypeError):
        return False


def extract_final_number(text):
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*\$?([+-]?\d[\d,]*\.?\d*)",
        r"####\s*([+-]?\d[\d,]*\.?\d*)",
    ]
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


GSM8K_PROMPT = (
    "Solve this math problem step by step. After your solution, "
    "state your confidence as a single digit from 0 (no confidence) to 9 (very confident) "
    "in the format 'Confidence: D'.\n\n"
    "Question: {question}"
)


def find_confidence_position(tokens, digit_token_ids):
    """Find the position just before the confidence digit token.
    
    The hidden state at this position is what the LM head uses to
    predict the confidence digit. This is where correctness info
    must be encoded for the logit readout to work.
    """
    digit_set = set(digit_token_ids)
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in digit_set:
            return i - 1  # position BEFORE the digit (predicts the digit)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--digit-token-ids", required=True,
                        help="Comma-separated digit token IDs (0-9)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which layer's hidden state to extract (-1 = last)")
    args = parser.parse_args()

    digit_token_ids = [int(x) for x in args.digit_token_ids.split(",")]
    assert len(digit_token_ids) == 10, f"Need 10 digit token IDs, got {len(digit_token_ids)}"
    print(f"Digit token IDs: {digit_token_ids}")

    print(f"Loading model from {args.model_path}")
    print(f"Loading adapter from {args.adapter_path}")
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)

    # Determine hidden dimension from model
    # Try common attribute names
    hidden_dim = None
    for attr in ['hidden_size', 'model_dim', 'dim']:
        if hasattr(model.model, attr):
            hidden_dim = getattr(model.model, attr)
            break
        if hasattr(model, attr):
            hidden_dim = getattr(model, attr)
            break
    if hidden_dim is None:
        # Try from config
        try:
            hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
        except:
            pass
    print(f"Hidden dimension: {hidden_dim}")

    eval_items = load_gsm8k_eval()
    if args.max_items:
        eval_items = eval_items[:args.max_items]
    print(f"Evaluating on {len(eval_items)} items")

    hidden_states_list = []
    logit_confs = []
    argmax_confs = []
    corrects = []
    valid_count = 0
    t0 = time.time()

    for i, item in enumerate(eval_items):
        user_msg = GSM8K_PROMPT.format(question=item["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        # Generate response
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens, verbose=False,
        )

        # Check correctness
        answer = extract_final_number(response)
        correct = is_correct_gsm8k(answer, item["gold_answer"])
        corrects.append(correct)

        # Encode full sequence (prompt + response)
        full_text = prompt + response
        tokens = tokenizer.encode(full_text, add_special_tokens=False)

        # Find confidence position
        pos = find_confidence_position(tokens, digit_token_ids)
        if pos is None or pos < 1:
            hidden_states_list.append(None)
            logit_confs.append(float("nan"))
            argmax_confs.append(float("nan"))
            continue

        valid_count += 1

        # Forward pass to get hidden states AND logits
        input_ids = mx.array(tokens[:pos + 1])[None, :]  # up to and including the pre-digit position

        # We need intermediate hidden states, not just logits
        # Use model.model (the transformer) to get hidden states
        # Then model.lm_head to get logits from the hidden state
        
        # Get the hidden state at the confidence position
        # Most MLX models: model.model() returns hidden states, model.lm_head projects to vocab
        try:
            # Try standard MLX model structure
            h = input_ids
            
            # Embed
            if hasattr(model.model, 'embed_tokens'):
                h = model.model.embed_tokens(h)
            elif hasattr(model.model, 'embeddings'):
                h = model.model.embeddings(h)
            
            # Apply transformer layers
            if hasattr(model.model, 'layers'):
                for layer_idx, layer in enumerate(model.model.layers):
                    h = layer(h, mask=None)
                    # Handle tuple output (some layers return (hidden, cache))
                    if isinstance(h, tuple):
                        h = h[0]
            
            # Apply final norm
            if hasattr(model.model, 'norm'):
                h = model.model.norm(h)
            elif hasattr(model.model, 'final_layernorm'):
                h = model.model.final_layernorm(h)
            
            # Extract hidden state at the confidence position (last position)
            hidden_state = h[0, -1, :]  # (hidden_dim,)
            hidden_np = np.array(hidden_state.astype(mx.float32))
            hidden_states_list.append(hidden_np)
            
            # Get logits from the hidden state
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(hidden_state[None, :])  # (1, vocab)
            else:
                logits = model.model.lm_head(hidden_state[None, :])
            logits = logits[0]  # (vocab,)
            
            # Extract digit logits and compute expected confidence
            digit_ids_mx = mx.array(digit_token_ids)
            digit_logits = logits[digit_ids_mx]
            digit_probs = mx.softmax(digit_logits)
            digit_probs_np = np.array(digit_probs.astype(mx.float32))
            
            scores = np.arange(10) / 9.0
            expected_conf = float(np.sum(digit_probs_np * scores) * 100.0)
            logit_confs.append(expected_conf)
            
            argmax_digit = int(np.argmax(digit_probs_np))
            argmax_confs.append(argmax_digit * (100.0 / 9.0))
            
        except Exception as e:
            print(f"  Item {i}: hidden state extraction failed: {e}")
            hidden_states_list.append(None)
            logit_confs.append(float("nan"))
            argmax_confs.append(float("nan"))
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_correct = sum(corrects)
            print(f"  {i+1}/{len(eval_items)} ({elapsed:.0f}s) "
                  f"acc={n_correct/(i+1):.3f} valid_hs={valid_count}")

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {len(eval_items)} items in {elapsed:.0f}s")
    print(f"Valid hidden states: {valid_count}/{len(eval_items)}")

    # ── Train diagnostic probe ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    # Filter to items with valid hidden states
    valid_mask = [h is not None for h in hidden_states_list]
    X = np.array([h for h in hidden_states_list if h is not None])
    y = np.array([c for c, v in zip(corrects, valid_mask) if v], dtype=int)
    logit_c = np.array([l for l, v in zip(logit_confs, valid_mask) if v])

    print(f"\nProbe training: {X.shape[0]} items, {X.shape[1]} features")
    print(f"  Class balance: {y.mean():.3f} correct")

    # Cross-validated probe predictions (5-fold)
    probe = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    probe_preds = cross_val_predict(probe, X, y, cv=5, method='predict_proba')[:, 1]
    probe_auroc = roc_auc_score(y, probe_preds)

    # Logit AUROC₂
    logit_valid = ~np.isnan(logit_c)
    logit_auroc = roc_auc_score(y[logit_valid], logit_c[logit_valid]) if logit_valid.sum() > 10 else float("nan")

    # Correlation between probe and logit
    if logit_valid.sum() > 10:
        from scipy.stats import pearsonr
        r, p = pearsonr(probe_preds[logit_valid], logit_c[logit_valid])
        probe_logit_corr = float(r)
        probe_logit_p = float(p)
    else:
        probe_logit_corr = float("nan")
        probe_logit_p = float("nan")

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC RESULTS")
    print(f"{'='*60}")
    print(f"  Model:              {args.model_path.split('/')[-1]}")
    print(f"  Adapter:            {args.adapter_path.split('/')[-1]}")
    print(f"  Items:              {X.shape[0]}")
    print(f"  Hidden dim:         {X.shape[1]}")
    print(f"  Accuracy:           {y.mean():.3f}")
    print(f"")
    print(f"  Probe AUROC₂:       {probe_auroc:.3f}  (correctness info at confidence position)")
    print(f"  Logit AUROC₂:       {logit_auroc:.3f}  (info routed to digit logits)")
    print(f"  Routing efficiency: {logit_auroc/probe_auroc:.3f}  (logit/probe ratio)")
    print(f"  Probe-logit corr:   r={probe_logit_corr:.3f} (p={probe_logit_p:.2e})")
    print(f"")
    print(f"  INTERPRETATION:")
    if probe_auroc > 0.7 and logit_auroc > 0.7:
        print(f"  → Model ENCODES and ROUTES correctness info. Logit readout works.")
    elif probe_auroc > 0.7 and logit_auroc < 0.6:
        print(f"  → Model ENCODES but FAILS TO ROUTE. LoRA doesn't project to digit logits.")
    elif probe_auroc < 0.6:
        print(f"  → Model does NOT ENCODE correctness at the confidence position.")
    else:
        print(f"  → Intermediate: partial encoding/routing.")
    print(f"{'='*60}")

    # Save
    results = {
        "model": args.model_path,
        "adapter": args.adapter_path,
        "n_items": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "accuracy": float(y.mean()),
        "probe_auroc2": float(probe_auroc),
        "logit_auroc2": float(logit_auroc),
        "routing_efficiency": float(logit_auroc / probe_auroc) if probe_auroc > 0 else None,
        "probe_logit_correlation": probe_logit_corr,
        "probe_logit_p_value": probe_logit_p,
        "elapsed_s": elapsed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Save hidden states for further analysis
    hs_path = args.output.with_name(args.output.stem + "_hidden_states.npz")
    np.savez_compressed(hs_path, X=X, y=y, probe_preds=probe_preds, logit_confs=logit_c)
    print(f"Saved: {hs_path}")


if __name__ == "__main__":
    main()
