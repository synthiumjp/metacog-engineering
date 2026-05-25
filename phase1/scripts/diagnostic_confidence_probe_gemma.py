"""
diagnostic_confidence_probe_gemma.py — Fixed diagnostic probe for Gemma MLX models.

The existing diagnostic_confidence_probe.py assumes model.model.layers (Llama/Qwen
structure). Gemma MLX models use model.layers directly. This version auto-detects
the model structure.

Runs the same diagnostic as Llama 8B: loads an E2E adapter, generates GSM8K
responses, extracts hidden states at the confidence token position, trains
a probe on those hidden states, and measures routing efficiency.

This fills in the diagnostic table with actual Gemma 12B numbers instead of
"implied ≥0.86".

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/diagnostic_confidence_probe_gemma.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --adapter-path finetune/gemma-3-12b-it/brier_e2e_gsm8k/adapter_e2e_ce_gentle_seed42 \
        --output results_raw/domain_gen/diagnostic_gemma12b_gsm8k.json

Environment:
    source ~/jpwork/.venv_metacog/bin/activate
    Hardware: M3 Ultra 512GB, MLX
"""

import argparse, json, os, re, sys, time
import numpy as np

SEED = 42
MAX_TOKENS = 512
CONFIDENCE_DIGIT_PATTERN = re.compile(r"\*{0,2}Confidence\*{0,2}\s*:?\*{0,2}\s*(\d)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Model structure detection
# ---------------------------------------------------------------------------
def get_model_internals(model):
    """Auto-detect model structure for MLX models.

    Gemma 3 MLX structure:
        model.language_model.model.embed_tokens  (Embedding)
        model.language_model.model.layers         (list of 48)
        model.language_model.model.norm            (RMSNorm)

    Standard (Llama/Qwen) MLX structure:
        model.model.embed_tokens
        model.model.layers
    """
    # Gemma 3 MLX: model.language_model.model.{embed_tokens, layers}
    if hasattr(model, 'language_model'):
        lm = model.language_model
        inner = getattr(lm, 'model', None)
        if inner is not None:
            embed = getattr(inner, 'embed_tokens', None)
            layers = getattr(inner, 'layers', None)
            if embed is not None and layers is not None:
                return {
                    "embed_fn": embed,
                    "layers": layers,
                    "n_layers": len(layers),
                    "model_type": "gemma3",
                    "scale_embeddings": True,
                }

    # Standard structure: model.model.layers (Llama, Qwen)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return {
            "embed_fn": model.model.embed_tokens,
            "layers": model.model.layers,
            "n_layers": len(model.model.layers),
            "model_type": "standard",
            "scale_embeddings": False,
        }

    # Flat: model.layers + model.embed_tokens
    if hasattr(model, 'layers') and hasattr(model, 'embed_tokens'):
        return {
            "embed_fn": model.embed_tokens,
            "layers": model.layers,
            "n_layers": len(model.layers),
            "model_type": "flat",
            "scale_embeddings": False,
        }

    raise RuntimeError(
        f"Cannot detect model structure. "
        f"Top-level attrs: {[a for a in dir(model) if not a.startswith('_')]}"
    )

# ---------------------------------------------------------------------------
# GSM8K data
# ---------------------------------------------------------------------------
def load_gsm8k_eval(n_eval=519, seed=SEED):
    """Load GSM8K test set for evaluation (same split as E2E training)."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        answer_text = row["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({
            "id": f"gsm8k_{len(items)}",
            "question": row["question"],
            "gold_answer": gold,
        })

    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    return items[:n_eval]

def is_correct_gsm8k(predicted, gold):
    # Strip everything from "Confidence" onward so the confidence digit
    # doesn't get picked up as the answer
    text = re.split(r"\*{0,2}Confidence\*{0,2}", predicted, flags=re.IGNORECASE)[0]
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if not numbers:
        return False
    pred = numbers[-1].replace(",", "")
    gold_clean = gold.replace(",", "").strip()
    try:
        return float(pred) == float(gold_clean)
    except ValueError:
        return pred.strip() == gold_clean.strip()

# ---------------------------------------------------------------------------
# Generation + hidden state extraction
# ---------------------------------------------------------------------------
GSM8K_E2E_PROMPT = (
    "Solve this math problem step by step, then state your confidence "
    "as a single digit from 0 (no confidence) to 9 (very confident).\n\n"
    "Question: {question}"
)

def generate_and_extract(model, tokenizer, items, internals, adapter_loaded=True):
    """Generate responses with adapter, extract hidden states at confidence position."""
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate

    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        user_msg = GSM8K_E2E_PROMPT.format(question=item["question"])
        messages = [{"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        response = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_TOKENS, verbose=False,
        )

        # Parse answer and confidence
        correct = is_correct_gsm8k(response, item["gold_answer"])
        conf_match = CONFIDENCE_DIGIT_PATTERN.search(response)
        conf_digit = int(conf_match.group(1)) if conf_match else -1

        # Find confidence token position and extract hidden states
        full_text = prompt + response
        tokens = tokenizer.encode(full_text)
        prompt_len = len(tokenizer.encode(prompt))
        gen_tokens = tokens[prompt_len:]

        # Find the confidence digit token
        # Look for "Confidence: X" pattern in generated tokens
        conf_token_pos = find_confidence_position(tokenizer, gen_tokens, response)

        hidden_state = None
        if conf_token_pos >= 0:
            # Extract hidden state at the confidence position
            abs_pos = prompt_len + conf_token_pos
            hidden_state = extract_hidden_at_position(
                model, tokenizer, tokens, abs_pos, internals
            )

        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold": item["gold_answer"],
            "response": response,
            "correct": correct,
            "conf_digit": conf_digit,
            "conf_token_pos": conf_token_pos,
            "hidden_state": hidden_state,
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in results])
            hs_rate = np.mean([r["hidden_state"] is not None for r in results])
            print(f"  [{i+1}/{len(items)}]  acc={acc:.3f}  "
                  f"hs_rate={hs_rate:.2f}  elapsed={elapsed:.0f}s")

    return results


def find_confidence_position(tokenizer, gen_tokens, response_text):
    """Find the position of the confidence digit token in generated tokens."""
    # Strategy: find "Confidence:" in the response, then the digit after it
    match = CONFIDENCE_DIGIT_PATTERN.search(response_text)
    if not match:
        return -1

    # Tokenize up to the confidence digit
    prefix = response_text[:match.start() + len(match.group(0))]
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

    # The confidence digit should be near the end of prefix_tokens
    # Return position of last token (the digit)
    if len(prefix_tokens) > 0 and len(prefix_tokens) <= len(gen_tokens):
        return len(prefix_tokens) - 1

    # Fallback: search for digit token directly
    digit_str = match.group(1)
    digit_token_ids = tokenizer.encode(digit_str, add_special_tokens=False)
    if digit_token_ids:
        target_id = digit_token_ids[0]
        # Search from the end
        for j in range(len(gen_tokens) - 1, -1, -1):
            if gen_tokens[j] == target_id:
                return j

    return -1


def extract_hidden_at_position(model, tokenizer, all_tokens, position, internals):
    """Extract hidden state at a specific token position across key layers."""
    import mlx.core as mx

    x = mx.array([all_tokens[:position + 1]])
    h = internals["embed_fn"](x)

    if internals.get("scale_embeddings"):
        hidden_size = h.shape[-1]
        h = h * (hidden_size ** 0.5)

    layers = internals["layers"]
    n_layers = internals["n_layers"]

    # Extract at first, middle, last layers
    target_indices = {
        "first": 0,
        "middle": n_layers // 2,
        "last": n_layers - 1,
    }

    hidden_states = {}
    for i, layer in enumerate(layers):
        h = layer(h, cache=None)
        for label, idx in target_indices.items():
            if i == idx:
                # Last position = the confidence token
                hidden_states[f"{label}_last"] = np.array(
                    h[0, -1].astype(mx.float32)
                )

    mx.eval(h)
    del h, x
    return hidden_states

# ---------------------------------------------------------------------------
# Probe training and diagnostic
# ---------------------------------------------------------------------------
def run_diagnostic(results):
    """Train probe on hidden states at confidence position, compute routing efficiency."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    print("\n--- Diagnostic Probe Analysis ---")

    # Filter to items with hidden states
    items_with_hs = [r for r in results if r["hidden_state"] is not None]
    print(f"  Items with hidden states: {len(items_with_hs)}/{len(results)}")

    if len(items_with_hs) < 50:
        print("  ERROR: insufficient items with hidden states")
        return None

    correct_arr = np.array([int(r["correct"]) for r in items_with_hs])
    conf_digits = np.array([r["conf_digit"] for r in items_with_hs])

    print(f"  Accuracy: {np.mean(correct_arr):.3f}")
    print(f"  Conf digit distribution: {np.bincount(conf_digits.clip(0, 9), minlength=10)}")

    diagnostic = {}

    for layer_key in ["first_last", "middle_last", "last_last"]:
        X = []
        y = []
        confs = []
        for r in items_with_hs:
            if layer_key in r["hidden_state"]:
                X.append(r["hidden_state"][layer_key])
                y.append(int(r["correct"]))
                confs.append(r["conf_digit"])

        X = np.array(X)
        y = np.array(y)
        confs = np.array(confs)

        if len(X) < 50:
            print(f"  {layer_key}: insufficient data ({len(X)})")
            continue

        # 5-fold CV probe
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)

        clf = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
            cv=5, penalty="l2", max_iter=2000,
            scoring="roc_auc", random_state=SEED,
        )
        clf.fit(X_s, y)

        probe_scores = clf.predict_proba(X_s)[:, 1]
        probe_auc = roc_auc_score(y, probe_scores)

        # Logit-based confidence
        logit_conf = confs / 9.0  # normalise to 0-1
        mask = confs >= 0
        if mask.sum() > 10 and y[mask].sum() > 0 and y[mask].sum() < mask.sum():
            logit_auc = roc_auc_score(y[mask], logit_conf[mask])
        else:
            logit_auc = float("nan")

        # Routing efficiency: logit_auc / probe_auc (how much probe signal reaches output)
        routing = (logit_auc - 0.5) / (probe_auc - 0.5) if probe_auc > 0.5 else float("nan")

        # Probe-logit correlation
        from scipy.stats import pearsonr
        if mask.sum() > 10:
            r_val, p_val = pearsonr(probe_scores[mask], logit_conf[mask])
        else:
            r_val, p_val = float("nan"), float("nan")

        diagnostic[layer_key] = {
            "probe_auroc2": probe_auc,
            "logit_auroc2": logit_auc,
            "routing_efficiency": routing,
            "probe_logit_r": r_val,
            "probe_logit_p": p_val,
            "n_items": len(X),
            "best_C": float(clf.C_[0]),
        }

        print(f"  {layer_key}:")
        print(f"    Probe AUROC₂:      {probe_auc:.3f}")
        print(f"    Logit AUROC₂:      {logit_auc:.3f}")
        print(f"    Routing efficiency: {routing:.3f}")
        print(f"    Probe-logit r:     {r_val:.3f} (p={p_val:.2e})")

    return diagnostic

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True,
                        help="Path to E2E GSM8K adapter (e.g. adapter_e2e_ce_gentle_seed42)")
    parser.add_argument("--output", default="results_raw/domain_gen/diagnostic_gemma12b_gsm8k.json")
    parser.add_argument("--n-eval", type=int, default=519)
    args = parser.parse_args()

    print("=" * 60)
    print("Diagnostic Confidence Probe — Gemma 12B GSM8K")
    print("=" * 60)

    import mlx.core as mx
    from mlx_lm import load

    # Load model with adapter
    print(f"Loading model: {args.model_path}")
    print(f"Loading adapter: {args.adapter_path}")
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)

    # Detect model structure
    internals = get_model_internals(model)
    print(f"  Model type: {internals['model_type']}")
    print(f"  Layers: {internals['n_layers']}")
    print(f"  Scale embeddings: {internals['scale_embeddings']}")

    # Load eval data
    items = load_gsm8k_eval(n_eval=args.n_eval)
    print(f"  Eval items: {len(items)}")

    # Generate and extract
    results = generate_and_extract(model, tokenizer, items, internals)

    # Run diagnostic
    diagnostic = run_diagnostic(results)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Strip hidden states from JSON (save separately)
    results_json = [{k: v for k, v in r.items() if k != "hidden_state"} for r in results]
    output = {
        "model": args.model_path,
        "adapter": args.adapter_path,
        "n_items": len(results),
        "accuracy": float(np.mean([r["correct"] for r in results])),
        "diagnostic": diagnostic,
        "items": results_json,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Save hidden states
    hs_path = args.output.replace(".json", "_hidden_states.npz")
    hs_dict = {}
    for r in results:
        if r["hidden_state"] is not None:
            for key, arr in r["hidden_state"].items():
                hs_dict[f"{r['id']}_{key}"] = arr
    np.savez_compressed(hs_path, **hs_dict)
    print(f"Saved: {hs_path} ({len(hs_dict)} entries)")

    # Print paper-ready summary
    print("\n" + "=" * 60)
    print("Paper-ready diagnostic table row:")
    print("=" * 60)
    if diagnostic:
        best = diagnostic.get("middle_last", diagnostic.get("last_last", {}))
        print(f"  Gemma 12B GSM8K E2E | "
              f"Probe: {best.get('probe_auroc2', '?'):.3f} | "
              f"Logit: {best.get('logit_auroc2', '?'):.3f} | "
              f"Routing: {best.get('routing_efficiency', '?'):.3f} | "
              f"r: {best.get('probe_logit_r', '?'):.3f}")


if __name__ == "__main__":
    main()
