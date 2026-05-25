"""
eval_70b_logit.py — Logit readout for Llama 70B balanced confonly adapter.

Tests whether the logit readout rescues 70B confonly from AUROC₂ = 0.674.
Llama supports 101 single-token values (0-100) for fine-mode logit readout.

Usage:
    cd ~/jpwork/metacog-engineering/phase1
    python3 scripts/eval_70b_logit.py

Runs overnight (~6-10 hours on M3 Ultra 512GB).
"""

import json, os, re, sys, time
import numpy as np
from pathlib import Path

MODEL_PATH = os.path.expanduser(
    "~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED"
)
ADAPTER_PATH = "results_raw/finetune/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED/balanced_confonly_v2/adapters"
RESPONSES_PATH = Path("results_raw/step4/balanced_confonly_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED_responses.json")
OUTPUT_PATH = Path("results_raw/domain_gen/logit_70b_confonly.json")

CONFONLY_USER = (
    "You answered the following trivia question.\n"
    "Question: {question}\n"
    "Your answer: {answer}\n"
    "How confident are you that your answer is correct? "
    "State your confidence as a percentage from 0 to 100."
)


def main():
    print("=" * 60)
    print("70B Logit Readout — Balanced Confonly v2")
    print("=" * 60)

    # Load existing responses
    with open(RESPONSES_PATH) as f:
        responses = json.load(f)
    print(f"  Items: {len(responses)}")
    print(f"  Text AUROC₂ (known): 0.674")

    # Load model + adapter
    print(f"\n  Loading 70B model + adapter...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Adapter: {ADAPTER_PATH}")
    t_load = time.time()

    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    print(f"  Loaded in {time.time() - t_load:.0f}s")

    # Detect digit token IDs (0-100)
    digit_ids = {}
    n_single = 0
    for d in range(101):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if len(toks) == 1:
            digit_ids[d] = toks[0]
            n_single += 1
        else:
            # Multi-token — record first token for partial info
            digit_ids[d] = toks[0]

    print(f"  Single-token digits: {n_single}/101")
    if n_single < 10:
        print(f"  WARNING: Few single-token digits. Falling back to coarse mode (0-9).")
        # Recompute for 0-9 only
        digit_ids = {}
        for d in range(10):
            toks = tokenizer.encode(str(d), add_special_tokens=False)
            digit_ids[d] = toks[0]
        mode = "coarse"
        max_digit = 9
    else:
        mode = "fine"
        max_digit = max(d for d in range(101)
                       if len(tokenizer.encode(str(d), add_special_tokens=False)) == 1)

    print(f"  Mode: {mode}, max digit: {max_digit}")
    print(f"  Token IDs: {list(digit_ids.items())[:5]}...")

    # Process items
    results = []
    t0 = time.time()

    for i, resp in enumerate(responses):
        question = resp['question']
        answer = resp.get('answer_text', resp.get('raw_answer', ''))
        correct = resp['correct']
        text_conf = resp['confidence']

        # Construct confonly prompt
        user_msg = CONFONLY_USER.format(question=question, answer=answer)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        # Add "Confidence: " prefix to get logits at the number position
        prompt_with_prefix = prompt + "Confidence: "
        tokens = tokenizer.encode(prompt_with_prefix)

        # Forward pass to get logits
        try:
            x = mx.array([tokens])
            logits = model(x)  # (1, seq_len, vocab_size)
            last_logits = logits[0, -1, :]  # logits predicting next token

            # Extract digit logits
            d_logits = mx.array([last_logits[digit_ids[d]] for d in sorted(digit_ids.keys())])
            d_probs = mx.softmax(d_logits)
            d_probs_np = np.array(d_probs.astype(mx.float32))

            # Expected value
            digits = np.array(sorted(digit_ids.keys()))
            logit_ev = float(np.sum((digits / max_digit) * d_probs_np)) * 100  # scale to 0-100

            # Argmax digit
            argmax_digit = int(digits[np.argmax(d_probs_np)])

            mx.eval(logits)
            del x, logits

        except Exception as e:
            logit_ev = float("nan")
            argmax_digit = -1
            d_probs_np = None
            if i < 5:
                print(f"  Error on item {i}: {e}")

        results.append({
            'id': resp['id'],
            'correct': correct,
            'text_confidence': text_conf,
            'logit_confidence': logit_ev,
            'argmax_digit': argmax_digit,
            'digit_probs': d_probs_np.tolist() if d_probs_np is not None else None,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(responses) - i - 1) / rate / 3600
            valid = [r for r in results if not np.isnan(r['logit_confidence'])]
            if valid:
                from sklearn.metrics import roc_auc_score
                c = np.array([r['correct'] for r in valid])
                lc = np.array([r['logit_confidence'] for r in valid])
                tc = np.array([r['text_confidence'] for r in valid])
                try:
                    l_auc = roc_auc_score(c, lc)
                    t_auc = roc_auc_score(c, tc)
                    print(f"  [{i+1}/{len(responses)}]  "
                          f"text={t_auc:.3f} logit={l_auc:.3f}  "
                          f"ETA={eta:.1f}h  elapsed={elapsed:.0f}s")
                except:
                    print(f"  [{i+1}/{len(responses)}]  "
                          f"ETA={eta:.1f}h  elapsed={elapsed:.0f}s")

        # Periodic save
        if (i + 1) % 100 == 0:
            _save(results, responses)

    # Final save and analysis
    _save(results, responses)
    _analyze(results)


def _save(results, responses):
    """Save intermediate results."""
    from sklearn.metrics import roc_auc_score

    valid = [r for r in results if not np.isnan(r['logit_confidence'])]
    correct = np.array([int(r['correct']) for r in valid])
    logit_conf = np.array([r['logit_confidence'] for r in valid])
    text_conf = np.array([r['text_confidence'] for r in valid])

    summary = {
        "model": "Llama-3.1-70B-Instruct",
        "adapter": "balanced_confonly_v2",
        "n_total": len(results),
        "n_valid": len(valid),
    }

    if len(valid) > 10 and correct.sum() > 0 and correct.sum() < len(correct):
        summary["text_auroc2"] = float(roc_auc_score(correct, text_conf))
        summary["logit_auroc2"] = float(roc_auc_score(correct, logit_conf))
        summary["logit_conf_mean"] = float(logit_conf.mean())
        summary["logit_conf_std"] = float(logit_conf.std())
        summary["text_conf_mean"] = float(text_conf.mean())

    summary["items"] = results

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)


def _analyze(results):
    """Final analysis."""
    from sklearn.metrics import roc_auc_score

    valid = [r for r in results if not np.isnan(r['logit_confidence'])]
    correct = np.array([int(r['correct']) for r in valid])
    logit_conf = np.array([r['logit_confidence'] for r in valid])
    text_conf = np.array([r['text_confidence'] for r in valid])

    print("\n" + "=" * 60)
    print("Final Results — 70B Logit Readout")
    print("=" * 60)
    print(f"  Items: {len(valid)}/{len(results)}")
    print(f"  Accuracy: {correct.mean():.3f}")

    if len(valid) > 10 and correct.sum() > 0:
        t_auc = roc_auc_score(correct, text_conf)
        l_auc = roc_auc_score(correct, logit_conf)
        print(f"  Text AUROC₂:  {t_auc:.3f}")
        print(f"  Logit AUROC₂: {l_auc:.3f}")
        print(f"  Δ: {l_auc - t_auc:+.3f}")
        print(f"  Logit conf mean: {logit_conf.mean():.1f} (std {logit_conf.std():.1f})")
        print(f"  Text conf mean:  {text_conf.mean():.1f}")

        if l_auc > t_auc + 0.03:
            print(f"\n  ✓ Logit readout IMPROVES 70B discrimination")
        elif l_auc > t_auc:
            print(f"\n  ~ Modest logit improvement")
        else:
            print(f"\n  ✗ Logit readout does not help at 70B")

        # Compare to known values
        print(f"\n  Context:")
        print(f"    70B confonly text:  0.674 (known)")
        print(f"    70B baseline:      0.724 (known)")
        print(f"    12B E2E logit:     0.862 (headline)")
        print(f"    If logit > 0.724: 70B gap closed by logit readout")

    print(f"\n  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
