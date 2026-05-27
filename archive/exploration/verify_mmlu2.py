import json
import sys
import numpy as np
sys.path.insert(0, r"D:\metacog\scripts")
from utils_phase0 import auroc2

with open(r"D:\metacog\results\evaluation\ft_meval_responses.json") as f:
    ft = json.load(f)
with open(r"D:\metacog\results\baseline\meval_responses.json") as f:
    base = json.load(f)

# Use each model's BEST parser, not the same parser
# Baseline: original parser worked (parsed_answer field is correct)
# Post-SFT: case-insensitive first-letter parser needed

def reparse_letter(response):
    resp = response.strip()
    if not resp:
        return ""
    first = resp[0].upper()
    if first in "ABCD":
        return first
    return ""

# Baseline: use original parser results
base_acc = np.mean([r["correct"] for r in base])
base_conf = np.array([r["parsed_confidence"] for r in base])
base_corr = np.array([int(r["correct"]) for r in base])
base_a2 = auroc2(base_conf, base_corr)

# Post-SFT: use case-insensitive reparse
ft_conf = []
ft_corr = []
ft_correct = 0
ft_total = 0
for i, r in enumerate(ft):
    letter = reparse_letter(r["response"])
    gold = base[i]["parsed_answer"]
    if not letter:
        continue
    ft_total += 1
    is_correct = (letter == gold.upper())
    if is_correct:
        ft_correct += 1
    ft_conf.append(r["confidence"])
    ft_corr.append(int(is_correct))

ft_acc = ft_correct / ft_total
ft_a2 = auroc2(np.array(ft_conf), np.array(ft_corr))

print("=== MMLU: each model with its appropriate parser ===")
print(f"Baseline: accuracy={base_acc:.3f} AUROC2={base_a2:.3f} (n={len(base)})")
print(f"Post-SFT: accuracy={ft_acc:.3f} AUROC2={ft_a2:.3f} (n={ft_total})")
print(f"Delta accuracy: {ft_acc - base_acc:+.3f}")
print(f"Delta AUROC2: {ft_a2 - base_a2:+.3f}")

# Verify: show some baseline raw responses to understand format
print("\n=== Baseline response format (first 10) ===")
for r in base[:10]:
    print(f"  parsed='{r['parsed_answer']}' correct={r['correct']} raw='{r['raw_response'][:80]}'")

# Show post-SFT format
print("\n=== Post-SFT response format (first 10) ===")
for r in ft[:10]:
    letter = reparse_letter(r["response"])
    print(f"  reparsed='{letter}' conf={r['confidence']} raw='{r['response'][:80]}'")
