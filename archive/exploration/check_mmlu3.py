import json

with open(r"D:\metacog\results\evaluation\ft_meval_responses.json") as f:
    ft = json.load(f)
with open(r"D:\metacog\results\baseline\meval_responses.json") as f:
    base = json.load(f)

# Get gold answers from baseline (which has correct parsing)
# We need the gold letters - check baseline structure
print("Baseline keys:", list(base[0].keys()))

# Manual re-parse: extract first letter from response
import numpy as np

correct = 0
total = 0
conf_correct = []
conf_incorrect = []

for i, r in enumerate(ft):
    resp = r["response"].strip()
    if not resp:
        continue
    # Extract first letter
    first = resp[0].upper()
    if first not in "ABCD":
        continue
    
    # Get gold from baseline
    gold = base[i].get("parsed_answer", "") if i < len(base) else ""
    if not gold:
        gold = base[i].get("answer", "")
    
    is_correct = (first == gold.upper())
    total += 1
    if is_correct:
        correct += 1
        conf_correct.append(r["confidence"])
    else:
        conf_incorrect.append(r["confidence"])

print(f"\nRe-parsed MMLU accuracy: {correct}/{total} = {correct/total:.3f}")
print(f"Baseline accuracy: {np.mean([r['correct'] for r in base]):.3f}")
print(f"Mean conf (correct): {np.nanmean(conf_correct):.1f}")
print(f"Mean conf (incorrect): {np.nanmean(conf_incorrect):.1f}")

from utils_phase0 import auroc2
conf_all = []
corr_all = []
for i, r in enumerate(ft):
    resp = r["response"].strip()
    if not resp:
        continue
    first = resp[0].upper()
    if first not in "ABCD":
        continue
    gold = base[i].get("parsed_answer", base[i].get("answer", ""))
    conf_all.append(r["confidence"])
    corr_all.append(int(first == gold.upper()))

a = auroc2(np.array(conf_all), np.array(corr_all))
print(f"Re-parsed AUROC2: {a:.3f}")
