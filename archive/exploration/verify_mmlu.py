import json
import sys
import numpy as np
sys.path.insert(0, r"D:\metacog\scripts")
from utils_phase0 import auroc2

with open(r"D:\metacog\results\evaluation\ft_meval_responses.json") as f:
    ft = json.load(f)
with open(r"D:\metacog\results\baseline\meval_responses.json") as f:
    base = json.load(f)

def reparse_letter(response):
    """Case-insensitive letter extraction from response."""
    resp = response.strip()
    if not resp:
        return ""
    first = resp[0].upper()
    if first in "ABCD":
        return first
    return ""

# Re-parse BOTH baseline and post-SFT with identical parser
print("=== Re-parsing both with identical case-insensitive parser ===\n")

# Baseline
base_reparsed_correct = 0
base_reparsed_total = 0
base_conf_correct = []
base_conf_incorrect = []
base_unparseable = 0

for r in base:
    gold = r["parsed_answer"]  # baseline parser worked, so this is the gold
    letter = reparse_letter(r["raw_response"])
    if not letter:
        base_unparseable += 1
        continue
    base_reparsed_total += 1
    is_correct = (letter == gold.upper())
    if is_correct:
        base_reparsed_correct += 1
        base_conf_correct.append(r["parsed_confidence"])
    else:
        base_conf_incorrect.append(r["parsed_confidence"])

print(f"BASELINE (re-parsed):")
print(f"  parsed: {base_reparsed_total}, unparseable: {base_unparseable}")
print(f"  accuracy: {base_reparsed_correct}/{base_reparsed_total} = {base_reparsed_correct/base_reparsed_total:.3f}")
print(f"  original accuracy: {np.mean([r['correct'] for r in base]):.3f}")
print(f"  mean conf correct: {np.nanmean(base_conf_correct):.1f}")
print(f"  mean conf incorrect: {np.nanmean(base_conf_incorrect):.1f}")

# Post-SFT
ft_reparsed_correct = 0
ft_reparsed_total = 0
ft_conf_correct = []
ft_conf_incorrect = []
ft_unparseable = 0
ft_conf_all = []
ft_corr_all = []

for i, r in enumerate(ft):
    gold = base[i]["parsed_answer"]
    letter = reparse_letter(r["response"])
    if not letter:
        ft_unparseable += 1
        continue
    ft_reparsed_total += 1
    is_correct = (letter == gold.upper())
    ft_conf_all.append(r["confidence"])
    ft_corr_all.append(int(is_correct))
    if is_correct:
        ft_reparsed_correct += 1
        ft_conf_correct.append(r["confidence"])
    else:
        ft_conf_incorrect.append(r["confidence"])

ft_auroc2 = auroc2(np.array(ft_conf_all), np.array(ft_corr_all))

# Also compute baseline AUROC2 with reparsed data
base_conf_all_rp = []
base_corr_all_rp = []
for r in base:
    letter = reparse_letter(r["raw_response"])
    if not letter:
        continue
    gold = r["parsed_answer"]
    base_conf_all_rp.append(r["parsed_confidence"])
    base_corr_all_rp.append(int(letter == gold.upper()))

base_auroc2_rp = auroc2(np.array(base_conf_all_rp), np.array(base_corr_all_rp))

print(f"\nPOST-SFT (re-parsed):")
print(f"  parsed: {ft_reparsed_total}, unparseable: {ft_unparseable}")
print(f"  accuracy: {ft_reparsed_correct}/{ft_reparsed_total} = {ft_reparsed_correct/ft_reparsed_total:.3f}")
print(f"  mean conf correct: {np.nanmean(ft_conf_correct):.1f}")
print(f"  mean conf incorrect: {np.nanmean(ft_conf_incorrect):.1f}")
print(f"  AUROC2: {ft_auroc2:.3f}")

print(f"\n=== COMPARISON (same parser) ===")
print(f"  Baseline accuracy: {base_reparsed_correct/base_reparsed_total:.3f} (n={base_reparsed_total})")
print(f"  Post-SFT accuracy: {ft_reparsed_correct/ft_reparsed_total:.3f} (n={ft_reparsed_total})")
print(f"  Baseline AUROC2: {base_auroc2_rp:.3f}")
print(f"  Post-SFT AUROC2: {ft_auroc2:.3f}")
print(f"  Delta accuracy: {ft_reparsed_correct/ft_reparsed_total - base_reparsed_correct/base_reparsed_total:+.3f}")
print(f"  Delta AUROC2: {ft_auroc2 - base_auroc2_rp:+.3f}")

# Audit: show 10 cases where baseline and post-SFT disagree
print(f"\n=== AUDIT: 10 sample comparisons ===")
count = 0
for i, (b, f) in enumerate(zip(base, ft)):
    b_letter = reparse_letter(b["raw_response"])
    f_letter = reparse_letter(f["response"])
    gold = b["parsed_answer"]
    if b_letter != f_letter and count < 10:
        print(f"  Item {i}: gold={gold}")
        print(f"    base: '{b_letter}' (conf={b['parsed_confidence']}) correct={b_letter==gold}")
        print(f"    ft:   '{f_letter}' (conf={f['confidence']}) correct={f_letter==gold}")
        print(f"    ft_response: {f['response'][:100]}")
        count += 1
