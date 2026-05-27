import json, sys, numpy as np
sys.path.insert(0, r"D:\metacog\scripts")
from utils_phase0 import auroc2

with open(r"D:\metacog\results\evaluation\ft_meval_responses.json") as f:
    ft = json.load(f)
with open(r"D:\metacog\results\baseline\meval_responses.json") as f:
    base = json.load(f)

conf_all = []
corr_all = []
for i, r in enumerate(ft):
    resp = r["response"].strip()
    if not resp:
        continue
    first = resp[0].upper()
    if first not in "ABCD":
        continue
    gold = base[i].get("parsed_answer", "")
    conf_all.append(r["confidence"])
    corr_all.append(int(first == gold.upper()))

a = auroc2(np.array(conf_all), np.array(corr_all))
print(f"Re-parsed MMLU AUROC2: {a:.3f}")
print(f"Baseline MMLU AUROC2: 0.535")
print(f"Delta: {a - 0.535:.3f}")
print(f"n={len(conf_all)}")
