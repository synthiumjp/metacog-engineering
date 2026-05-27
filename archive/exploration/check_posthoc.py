import json
import numpy as np
from collections import Counter

with open(r"D:\metacog\results\evaluation\ft_teval_responses.json") as f:
    ft = json.load(f)

confs = [r["confidence"] for r in ft if not np.isnan(r["confidence"])]
print("Confidence value counts:")
for val, count in sorted(Counter(confs).items()):
    print(f"  {val}: {count}")

# Accuracy by confidence value
for val in sorted(set(confs)):
    items = [r for r in ft if r["confidence"] == val]
    acc = np.mean([r["correct"] for r in items])
    print(f"  conf={val}: n={len(items)} accuracy={acc:.3f}")
