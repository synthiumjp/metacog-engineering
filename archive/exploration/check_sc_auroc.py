import json
import sys
import numpy as np
sys.path.insert(0, r"D:\metacog\scripts")
from utils_phase0 import auroc2

with open(r"D:\metacog\data\step2_teval_difficulty.json") as f:
    teval_diff = json.load(f)
with open(r"D:\metacog\results\baseline\teval_responses.json") as f:
    teval_base = json.load(f)

# Raw self-consistency as confidence signal
sc_conf = np.array([d["n_correct_eval"] / 10.0 for d in teval_diff])
sc_corr = np.array([int(r["correct"]) for r in teval_base])

# Align lengths
n = min(len(sc_conf), len(sc_corr))
a2 = auroc2(sc_conf[:n], sc_corr[:n])

print(f"Raw self-consistency AUROC2 on T-eval: {a2:.3f}")
print(f"Verbal baseline AUROC2: 0.554")
print(f"Entropy AUROC2: 0.701")
print(f"Post-SFT verbal AUROC2: 0.774")
print(f"n={n}")
