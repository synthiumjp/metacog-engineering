import json
import numpy as np

with open(r"D:\metacog\results\evaluation\ft_teval_responses.json") as f:
    ft = json.load(f)

ft_corr = [r["correct"] for r in ft]
ft_conf = [r["confidence"] for r in ft]

print(f"Post-SFT T-eval:")
print(f"  accuracy={np.mean(ft_corr):.3f}")
print(f"  mean conf (correct): {np.nanmean([c for c,y in zip(ft_conf,ft_corr) if y]):.1f}")
print(f"  mean conf (incorrect): {np.nanmean([c for c,y in zip(ft_conf,ft_corr) if not y]):.1f}")
print(f"  conf histogram: {np.histogram([c for c in ft_conf if not np.isnan(c)], bins=[0,10,20,30,40,50,60,70,80,90,100])[0]}")

print(f"\nSample responses:")
for r in ft[:5]:
    qid = r.get("question_id", "?")
    print(f"  correct={r['correct']} conf={r['confidence']} ans={r['answer'][:80]}")
