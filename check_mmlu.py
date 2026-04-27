import json

with open(r"D:\metacog\results\evaluation\ft_meval_responses.json") as f:
    ft = json.load(f)

print("Sample MMLU responses:")
for r in ft[:5]:
    print(f"  correct={r['correct']} conf={r['confidence']}")
    print(f"  response: {r['response'][:150]}")
    print()
