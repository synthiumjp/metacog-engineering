import json

with open(r"D:\metacog\results\evaluation\ft_meval_responses.json") as f:
    ft = json.load(f)
with open(r"D:\metacog\results\baseline\meval_responses.json") as f:
    base = json.load(f)

# Check what the parser extracted vs what would be correct
correct_count = 0
for r in ft[:20]:
    resp = r["response"].strip().lower()
    first_char = resp[0] if resp else ""
    print(f"  parsed_answer='{r['answer']}' first_char='{first_char}' correct={r['correct']} conf={r['confidence']}")

# Check baseline format for comparison
print("\nBaseline samples:")
for r in base[:5]:
    print(f"  answer='{r['parsed_answer']}' correct={r['correct']}")
