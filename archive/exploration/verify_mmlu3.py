import json
import numpy as np

with open(r"D:\metacog\results\baseline\meval_responses.json") as f:
    base = json.load(f)

# Check: does baseline ever produce lowercase answers?
lowercase_count = 0
for r in base:
    raw = r["raw_response"].strip()
    if raw and raw[0] in "abcd":
        lowercase_count += 1
        
print(f"Baseline responses starting with lowercase a-d: {lowercase_count}/{len(base)}")
print(f"Baseline responses starting with 'Answer:': {sum(1 for r in base if r['raw_response'].strip().startswith('Answer'))}")

# The key question: were any baseline answers miscounted?
# Compare original parser vs raw first letter after 'Answer: '
mismatches = 0
for r in base:
    raw = r["raw_response"].strip()
    parsed = r["parsed_answer"]
    if "Answer:" in raw:
        after = raw.split("Answer:")[1].strip()
        letter = after[0].upper() if after else ""
        if letter in "ABCD" and letter != parsed:
            mismatches += 1
            print(f"  MISMATCH: parsed={parsed} extracted={letter} raw={raw[:80]}")

print(f"\nBaseline parser mismatches: {mismatches}")
print(f"\nConclusion: baseline parser was {'consistent' if mismatches == 0 else 'INCONSISTENT'}")
