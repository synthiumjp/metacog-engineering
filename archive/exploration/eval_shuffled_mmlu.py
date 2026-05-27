import json
import sys
import numpy as np
sys.path.insert(0, r"D:\metacog\scripts")
from utils_phase0 import auroc2
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load shuffled model
model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
model = PeftModel.from_pretrained(base, r"D:\metacog\results\finetune\lora_shuffled")
model.eval()
print(f"Shuffled model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Load baseline MMLU for gold answers
with open(r"D:\metacog\results\baseline\meval_responses.json") as f:
    base_meval = json.load(f)

# Load MMLU items
from step1_baseline import load_mmlu_stratified
meval_items = load_mmlu_stratified(500, 42)

MMLU_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
)

def reparse_letter(response):
    resp = response.strip()
    if not resp:
        return ""
    first = resp[0].upper()
    if first in "ABCD":
        return first
    return ""

# Evaluate
results = []
for idx, item in enumerate(meval_items):
    prompt_text = MMLU_PROMPT.format(
        question=item["question"],
        A=item["choices"][0], B=item["choices"][1],
        C=item["choices"][2], D=item["choices"][3],
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Parse confidence
    conf = float("nan")
    import re
    conf_match = re.search(r"(\d+)%", response)
    if conf_match:
        conf = float(conf_match.group(1))

    # Parse answer
    letter = reparse_letter(response)
    gold = item["answer_letter"]
    correct = (letter == gold.upper()) if letter else False

    results.append({
        "response": response,
        "letter": letter,
        "gold": gold,
        "correct": correct,
        "confidence": conf,
    })

    if (idx + 1) % 50 == 0:
        elapsed_correct = sum(r["correct"] for r in results)
        print(f"  [{idx+1}/{len(meval_items)}] acc={elapsed_correct/(idx+1):.3f}")

# Compute metrics
parsed = [r for r in results if r["letter"]]
correct_count = sum(r["correct"] for r in parsed)
conf_arr = np.array([r["confidence"] for r in parsed])
corr_arr = np.array([int(r["correct"]) for r in parsed])

valid_conf = ~np.isnan(conf_arr)
a2 = auroc2(conf_arr[valid_conf], corr_arr[valid_conf]) if valid_conf.sum() > 10 else float("nan")

print(f"\n=== SHUFFLED MODEL ON MMLU ===")
print(f"  Parsed: {len(parsed)}/{len(results)}")
print(f"  Accuracy: {correct_count}/{len(parsed)} = {correct_count/len(parsed):.3f}")
print(f"  AUROC2: {a2:.3f}")
print(f"  Mean conf (correct): {np.nanmean([r['confidence'] for r in parsed if r['correct']]):.1f}")
print(f"  Mean conf (incorrect): {np.nanmean([r['confidence'] for r in parsed if not r['correct']]):.1f}")

# Compare
print(f"\n=== COMPARISON ===")
print(f"  Baseline MMLU:  acc=0.542  AUROC2=0.535")
print(f"  Real-target:    acc=0.774  AUROC2=0.616")
print(f"  Shuffled:       acc={correct_count/len(parsed):.3f}  AUROC2={a2:.3f}")

# Sample responses
print(f"\n=== Sample shuffled MMLU responses ===")
for r in results[:5]:
    print(f"  letter={r['letter']} gold={r['gold']} correct={r['correct']} conf={r['confidence']}")
    print(f"  response: {r['response'][:100]}")
