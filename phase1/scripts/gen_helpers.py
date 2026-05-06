import mlx.core as mx
from mlx_lm import generate

MAX_TOKENS = 256

def generate_greedy(model, tokenizer, prompt):
    sampler = lambda logits: mx.argmax(logits, axis=-1)
    return generate(model, tokenizer, prompt=prompt,
                    max_tokens=MAX_TOKENS, sampler=sampler, verbose=False)

def generate_sampled(model, tokenizer, prompt, temp):
    sampler = lambda logits: mx.random.categorical(logits / temp)
    return generate(model, tokenizer, prompt=prompt,
                    max_tokens=MAX_TOKENS, sampler=sampler, verbose=False)
