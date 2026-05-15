"""Model-family configuration for cross-family replication.

Supports: Gemma 3 (12B, 27B), Llama 3.1 (8B, 70B), Qwen 2.5 (7B),
          Mistral (7B v0.3)

Single source of truth for MODEL_LAYERS — import from here instead of
duplicating in step4 / step_probe_target / step_steering.
"""

MODEL_LAYERS = {
    "gemma-3-12b-it": 48,
    "gemma-3-27b-it": 62,
    "Meta-Llama-3.1-8B-Instruct-bf16": 32,
    "Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED": 80,
    "Qwen2.5-7B-Instruct-bf16": 28,
    "Mistral-7B-Instruct-v0.3": 32,
}


def get_model_config(model, tokenizer, model_name: str = "") -> dict:
    """Return model-specific config for hidden state extraction.

    Detects model family from attribute structure and returns a unified
    interface for hidden-state extraction, embedding scaling, and BOS handling.

    Families:
        Gemma 3: model.language_model.model.layers, embedding scaling
        Llama / Mistral / Qwen: model.model.layers, no embedding scaling
    """

    # Gemma wraps in language_model; Llama/Qwen/Mistral don't
    if hasattr(model, 'language_model'):
        lm = model.language_model
        layers = lm.model.layers
        is_gemma = True
    else:
        lm = model
        layers = model.model.layers
        is_gemma = False

    n_layers = len(layers)
    hidden_dim = layers[0].hidden_size
    bos_id = tokenizer.bos_token_id  # 2=Gemma, 128000=Llama, 1=Mistral, None=Qwen

    return {
        "lm": lm,
        "layers": layers,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "bos_id": bos_id,
        "scale_embeddings": is_gemma,
        "layer_indices": {
            "first": 0,
            "middle": n_layers // 2,
            "last": n_layers - 1,
        },
    }
