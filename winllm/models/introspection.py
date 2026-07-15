"""Model architecture introspection."""

from __future__ import annotations

from typing import Any


def extract_kv_params(model: Any) -> dict:
    """Extract KV cache dimensions from a loaded model's config.

    Different model architectures use different config attribute names
    for the same concept, so we check multiple alternatives.
    """
    config = model.config
    result = {}

    # Number of decoder layers
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        val = getattr(config, attr, None)
        if val is not None:
            result["num_layers"] = val
            break

    # Number of KV heads (may differ from attention heads in GQA models)
    for attr in ("num_key_value_heads", "num_kv_heads"):
        val = getattr(config, attr, None)
        if val is not None:
            result["num_kv_heads"] = val
            break
    else:
        # Fallback: use full attention heads (non-GQA models)
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            val = getattr(config, attr, None)
            if val is not None:
                result["num_kv_heads"] = val
                break

    # Head dimension (compute from hidden_size if not explicit)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_heads = getattr(config, "num_attention_heads", None)
        if hidden_size and num_heads:
            head_dim = hidden_size // num_heads
    if head_dim:
        result["head_dim"] = head_dim

    return result
