"""Token sampling strategies for text generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import SamplingParams


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int] | list[list[int]],
    penalty: float | list[float],
) -> torch.Tensor:
    """Apply repetition penalty to logits for already-generated tokens."""
    batch_size = logits.shape[0]
    
    # Handle both single and batched penalties
    penalties = [penalty] * batch_size if isinstance(penalty, float) else penalty
    
    # Handle both single and batched generated_ids
    if not generated_ids or (isinstance(generated_ids[0], list) and not any(generated_ids)):
        return logits

    for i in range(batch_size):
        row_penalty = penalties[i]
        row_ids = generated_ids[i] if isinstance(generated_ids[0], list) else generated_ids
        
        if row_penalty == 1.0 or not row_ids:
            continue

        # Get unique generated token ids for this row
        token_ids = list(set(row_ids))
        scores = logits[i, token_ids]

        # Apply penalty: reduce probability of repeated tokens
        # If score > 0, divide by penalty; if score < 0, multiply by penalty
        logits[i, token_ids] = torch.where(scores > 0, scores / row_penalty, scores * row_penalty)

    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature."""
    if temperature == 0:
        return logits  # Greedy — handled at sampling stage
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only top-k logits, set rest to -inf."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    # Find the k-th largest value
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float("-inf")
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep tokens with cumulative probability <= top_p (nucleus sampling)."""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens with cumulative probability exceeding top_p
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order
    logits = sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits


def sample_token(
    logits: torch.Tensor, 
    params: SamplingParams | list[SamplingParams],
    generated_ids: list[int] | list[list[int]] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full sampling pipeline: penalties → temperature → top-k → top-p → sample.

    Args:
        logits: Raw logits from the model, shape (batch_size, vocab_size).
        params: Sampling parameters (single or list for batch).
        generated_ids: Previously generated token IDs (single list or list of lists).
        generator: Optional random generator for reproducibility.

    Returns:
        Sampled token IDs, shape (batch_size,).
    """
    logits = logits.clone()
    batch_size = logits.shape[0]
    
    # Normalize inputs to lists for consistent processing
    params_list = [params] * batch_size if isinstance(params, SamplingParams) else params
    
    # 1. Repetition penalty (handles batches internally)
    penalties = [p.repetition_penalty for p in params_list]
    logits = apply_repetition_penalty(logits, generated_ids, penalties)

    # 2. Per-row sampling logic for temperature, top-k, top-p
    # For performance, we could group similar params, but for Python overhead 
    # a row-wise loop is clearer and safe for common batch sizes.
    result_tokens = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    
    for i in range(batch_size):
        row_params = params_list[i]
        row_logits = logits[i:i+1] # Keep dim for functions
        
        # Greedy decoding (temperature == 0)
        if row_params.temperature == 0:
            result_tokens[i] = torch.argmax(row_logits, dim=-1)
            continue

        # Temperature
        row_logits = apply_temperature(row_logits, row_params.temperature)

        # Top-k
        row_logits = apply_top_k(row_logits, row_params.top_k)

        # Top-p
        row_logits = apply_top_p(row_logits, row_params.top_p)

        # Sample from distribution
        probs = F.softmax(row_logits, dim=-1)
        result_tokens[i] = torch.multinomial(probs, num_samples=1, generator=generator).squeeze()

    return result_tokens
