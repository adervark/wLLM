"""Token sampling strategies for text generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import SamplingParams


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits for already-generated tokens."""
    if penalty == 1.0 or not generated_ids:
        return logits

    # Get unique generated token ids
    token_ids = list(set(generated_ids))
    scores = logits[:, token_ids]

    # Apply penalty: reduce probability of repeated tokens
    # If score > 0, divide by penalty; if score < 0, multiply by penalty
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[:, token_ids] = scores

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
    params: SamplingParams,
    generated_ids: list[int] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full sampling pipeline: penalties → temperature → top-k → top-p → sample.

    Args:
        logits: Raw logits from the model, shape (batch_size, vocab_size).
        params: Sampling parameters.
        generated_ids: Previously generated token IDs for repetition penalty.
        generator: Optional random generator for reproducibility.

    Returns:
        Sampled token IDs, shape (batch_size,).
    """
    logits = logits.clone()

    # 1. Repetition penalty
    if generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, params.repetition_penalty)

    # 2. Greedy decoding (temperature == 0)
    if params.temperature == 0:
        return torch.argmax(logits, dim=-1)

    # 3. Temperature
    logits = apply_temperature(logits, params.temperature)

    # 4. Top-k
    logits = apply_top_k(logits, params.top_k)

    # 5. Top-p
    logits = apply_top_p(logits, params.top_p)

    # 6. Sample from distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
