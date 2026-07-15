"""Batched token sampling entry point."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import SamplingParams
from .native import fused_sampler
from .ops import _uniform
from .processors import DEFAULT_PIPELINE


def _per_row_ids(
    generated_ids: list[int] | list[list[int]] | None, batch_size: int
) -> list[list[int]]:
    """Normalize generated_ids into one id-list per batch row."""
    if not generated_ids:
        return [[] for _ in range(batch_size)]
    if isinstance(generated_ids[0], list):
        return generated_ids
    return [generated_ids] * batch_size


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams | list[SamplingParams],
    generated_ids: list[int] | list[list[int]] | None = None,
    generator: torch.Generator | None = None,
    grammars: list | None = None,
) -> torch.Tensor:
    """Full batched sampling pipeline: grammar mask -> penalties -> temperature -> top-k -> top-p -> sample."""
    batch_size = logits.shape[0]
    params_list = [params] * batch_size if isinstance(params, SamplingParams) else params

    # Grammar-constrained rows are masked before everything else (including
    # the greedy fast paths) so structured output holds for all sampling modes.
    owned = False  # True once logits is our private copy, safe to mutate
    if grammars is not None and any(g is not None for g in grammars):
        logits = logits.clone()
        owned = True
        per_row_ids = _per_row_ids(generated_ids, batch_size)
        for i, grammar in enumerate(grammars):
            if grammar is not None:
                grammar.apply(logits[i], per_row_ids[i])

    penalties = [p.repetition_penalty for p in params_list]
    temperatures = [p.temperature for p in params_list]

    # Fast path: pure greedy with no repetition penalty -- skip everything
    all_greedy = all(t == 0 for t in temperatures)
    no_penalties = all(p == 1.0 for p in penalties)
    if all_greedy and no_penalties:
        return torch.argmax(logits, dim=-1)

    # Fused native kernel: penalty/temperature/top-k/top-p/draw in ONE
    # launch instead of ~15 (native.py has the why). Requirements: uniform
    # parameters across the batch, and no request generator — the kernel
    # draws from its own RNG stream, which would break seeded
    # reproducibility. Greedy rows never touch the RNG, so seeded greedy
    # still fuses. Grammar-masked rows are fine: the mask is already in the
    # (cloned) logits and the kernel never mutates its input.
    if generator is None or all_greedy:
        temp = _uniform(temperatures)
        pen = _uniform(penalties)
        top_k = _uniform([p.top_k for p in params_list])
        top_p = _uniform([p.top_p for p in params_list])
        if None not in (temp, pen, top_k, top_p):
            if generated_ids and isinstance(generated_ids[0], list):
                # Per-row id lists only fuse when there is one row (the
                # sequential verify path); the kernel takes one id set.
                flat_ids = generated_ids[0] if batch_size == 1 else None
                fusable = pen == 1.0 or batch_size == 1
            else:
                flat_ids = generated_ids
                fusable = True
            if fusable:
                sampled = fused_sampler.sample(logits, temp, top_k, top_p, pen, flat_ids)
                if sampled is not None:
                    return sampled

    # Fast path: greedy with only repetition penalty -- minimal work
    if all_greedy and not no_penalties:
        if not owned:
            logits = logits.clone()
        logits = DEFAULT_PIPELINE[0](logits, params=params_list, generated_ids=generated_ids)
        return torch.argmax(logits, dim=-1)

    # The pipeline mutates in place; clone unless the grammar path already did
    if not owned:
        logits = logits.clone()

    for processor in DEFAULT_PIPELINE:
        logits = processor(logits, params=params_list, generated_ids=generated_ids)

    # Multinomial sampling for non-greedy rows
    probs = F.softmax(logits, dim=-1)
    # Multinomial needs valid probabilities: an all-masked row softmaxes to
    # NaN. Scrub unconditionally — checking isnan().any() first would be a
    # GPU->CPU sync on every step, costlier than the kernel it tries to skip.
    probs = torch.nan_to_num(probs, nan=0.0)

    try:
        sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    except RuntimeError:
        # Fallback if multinomial fails due to 0-probability across all tokens
        sampled = torch.argmax(logits, dim=-1)

    # Handle greedy overrides where temp == 0 in a mixed batch. Decided in
    # Python so the pure-sampling common case builds no mask tensor.
    if any(t == 0 for t in temperatures):
        greedy_mask = torch.tensor([t == 0 for t in temperatures], dtype=torch.bool).to(
            logits.device, non_blocking=True
        )
        greedy_samples = torch.argmax(logits, dim=-1)
        sampled = torch.where(greedy_mask, greedy_samples, sampled)

    return sampled
