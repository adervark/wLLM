"""Composable logits processors implementing ``winllm.core.LogitsProcessor``.

The sampler runs DEFAULT_PIPELINE in order. Adding a new sampling
technique (min-p, typical sampling, ...) means writing one new class
and inserting it into a pipeline — no existing code changes (OCP).
"""

from __future__ import annotations

import torch

from . import ops


class RepetitionPenaltyProcessor:
    """Penalizes tokens that have already been generated."""

    def __call__(self, logits: torch.Tensor, *, params: list, generated_ids: list) -> torch.Tensor:
        penalties = [p.repetition_penalty for p in params]
        if all(p == 1.0 for p in penalties):
            return logits
        return ops.apply_repetition_penalty(logits, generated_ids, penalties)


class TemperatureProcessor:
    """Scales logits by per-request temperature."""

    def __call__(self, logits: torch.Tensor, *, params: list, generated_ids: list) -> torch.Tensor:
        return ops.apply_temperature(logits, [p.temperature for p in params])


class TopKProcessor:
    """Keeps only the top-k logits per request."""

    def __call__(self, logits: torch.Tensor, *, params: list, generated_ids: list) -> torch.Tensor:
        return ops.apply_top_k(logits, [p.top_k for p in params])


class TopPProcessor:
    """Nucleus sampling: keeps the smallest set of tokens with cumulative probability >= top_p."""

    def __call__(self, logits: torch.Tensor, *, params: list, generated_ids: list) -> torch.Tensor:
        return ops.apply_top_p(logits, [p.top_p for p in params])


#: Order matters: penalty before temperature before truncation.
DEFAULT_PIPELINE = (
    RepetitionPenaltyProcessor(),
    TemperatureProcessor(),
    TopKProcessor(),
    TopPProcessor(),
)
