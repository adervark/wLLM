"""Sampling parameter configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingParams:
    """Parameters controlling token sampling."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None
    # OpenAI-style structured output: {"type": "json_object"} or
    # {"type": "json_schema", "json_schema": {"schema": {...}}}.
    # Requires the optional xgrammar dependency (winllm[structured]).
    response_format: Optional[dict] = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
