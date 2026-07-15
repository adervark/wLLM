"""Scheduler throughput statistics."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.types import GenerationRequest


class LatencyWindow:
    """Bounded sliding window of samples with percentile queries.

    Keeps the most recent ``maxlen`` observations so percentiles reflect
    current behavior rather than the whole process lifetime.
    """

    def __init__(self, maxlen: int = 1000):
        self._values: deque[float] = deque(maxlen=maxlen)

    def record(self, value: float) -> None:
        self._values.append(value)

    def __len__(self) -> int:
        return len(self._values)

    def percentile(self, p: float) -> float:
        """Nearest-rank percentile; 0.0 when no samples recorded."""
        if not self._values:
            return 0.0
        ordered = sorted(self._values)
        rank = max(1, math.ceil(p * len(ordered)))
        return ordered[rank - 1]

    def percentiles(self, ps: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict[str, float]:
        return {f"p{int(p * 100)}": round(self.percentile(p), 4) for p in ps}


@dataclass
class SchedulerStats:
    """Runtime statistics for the scheduler."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_generation_tokens: int = 0
    total_generation_time: float = 0.0

    # Per-request latency distributions (recent window)
    ttft: LatencyWindow = field(default_factory=LatencyWindow)
    e2e_latency: LatencyWindow = field(default_factory=LatencyWindow)
    request_tokens_per_second: LatencyWindow = field(default_factory=LatencyWindow)

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_generation_time == 0:
            return 0.0
        return self.total_generation_tokens / self.total_generation_time

    def record_request(self, request: "GenerationRequest") -> None:
        """Record one completed request's latency profile."""
        if request.first_token_at is not None:
            # Queue wait included: this is the TTFT the client experienced.
            self.ttft.record(request.first_token_at - request.created_at)
        if request.finished_at is not None:
            self.e2e_latency.record(request.finished_at - request.created_at)
        if request.tokens_per_second > 0:
            self.request_tokens_per_second.record(request.tokens_per_second)

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 1),
            "ttft_seconds": self.ttft.percentiles(),
            "e2e_latency_seconds": self.e2e_latency.percentiles(),
            "tokens_per_second": self.request_tokens_per_second.percentiles(),
        }
