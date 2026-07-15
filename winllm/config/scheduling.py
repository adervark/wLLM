"""Request scheduler configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    """Configuration for the request scheduler."""
    max_batch_size: int = 4
    max_waiting_requests: int = 64
    max_num_seqs: int = 8
    scheduling_policy: str = "fcfs"
    max_completed_requests: int = 1000      # Max completed requests to keep in memory
    completed_request_ttl: float = 300.0    # Seconds before completed requests are evicted
    chunked_prefill_enabled: bool = True
    max_num_batched_tokens: int = 512       # Maximum tokens to process per forward pass
