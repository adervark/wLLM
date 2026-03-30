"""Common data types and enums for WinLLM."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .config import SamplingParams


class RequestStatus(str, Enum):
    """Status of an inference request."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequest:
    """A single generation request."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    status: RequestStatus = RequestStatus.PENDING
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    # Streaming
    _stream_callback: Optional[Callable[[str, bool], None]] = field(
        default=None, repr=False
    )
    _token_callback: Optional[Callable[[int, bool], None]] = field(
        default=None, repr=False
    )

    # Cancellation (thread-safe)
    _cancelled: threading.Event = field(
        default_factory=threading.Event, repr=False
    )

    # Internal state for batching
    _past_key_values: Optional[tuple] = field(default=None, repr=False)
    _prefix_cache_token_len: int = field(default=0, repr=False)
    _stream_text_cursor: int = field(default=0, repr=False)
    _prefix_past_key_values: Optional[tuple] = field(default=None, repr=False)
    _prefill_cursor: int = field(default=0, repr=False)
    _draft_past_key_values: Optional[tuple] = field(default=None, repr=False)

    # Polling optimization
    _completed_event: Optional[asyncio.Event] = field(default=None, repr=False)
    _loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)

    @property
    def is_prefill_complete(self) -> bool:
        """Returns True if the entire prompt has been processed by prefill."""
        return self._prefill_cursor >= len(self.prompt_token_ids)

    def cancel(self):
        """Signal this request to stop generating."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def generation_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed == 0:
            return 0.0
        return self.generation_tokens / self.elapsed
