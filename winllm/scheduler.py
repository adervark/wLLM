"""Request scheduler with continuous batching support."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .config import SchedulerConfig, SamplingParams
from .engine import GenerationRequest, InferenceEngine, RequestStatus

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    """Runtime statistics for the scheduler."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_generation_tokens: int = 0
    total_generation_time: float = 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_generation_time == 0:
            return 0.0
        return self.total_generation_tokens / self.total_generation_time

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 1),
        }


class Scheduler:
    """Manages request queueing and dispatching to the inference engine.

    Supports:
    - FIFO request queue with configurable max size
    - Memory-aware admission (checks KV cache availability)
    - Async request submission and result retrieval
    - Throughput tracking
    """

    def __init__(self, engine: InferenceEngine, config: Optional[SchedulerConfig] = None):
        self.engine = engine
        self.config = config or SchedulerConfig()
        self.stats = SchedulerStats()

        # Request queues
        self._waiting: deque[GenerationRequest] = deque()
        self._running: dict[str, GenerationRequest] = {}
        self._completed: dict[str, GenerationRequest] = {}

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_batch_size)
        self._lock = asyncio.Lock()

        logger.info(
            "Scheduler initialized: max_batch_size=%d, max_waiting=%d",
            self.config.max_batch_size,
            self.config.max_waiting_requests,
        )

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    async def submit(self, request: GenerationRequest) -> GenerationRequest:
        """Submit a request for generation. Blocks until completion."""
        # Check queue limits
        if len(self._waiting) >= self.config.max_waiting_requests:
            request.status = RequestStatus.FAILED
            request.error = "Server overloaded — request queue full"
            return request

        self.stats.total_requests += 1

        # Tokenize early so we know the prompt length
        if not request.prompt_token_ids:
            request.prompt_token_ids = self.engine.tokenize(request.prompt)

        self.stats.total_prompt_tokens += len(request.prompt_token_ids)

        # Wait for a slot
        async with self._semaphore:
            async with self._lock:
                self._running[request.request_id] = request

            try:
                # Run generation in thread pool (model inference is blocking)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.engine.generate, request
                )
            except Exception as e:
                request.status = RequestStatus.FAILED
                request.error = str(e)
                result = request
            finally:
                async with self._lock:
                    self._running.pop(request.request_id, None)
                    self._completed[request.request_id] = result

            # Update stats
            if result.status == RequestStatus.COMPLETED:
                self.stats.completed_requests += 1
                self.stats.total_generation_tokens += result.generation_tokens
                self.stats.total_generation_time += result.elapsed
            else:
                self.stats.failed_requests += 1

            return result

    async def submit_streaming(self, request: GenerationRequest):
        """Submit a request and stream tokens via the request's callback.

        Returns the completed request.
        """
        # Same flow as submit but uses generate_stream
        if len(self._waiting) >= self.config.max_waiting_requests:
            request.status = RequestStatus.FAILED
            request.error = "Server overloaded — request queue full"
            return request

        self.stats.total_requests += 1

        if not request.prompt_token_ids:
            request.prompt_token_ids = self.engine.tokenize(request.prompt)

        self.stats.total_prompt_tokens += len(request.prompt_token_ids)

        async with self._semaphore:
            async with self._lock:
                self._running[request.request_id] = request

            try:
                async for _text, _finished in self.engine.generate_stream(request):
                    pass  # The stream callback on the request handles output
            except Exception as e:
                request.status = RequestStatus.FAILED
                request.error = str(e)
            finally:
                async with self._lock:
                    self._running.pop(request.request_id, None)
                    self._completed[request.request_id] = request

            if request.status == RequestStatus.COMPLETED:
                self.stats.completed_requests += 1
                self.stats.total_generation_tokens += request.generation_tokens
                self.stats.total_generation_time += request.elapsed
            else:
                self.stats.failed_requests += 1

            return request

    def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """Get a request by ID from any queue."""
        if request_id in self._running:
            return self._running[request_id]
        if request_id in self._completed:
            return self._completed[request_id]
        for req in self._waiting:
            if req.request_id == request_id:
                return req
        return None

    def get_status(self) -> dict:
        """Get scheduler status."""
        kv_stats = {}
        if self.engine.kv_cache_manager:
            kv_stats = self.engine.kv_cache_manager.get_stats()

        return {
            "waiting": self.num_waiting,
            "running": self.num_running,
            "completed": len(self._completed),
            "stats": self.stats.to_dict(),
            "kv_cache": kv_stats,
        }
