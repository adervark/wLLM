"""Request scheduler with continuous batching support."""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .config import SchedulerConfig, SamplingParams
from .engine import InferenceEngine
from .types import GenerationRequest, RequestStatus

def _get_prefix_hashes(tokens: list[int], block_size: int) -> list[int]:
    """Generate hashes for every complete block prefix."""
    hashes = []
    num_blocks = len(tokens) // block_size
    for i in range(1, num_blocks + 1):
        # Hash the prefix up to the end of block i
        hashes.append(hash(tuple(tokens[:i * block_size])))
    return hashes

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
        self._completed: dict[str, tuple[float, GenerationRequest]] = {}  # req_id -> (finished_time, request)

        # Concurrency control
        self._lock = asyncio.Lock()
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._new_request_event = threading.Event()
        
        # Track active/pending requests
        self._active_reqs: list[GenerationRequest] = []
        
        self._start_loop()

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

    def _start_loop(self):
        """Start the background inference loop thread."""
        self._loop_thread = threading.Thread(target=self._run_inference_loop, daemon=True)
        self._loop_thread.start()

    def _run_inference_loop(self):
        """Main inference loop running in a background thread."""
        logger.info("Inference loop started")
        
        while not self._stop_event.is_set():
            # Wait for requests if none are running
            if not self._active_reqs:
                self._new_request_event.wait(timeout=0.1)
                self._new_request_event.clear()

            # 1. Admit new requests from waiting queue
            while len(self._active_reqs) < self.config.max_batch_size and self._waiting:
                req = self._waiting.popleft()
                
                # Check KV cache admission with prefix matching
                prompt_token_ids = req.prompt_token_ids
                block_size = self.engine.kv_cache_manager.block_size
                prefix_hashes = _get_prefix_hashes(prompt_token_ids, block_size)
                
                matched_blocks, matched_tensors = self.engine.kv_cache_manager.match_prefix(prefix_hashes)
                matched_len = len(matched_blocks) * block_size
                
                req._prefix_len = matched_len
                req._prefix_past_key_values = matched_tensors
                
                prompt_len = len(prompt_token_ids)
                max_new = req.sampling_params.max_tokens
                if self.engine.kv_cache_manager.can_allocate(prompt_len + max_new - matched_len):
                    self.engine.kv_cache_manager.allocate_sequence(
                        req.request_id, 
                        prompt_len, 
                        prefix_blocks=matched_blocks
                    )
                    self._active_reqs.append(req)
                    logger.debug(
                        "Admitted request %s to batch (prefix match: %d tokens)", 
                        req.request_id, matched_len
                    )
                    
                    # If it was a perfect match for a long prefix, we can promote it
                    # actually promotion happens after prefill.
                else:
                    # In a more complex scheduler, we might keep it in waiting,
                    # but here we fail it if it can't fit even when batch is empty.
                    if not self._active_reqs:
                        req.status = RequestStatus.FAILED
                        req.error = "Request too large for KV cache"
                        self._handle_completed(req)
                    else:
                        # Put back in waiting to try later
                        self._waiting.appendleft(req)
                        break

            if not self._active_reqs:
                continue

            # 2. Run one inference step
            try:
                self.engine.generate_step(
                    self._active_reqs, 
                    chunked_prefill_enabled=self.config.chunked_prefill_enabled,
                    max_num_batched_tokens=self.config.max_num_batched_tokens
                )
            except Exception as e:
                logger.exception("Error in inference step")
                # Fail all active requests in the batch if a global error occurred
                for req in self._active_reqs:
                    req.status = RequestStatus.FAILED
                    req.error = str(e)
                
            # 3. Check for finished/cancelled requests
            finished = []
            for req in self._active_reqs:
                is_eos = req.output_token_ids[-1] == self.engine.tokenizer.eos_token_id
                is_max = len(req.output_token_ids) >= req.sampling_params.max_tokens
                
                if is_eos or is_max or req.is_cancelled:
                    if not req.is_cancelled:
                        req.status = RequestStatus.COMPLETED
                    else:
                        req.status = RequestStatus.CANCELLED
                    
                    req.finished_at = time.time()
                    req.output_text = self.engine.decode_tokens(req.output_token_ids)
                    
                    # Signal stream end
                    if req._stream_callback:
                        req._stream_callback("", True)
                        
                    finished.append(req)

            # 4. Cleanup finished/failed requests
            for req in finished:
                self._try_promote_prefix_cache(req)
                self._active_reqs.remove(req)
                self.engine.kv_cache_manager.free_sequence(req.request_id)
                self._handle_completed(req)

    def _try_promote_prefix_cache(self, req: GenerationRequest):
        """Try to save this request's prompt KV tensors in the prefix cache.

        If the prompt is long enough (>= 1 block), we slice out the first
        block's KV tensors and store them so future requests with the same
        prefix can skip re-computing those tokens.

        KV cache tensor layout:
          past_key_values = ((key_layer0, val_layer0), (key_layer1, val_layer1), ...)
          Each tensor has shape: [batch, num_heads, seq_len, head_dim]
        """
        if req.status != RequestStatus.COMPLETED:
            return

        block_size = self.engine.kv_cache_manager.block_size
        prompt_len = len(req.prompt_token_ids)
        if prompt_len < block_size:
            return

        # Slice each layer's KV tensors to just the first block of the prompt
        first_block_kv = []
        for layer_kv in req._past_key_values:
            k, v = layer_kv
            first_block_kv.append((
                k[:, :, :block_size, :],
                v[:, :, :block_size, :],
            ))

        first_block_hash = hash(tuple(req.prompt_token_ids[:block_size]))
        self.engine.kv_cache_manager.promote_to_prefix(
            first_block_hash, req.request_id, block_size, tuple(first_block_kv)
        )

    def _handle_completed(self, request: GenerationRequest):
        """Update stats and move request to completed dict."""
        self._completed[request.request_id] = (time.time(), request)
        
        if request.status == RequestStatus.COMPLETED:
            self.stats.completed_requests += 1
            self.stats.total_generation_tokens += request.generation_tokens
            self.stats.total_generation_time += request.elapsed
        elif request.status == RequestStatus.FAILED:
            self.stats.failed_requests += 1

    async def submit(self, request: GenerationRequest) -> GenerationRequest:
        """Submit a request for generation. Blocks until completion."""
        if len(self._waiting) >= self.config.max_waiting_requests:
            request.status = RequestStatus.FAILED
            request.error = "Server overloaded — request queue full"
            return request

        self.stats.total_requests += 1

        if not request.prompt_token_ids:
            request.prompt_token_ids = self.engine.tokenize(request.prompt)

        self.stats.total_prompt_tokens += len(request.prompt_token_ids)

        # Add to waiting queue
        self._waiting.append(request)
        self._new_request_event.set()

        # Wait for completion
        while request.status in (RequestStatus.PENDING, RequestStatus.RUNNING):
            await asyncio.sleep(0.05)

        return request

    async def submit_streaming(self, request: GenerationRequest):
        """Submit a request and stream tokens via the request's callback."""
        # Same flow as submit but returns once the background loop starts processing it
        # Actually, for consistency, we wait until it's finished.
        return await self.submit(request)

    def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """Get a request by ID from any queue."""
        if request_id in self._running:
            return self._running[request_id]
        if request_id in self._completed:
            return self._completed[request_id][1]
        for req in self._waiting:
            if req.request_id == request_id:
                return req
        return None

    def _evict_completed(self):
        """Remove old entries from _completed to prevent memory leaks.

        Evicts by both TTL and max count. Should be called under self._lock.
        """
        ttl = self.config.completed_request_ttl
        max_kept = self.config.max_completed_requests
        now = time.time()

        # 1. TTL eviction
        expired = [
            rid for rid, (finished_at, _) in self._completed.items()
            if (now - finished_at) > ttl
        ]
        for rid in expired:
            del self._completed[rid]

        # 2. Count eviction (oldest first)
        if len(self._completed) > max_kept:
            sorted_ids = sorted(
                self._completed, key=lambda rid: self._completed[rid][0]
            )
            for rid in sorted_ids[:len(self._completed) - max_kept]:
                del self._completed[rid]

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
