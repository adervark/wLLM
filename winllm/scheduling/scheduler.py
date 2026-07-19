"""The continuous-batching scheduler.

The Scheduler's single job is running the background inference loop:
admit → step → finish-check → clean up. The decisions themselves are
delegated to collaborators:

  - SchedulingPolicy (which request next)
  - AdmissionController (does it fit in the KV cache)
  - CompletedRequestStore (bounded result retention)
  - SchedulerStats (throughput accounting)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Optional

from ..config import SchedulerConfig
from ..core.types import GenerationRequest, RequestStatus, normalize_eos_ids
from ..inference.engine import InferenceEngine
from ..kvcache.prefix import compute_prefix_hashes, slice_prompt_blocks
from .admission import AdmissionController, AdmissionResult
from .completed import CompletedRequestStore
from .policies import get_policy
from .stats import SchedulerStats

logger = logging.getLogger(__name__)


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

        # Collaborators
        self._policy = get_policy(self.config.scheduling_policy)
        # Built lazily on first admission: the KV cache manager only exists
        # after the engine has loaded its model (post-construction).
        self._admission: Optional[AdmissionController] = None
        self._completed = CompletedRequestStore(
            max_kept=self.config.max_completed_requests,
            ttl=self.config.completed_request_ttl,
        )

        # Request queues
        self._waiting: deque[GenerationRequest] = deque()

        # Concurrency control
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
        return len(self._active_reqs)

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _start_loop(self):
        """Start the background inference loop thread."""
        self._loop_thread = threading.Thread(target=self._run_inference_loop, daemon=True)
        self._loop_thread.start()

    def _run_inference_loop(self):
        """Background inference loop — runs on a dedicated daemon thread.

        Continuously:
          1. Admits waiting requests into the active batch.
          2. Runs one forward-pass step via the engine.
          3. Checks for finished / cancelled requests and cleans up.
        """
        # Cache EOS token ids once to avoid repeated lookups in the hot loop.
        # Normalized to a set so multi-EOS tokenizers (e.g. Llama-3) stop correctly.
        eos_token_ids = normalize_eos_ids(self.engine.tokenizer.eos_token_id)

        while not self._stop_event.is_set():
            # Wait for requests if none are running
            if not self._active_reqs:
                self._new_request_event.wait(timeout=0.1)
                self._new_request_event.clear()

            # 1. Admit new requests from waiting queue
            if self._waiting and len(self._active_reqs) < self.config.max_batch_size:
                self._admit_requests()

            if not self._active_reqs:
                continue

            # 2. Run one inference step
            try:
                self.engine.generate_step(
                    self._active_reqs,
                    chunked_prefill_enabled=self.config.chunked_prefill_enabled,
                    max_num_batched_tokens=self.config.max_num_batched_tokens,
                )
            except Exception as e:
                logger.exception("Error in inference step")
                # Fail all active requests in the batch if a global error occurred.
                # _collect_finished picks them up (status FAILED -> "error"),
                # signals waiters, and cleanup below removes them from the
                # active batch and frees their KV blocks.
                for req in self._active_reqs:
                    req.status = RequestStatus.FAILED
                    req.error = str(e)

            # Stamp time-to-first-token for requests that just produced one
            for req in self._active_reqs:
                if req.first_token_at is None and req.output_token_ids:
                    req.first_token_at = time.time()

            # 3. Check for finished/cancelled requests
            finished = self._collect_finished(eos_token_ids)

            # 4. Cleanup finished/failed requests
            if finished:
                finished_set = set(id(r) for r in finished)
                self._active_reqs = [r for r in self._active_reqs if id(r) not in finished_set]
                for req in finished:
                    self.engine.kv_cache_manager.free_sequence(req.request_id)
                    self._handle_completed(req)
                    # Only promote if it was successful and didn't hit max tokens abruptly
                    if req.status == RequestStatus.COMPLETED:
                        self._try_promote_prefix_cache(req)
                    # Release KV references so completed requests don't keep the
                    # (possibly batched) cache tensors alive in the result store.
                    req._past_key_values = None
                    req._prefix_past_key_values = None
                    req._draft_past_key_values = None

            # 5. Evict old completed requests when the store grows too large
            if self._completed.needs_eviction:
                self._completed.evict()

    def _collect_finished(self, eos_token_ids: set[int]) -> list[GenerationRequest]:
        """Find active requests that finished: EOS, stop string, max tokens,
        cancellation, or failure."""
        finished = []
        now = time.time()
        for req in self._active_reqs:
            reason = self._check_finish(req, eos_token_ids)
            if reason is None:
                continue

            if reason == "error":
                req.status = RequestStatus.FAILED
            elif reason == "cancelled":
                req.status = RequestStatus.CANCELLED
            else:
                req.status = RequestStatus.COMPLETED

            req.finish_reason = reason
            req.finished_at = now

            # Signal stream end on whichever callback flavor is registered
            if req._token_callback:
                last_id = req.output_token_ids[-1] if req.output_token_ids else 0
                req._token_callback(last_id, True)
            if req._stream_callback:
                req._stream_callback("", True)

            self._signal_completion(req)

            finished.append(req)
        return finished

    def _check_finish(self, req: GenerationRequest, eos_token_ids: set[int]) -> Optional[str]:
        """Return the finish reason for a request, or None if still generating.

        Checked in priority order: failure and cancellation apply even before
        the first token; EOS / stop strings / max_tokens need output tokens.
        """
        if req.status == RequestStatus.FAILED:
            return "error"
        if req.is_cancelled:
            return "cancelled"
        if not req.output_token_ids:
            return None

        if req.output_token_ids[-1] in eos_token_ids:
            return "stop"

        if self._hit_stop_string(req):
            return "stop"

        if len(req.output_token_ids) >= req.sampling_params.max_tokens:
            return "length"
        return None

    def _hit_stop_string(self, req: GenerationRequest) -> bool:
        """Incrementally decode new output tokens and check for stop strings.

        Maintains ``req._running_text`` with a token cursor so each output
        token is decoded exactly once across the request's lifetime. On a hit,
        ``output_text`` is set to the text *before* the stop string, matching
        the blocking generator's trimming behavior.
        """
        stop_strings = req.sampling_params.stop
        if not stop_strings:
            return False

        tokenizer = self.engine.tokenizer
        while req._stop_check_cursor < len(req.output_token_ids):
            req._running_text += tokenizer.decode(
                [req.output_token_ids[req._stop_check_cursor]], skip_special_tokens=True
            )
            req._stop_check_cursor += 1

        # Earliest occurrence wins, not list order, so the trimmed output
        # matches what the streaming StopStringGate emits.
        hits = [i for i in (req._running_text.find(s) for s in stop_strings) if i != -1]
        if hits:
            req.output_text = req._running_text[: min(hits)]
            req._stop_trimmed = True
            return True
        return False

    def _admit_requests(self):
        """Move requests from the waiting queue into the active batch."""
        if self._admission is None:
            self._admission = AdmissionController(self.engine.kv_cache_manager)
        admission = self._admission

        while len(self._active_reqs) < self.config.max_batch_size and self._waiting:
            req = self._policy.select(self._waiting)
            if req is None:
                break

            # A request cancelled while queued (e.g. client disconnect) must
            # not cost a KV allocation and a prefill pass.
            if req.is_cancelled:
                req.status = RequestStatus.CANCELLED
                req.finish_reason = "cancelled"
                req.finished_at = time.time()
                if req._token_callback:
                    req._token_callback(0, True)
                if req._stream_callback:
                    req._stream_callback("", True)
                self._handle_completed(req)
                self._signal_completion(req)
                continue

            result = admission.try_admit(req)
            if result is AdmissionResult.ADMITTED:
                self._active_reqs.append(req)
            else:
                if not self._active_reqs:
                    # Nothing running and it still doesn't fit: it never will
                    req.status = RequestStatus.FAILED
                    req.error = "Request too large for KV cache"
                    self._handle_completed(req)
                    self._signal_completion(req)
                else:
                    # Wait for running requests to free memory
                    self._policy.requeue(self._waiting, req)
                    break

    @staticmethod
    def _signal_completion(req: GenerationRequest) -> None:
        """Wake the submitting coroutine (thread-safe)."""
        if req._completed_event and req._loop:
            req._loop.call_soon_threadsafe(req._completed_event.set)

    def _try_promote_prefix_cache(self, req: GenerationRequest):
        """Save this request's complete prompt blocks in the prefix cache.

        Every complete block of the prompt is promoted (not just the first),
        keyed by its cumulative hash, so future requests sharing an arbitrarily
        long prefix can skip re-computing all of those tokens. Each block's KV
        is stored once (O(n) memory); ``match`` concatenates the matched chain.

        Tensor carving is delegated to ``slice_prompt_blocks`` so the scheduler
        stays free of KV cache layout knowledge.
        """
        if req.status != RequestStatus.COMPLETED:
            return

        block_size = self.engine.kv_cache_manager.block_size
        prompt_len = len(req.prompt_token_ids)
        num_blocks = prompt_len // block_size
        if num_blocks == 0:
            return

        # Use the same hashing that admission uses for lookups,
        # so promoted entries are actually found during prefix matching.
        prefix_hashes = compute_prefix_hashes(req.prompt_token_ids, block_size)
        if len(prefix_hashes) < num_blocks:
            return

        per_block_kv = slice_prompt_blocks(req._past_key_values, block_size, num_blocks)
        self.engine.kv_cache_manager.promote_prefix_chain(
            prefix_hashes[:num_blocks], req.request_id, per_block_kv
        )

    def _handle_completed(self, request: GenerationRequest):
        """Update stats and move request to the completed store."""
        self._completed.add(request)

        if request.status == RequestStatus.COMPLETED:
            self.stats.completed_requests += 1
            self.stats.total_generation_tokens += request.generation_tokens
            self.stats.total_generation_time += request.elapsed
            self.stats.record_request(request)
        elif request.status == RequestStatus.FAILED:
            self.stats.failed_requests += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, request: GenerationRequest) -> GenerationRequest:
        """Submit a request for generation. Blocks until completion."""
        if len(self._waiting) >= self.config.max_waiting_requests:
            request.status = RequestStatus.FAILED
            request.error = "Server overloaded — request queue full"
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            return request

        self.stats.total_requests += 1

        if not request.prompt_token_ids:
            request.prompt_token_ids = self.engine.tokenize(request.prompt)

        self.stats.total_prompt_tokens += len(request.prompt_token_ids)

        request._loop = asyncio.get_running_loop()
        request._completed_event = asyncio.Event()

        # Add to waiting queue
        self._waiting.append(request)
        self._new_request_event.set()

        # Wait for completion
        await request._completed_event.wait()

        # Offload expensive final text decoding from GPU loop
        if (
            request.status == RequestStatus.COMPLETED
            and not request._stream_callback
            and not request.output_text
            and not request._stop_trimmed
        ):
            loop = asyncio.get_running_loop()
            request.output_text = await loop.run_in_executor(
                None, self.engine.decode_tokens, request.output_token_ids
            )

        return request

    async def submit_streaming(self, request: GenerationRequest):
        """Submit a request that streams tokens via the request's callback.

        This is a thin wrapper around submit() for semantic clarity at call sites.
        Streaming behavior is determined by the request's _token_callback or
        _stream_callback being set before submission. The scheduler's inference
        loop invokes these callbacks as tokens are generated, regardless of which
        submission method is used.
        """
        return await self.submit(request)

    def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """Get a request by ID from any queue."""
        completed = self._completed.get(request_id)
        if completed is not None:
            return completed
        for req in self._active_reqs:
            if req.request_id == request_id:
                return req
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
