"""The InferenceEngine facade.

The engine owns the lifecycle (load/unload) and *delegates* the actual
work to single-responsibility collaborators:

  - ModelRuntime:      shared loaded-model state
  - PrefillRunner:     prompt processing
  - DecodeRunner:      step-wise token generation (incl. batched decode)
  - BlockingGenerator: full single-request generation
  - StreamEmitter:     token streaming callbacks

Glossary for newcomers:
  - Prefill:  The first forward pass where the entire prompt is processed at once.
              This builds up the KV cache so future tokens can attend to the prompt.
  - Decode:   Each subsequent forward pass that generates one new token, using the
              cached key/value tensors from previous steps to avoid re-computation.
  - KV Cache: Stores the Key and Value tensors from previous steps so the model
              doesn't have to reprocess the full sequence every time.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Optional

import torch

from ..config import KVCacheConfig, ModelConfig
from ..core.types import GenerationRequest, RequestStatus
from ..hardware.cuda import configure_cuda_backends
from ..hardware.memory import get_gpu_memory_info
from ..kvcache import KVCacheManager
from ..models.loader import ModelLoader
from .buffers import DecodeInputBuffer, PersistentBatchCache
from .decode import DecodeRunner
from .generation import BlockingGenerator
from .prefill import PrefillRunner
from .runtime import ModelRuntime
from .streaming import StreamEmitter

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Core engine that runs model inference.

    Handles model loading, tokenization, and the generate loop
    with support for streaming token output.
    """

    def __init__(self, model_config: ModelConfig, kv_cache_config: Optional[KVCacheConfig] = None):
        self.model_config = model_config
        self.kv_cache_config = kv_cache_config or KVCacheConfig()

        self._loader = ModelLoader(model_config)
        self._runtime = ModelRuntime()
        self._ready = False
        self._grammar_backend = None  # built lazily on first structured request

        # Performance buffers shared by the decode paths
        self._input_buffer = DecodeInputBuffer()
        self._batch_cache = PersistentBatchCache()

        # Collaborators (all operate through the shared runtime)
        self._emitter = StreamEmitter(self._runtime)
        self._prefill_runner = PrefillRunner(self._runtime, self._emitter)
        self._decode_runner = DecodeRunner(
            self._runtime, self._emitter, self._input_buffer, self._batch_cache
        )
        self._blocking_generator = BlockingGenerator(
            self._runtime, model_config, self._emitter, self._input_buffer
        )

    # ------------------------------------------------------------------
    # Runtime state accessors
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._runtime.model

    @model.setter
    def model(self, value):
        self._runtime.model = value

    @property
    def tokenizer(self):
        return self._runtime.tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._runtime.tokenizer = value

    @property
    def kv_cache_manager(self) -> Optional[KVCacheManager]:
        return self._runtime.kv_cache_manager

    @kv_cache_manager.setter
    def kv_cache_manager(self, value):
        self._runtime.kv_cache_manager = value

    @property
    def speculative_engine(self):
        return self._runtime.speculative_engine

    @speculative_engine.setter
    def speculative_engine(self, value):
        self._runtime.speculative_engine = value

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self):
        """Load model and initialize KV cache manager."""
        # Configure CUDA backends before any allocation
        configure_cuda_backends()

        model, tokenizer = self._loader.load()
        self._runtime.model = model
        self._runtime.tokenizer = tokenizer
        try:
            self._runtime.device = next(model.parameters()).device
        except StopIteration:
            self._runtime.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate static decode input buffer (reused every decode step)
        self._input_buffer.allocate(self._runtime.device)

        # The graph decoder must come up before the KV block budget is
        # estimated: its StaticCache reserves VRAM for the full context
        # (materialized during the validation probe), and the estimator
        # reads free VRAM at estimation time.
        self._init_speculative_engine()
        self._init_graph_decoder()

        self._runtime.kv_cache_manager = KVCacheManager(self.kv_cache_config)

        # Feed actual model dimensions into KV cache for precise memory estimation
        kv_params = self._loader.get_kv_cache_params()
        if kv_params.get("num_layers") and kv_params.get("num_kv_heads") and kv_params.get("head_dim"):
            self._runtime.kv_cache_manager.update_model_params(
                num_layers=kv_params["num_layers"],
                num_kv_heads=kv_params["num_kv_heads"],
                head_dim=kv_params["head_dim"],
            )

        self._ready = True
        logger.info("InferenceEngine ready. GPU: %s", get_gpu_memory_info())

    def _probe_tokens(self) -> list[int]:
        """Short token sequence for load-time validation probes."""
        return self._runtime.tokenize("Hello world")[:8] or [1, 2, 3]

    def _init_graph_decoder(self):
        """Set up CUDA-graph decode if enabled, validated against eager output."""
        if not self.model_config.enable_cuda_graphs:
            return
        from .cudagraph import CUDAGraphDecoder

        decoder = CUDAGraphDecoder(self._runtime.model, self.model_config, self._runtime.device)
        if not decoder.ok:
            return
        if decoder.validate(self._probe_tokens()):
            self._runtime.graph_decoder = decoder

    def _ensure_grammar_states(self, requests: list[GenerationRequest]) -> None:
        """Attach a grammar matcher to requests asking for structured output.

        Failures (unparseable schema, backend init) fail only the offending
        request — an exception escaping here would make the scheduler fail
        every active request in the batch.
        """
        for req in requests:
            rf = req.sampling_params.response_format
            if rf and req._grammar is None:
                try:
                    if self._grammar_backend is None:
                        from ..sampling.grammar import GrammarBackend
                        self._grammar_backend = GrammarBackend(self._runtime.tokenizer)
                    req._grammar = self._grammar_backend.state_for(rf)
                except Exception as e:
                    req.status = RequestStatus.FAILED
                    req.error = f"Invalid response_format: {e}"
                    logger.warning("Grammar setup failed for request %s: %s", req.request_id, e)

    def _init_speculative_engine(self):
        """Set up speculative decoding: draft model if loaded, else suffix matching."""
        if self._loader.draft_model:
            from .speculative import SpeculativeEngine
            self._runtime.speculative_engine = SpeculativeEngine(
                target_model=self._runtime.model,
                draft_model=self._loader.draft_model,
                tokenizer=self._runtime.tokenizer,
            )
            logger.info("Speculative decoding enabled using draft model")
        elif self.model_config.enable_suffix_decoding:
            from .suffix_speculative import SuffixSpeculativeEngine
            spec = SuffixSpeculativeEngine(
                target_model=self._runtime.model,
                tokenizer=self._runtime.tokenizer,
                input_buffer=self._input_buffer,
            )
            # Rejection-rollback must be lossless for this architecture's
            # cache, or speculation would corrupt output (e.g. hybrid conv
            # caches don't rewind). Probe before enabling.
            if spec.validate(self._probe_tokens()):
                self._runtime.speculative_engine = spec
                logger.info("Suffix speculative decoding enabled (model-free)")

    def unload_model(self):
        """Unload model and free resources."""
        self._ready = False
        self._grammar_backend = None
        self._loader.unload()
        if self._runtime.kv_cache_manager:
            self._runtime.kv_cache_manager.reset()
        self._runtime.clear()

        # Free performance buffers to release GPU memory
        self._input_buffer.release()
        self._batch_cache.release()

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Tokenize input text."""
        return self._runtime.tokenize(text)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self._runtime.decode_tokens(token_ids)

    # ------------------------------------------------------------------
    # Batched step-level generation (used by the Scheduler)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_step(
        self,
        requests: list[GenerationRequest],
        chunked_prefill_enabled: bool = False,
        max_num_batched_tokens: int = 512,
    ) -> list[GenerationRequest]:
        """Perform a single inference step for a batch of requests.

        This handles both:
          - Prefill for NEW requests (first time the model sees the prompt).
          - Decode for EXISTING requests (generate one more token).
          - Chunked prefill: processing long prompts incrementally.
        """
        if not requests:
            return []

        device = self._runtime.resolve_device()
        self._ensure_grammar_states(requests)

        # Separate requests by prefill progress; requests failed during setup
        # (e.g. bad grammar) are skipped and collected by the scheduler.
        live = [r for r in requests if r.status != RequestStatus.FAILED]
        prefill_reqs = [r for r in live if not r.is_prefill_complete]
        decode_reqs = [r for r in live if r.is_prefill_complete]

        if prefill_reqs:
            self._prefill_runner.run(
                prefill_reqs, device, chunked_prefill_enabled, max_num_batched_tokens
            )

        if decode_reqs:
            self._decode_runner.run(decode_reqs, device)

        return requests

    # ------------------------------------------------------------------
    # Full single-request generation (blocking)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationRequest:
        """Run full generation for a single request (blocking)."""
        if not self._ready:
            request.status = RequestStatus.FAILED
            request.error = "Engine not ready -- model not loaded"
            return request

        request.status = RequestStatus.RUNNING
        request.started_at = time.time()

        try:
            self._ensure_grammar_states([request])
            return self._blocking_generator.generate(request)
        except Exception as e:
            request.status = RequestStatus.FAILED
            request.error = str(e)
            logger.exception("Generation failed for request %s", request.request_id)
            return request

    # ------------------------------------------------------------------
    # Async streaming interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[tuple[str, bool]]:
        """Async generator that yields (token_text, is_finished) tuples.

        Uses asyncio.Queue with call_soon_threadsafe for a proper
        async/sync bridge -- no busy-wait polling.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        token_queue: asyncio.Queue[tuple[str, bool] | BaseException] = asyncio.Queue()

        def stream_callback(text: str, finished: bool):
            loop.call_soon_threadsafe(token_queue.put_nowait, (text, finished))

        request._stream_callback = stream_callback

        def _run_generate():
            try:
                self.generate(request)
            except BaseException as exc:
                # Push the exception so the async consumer can surface it
                loop.call_soon_threadsafe(token_queue.put_nowait, exc)

        # Run generation in a thread to not block the event loop
        gen_task = loop.run_in_executor(None, _run_generate)

        # Yield tokens as they arrive
        while True:
            item = await token_queue.get()

            # If the generation thread raised, propagate the error
            if isinstance(item, BaseException):
                raise item

            text, finished = item
            yield text, finished
            if finished:
                break

        # Ensure gen_task completes
        await gen_task

