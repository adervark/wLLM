"""Core inference engine — prefill and decode loop with streaming."""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig, SamplingParams, KVCacheConfig
from .kv_cache import KVCacheManager
from .model_loader import ModelLoader, get_gpu_memory_info
from .sampler import sample_token
from .types import GenerationRequest, RequestStatus

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
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.kv_cache_manager: Optional[KVCacheManager] = None
        self.speculative_engine: Optional['SpeculativeEngine'] = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load_model(self):
        """Load model and initialize KV cache manager."""
        self.model, self.tokenizer = self._loader.load()
        self.kv_cache_manager = KVCacheManager(self.kv_cache_config)

        # Feed actual model dimensions into KV cache for precise estimation
        kv_params = self._loader.get_kv_cache_params()
        if kv_params.get("num_layers") and kv_params.get("num_kv_heads") and kv_params.get("head_dim"):
            self.kv_cache_manager.update_model_params(
                num_layers=kv_params["num_layers"],
                num_kv_heads=kv_params["num_kv_heads"],
                head_dim=kv_params["head_dim"],
            )

        # Initialize speculative engine if draft model is available
        if self._loader.draft_model:
            from .speculative import SpeculativeEngine
            self.speculative_engine = SpeculativeEngine(
                target_model=self.model,
                draft_model=self._loader.draft_model,
                tokenizer=self.tokenizer
            )
            logger.info("Speculative decoding enabled using draft model")

        if self.model_config.compile:
            logger.info("Compiling model with torch.compile (mode=reduce-overhead)...")
            try:
                # Use reduce-overhead to stay within VRAM and avoid too many re-compilations
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compilation scheduled (will occur during first forward pass)")
            except Exception as e:
                logger.warning("Failed to compile model: %s. Falling back to eager mode.", e)

        self._ready = True
        logger.info("InferenceEngine ready. GPU: %s", get_gpu_memory_info())

    def unload_model(self):
        """Unload model and free resources."""
        self._ready = False
        self._loader.unload()
        if self.kv_cache_manager:
            self.kv_cache_manager.reset()
        self.model = None
        self.tokenizer = None

    def tokenize(self, text: str) -> list[int]:
        """Tokenize input text."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_step(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
        """Perform a single inference step for a batch of requests.
        
        This handles both prefill for new requests and one decode step 
        for existing ones. It assumes all requests are on the same device.
        """
        if not requests:
            return []

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sort requests: prefill vs decode
        prefill_reqs = [r for r in requests if r._past_key_values is None]
        decode_reqs = [r for r in requests if r._past_key_values is not None]
        # 1. Prefill Phase
        for req in prefill_reqs:
            if not req.prompt_token_ids:
                req.prompt_token_ids = self.tokenize(req.prompt)
            
            req.started_at = time.time()
            req.status = RequestStatus.RUNNING
            
            # Prefix handling: skip prefilling tokens already in cache
            if req._prefix_past_key_values is not None:
                # We only prefill the suffix
                input_ids = torch.tensor([req.prompt_token_ids[req._prefix_len:]], device=device)
                outputs = self.model(
                    input_ids, 
                    past_key_values=req._prefix_past_key_values,
                    use_cache=True
                )
            else:
                input_ids = torch.tensor([req.prompt_token_ids], device=device)
                outputs = self.model(input_ids, use_cache=True)
            
            req._past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            
            # Sample first token
            next_token_id = sample_token(next_logits, req.sampling_params, req.output_token_ids)
            next_token = next_token_id.item()
            req.output_token_ids.append(next_token)
            
            # Initialize streaming prefix
            prompt_text = self.tokenizer.decode(req.prompt_token_ids, skip_special_tokens=True)
            req._prefix_len = len(prompt_text)
            self._stream_req_token(req)


        # 2. Decode Phase (Batched)
        if decode_reqs:
            # Prepare batched inputs
            # NOTE: True batched decoding with different lengths requires padding and attention masks.
            # However, in iteration-level scheduling, all sequences in the "decode batch" 
            # have exactly 1 new token input, so we can batch them easily IF the KV cache is handled.
            
            # For now, we perform sequential decode steps to ensure stability, 
            # then we'll vectorize this in Stage 2.1.
            for req in decode_reqs:
                if req.is_cancelled:
                    continue
                
                # Use speculative engine if available and only one request is being processed
                # (Speculative batching is a future enhancement)
                if self.speculative_engine and len(decode_reqs) == 1:
                    self.speculative_engine.step(req)
                    self.kv_cache_manager.extend_sequence(req.request_id, len(req.output_token_ids) - req.generation_tokens)
                    self._stream_req_token(req)
                    continue

                last_token = req.output_token_ids[-1]
                input_ids = torch.tensor([[last_token]], device=device)
                
                outputs = self.model(
                    input_ids,
                    past_key_values=req._past_key_values,
                    use_cache=True
                )
                
                req._past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
                
                next_token_id = sample_token(next_logits, req.sampling_params, req.output_token_ids)
                next_token = next_token_id.item()
                req.output_token_ids.append(next_token)
                
                self.kv_cache_manager.extend_sequence(req.request_id, 1)
                self._stream_req_token(req)

        return requests

    def _stream_req_token(self, request: GenerationRequest):
        """Helper to stream the latest token ID (Stage 7 optimization)."""
        if request._token_callback:
            # Pass the raw token ID (last one) to the callback for async processing
            last_id = request.output_token_ids[-1]
            request._token_callback(last_id, False)
            return

        if not request._stream_callback:
            return
            
        # Fallback for old blocking style:
        full_ids = request.prompt_token_ids + request.output_token_ids
        current_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        
        if len(current_text) > request._prefix_len:
            new_text = current_text[request._prefix_len:]
            request._stream_callback(new_text, False)
            request._prefix_len = len(current_text)

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationRequest:
        """Run full generation for a single request (blocking).

        This uses a manual decode loop for fine-grained control,
        streaming support, and integration with KV cache tracking.
        """
        if not self._ready:
            request.status = RequestStatus.FAILED
            request.error = "Engine not ready — model not loaded"
            return request

        request.status = RequestStatus.RUNNING
        request.started_at = time.time()

        try:
            return self._generate_impl(request)
        except Exception as e:
            request.status = RequestStatus.FAILED
            request.error = str(e)
            logger.exception("Generation failed for request %s", request.request_id)
            return request

    def _generate_impl(self, request: GenerationRequest) -> GenerationRequest:
        """Internal generation implementation."""
        params = request.sampling_params

        # For multi-GPU / device_map, resolve the device from the input embeddings
        # since self.model.device can be 'meta' or inconsistent with sharded models
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenize prompt
        if not request.prompt_token_ids:
            request.prompt_token_ids = self.tokenize(request.prompt)

        prompt_len = len(request.prompt_token_ids)

        # Check length limits
        if prompt_len >= self.model_config.max_model_len:
            request.status = RequestStatus.FAILED
            request.error = f"Prompt too long: {prompt_len} tokens (max {self.model_config.max_model_len})"
            return request

        max_new_tokens = min(
            params.max_tokens,
            self.model_config.max_model_len - prompt_len,
        )

        # Allocate KV cache tracking
        total_tokens_needed = prompt_len + max_new_tokens
        if not self.kv_cache_manager.can_allocate(total_tokens_needed):
            request.status = RequestStatus.FAILED
            request.error = "Insufficient KV cache memory"
            return request
        self.kv_cache_manager.allocate_sequence(request.request_id, prompt_len)

        # Prepare input
        input_ids = torch.tensor(
            [request.prompt_token_ids], dtype=torch.long, device=device
        )

        # Setup generator for reproducibility
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(params.seed)

        # Get stop token IDs
        eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = {eos_token_id} if eos_token_id is not None else set()

        # Convert stop strings to token sequences for checking
        stop_strings = params.stop if params.stop else []

        # === Prefill: process entire prompt ===
        outputs = self.model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Sample first token
        next_token_id = sample_token(
            next_token_logits, params, request.output_token_ids, generator
        )
        next_token = next_token_id.item()
        request.output_token_ids.append(next_token)

        # We need to handle subwords and spaces properly in the stream.
        # Natively decoding one token at a time drops leading spaces for many tokenizers (like SentencePiece).
        # To fix this, we decode the full sequence (prompt + output) to ensure correct spacing,
        # and only yield the new substring appended to the prefix.
        
        initial_prompt_text = self.tokenizer.decode(
            request.prompt_token_ids, skip_special_tokens=True
        )
        prefix_len = len(initial_prompt_text)
        
        def _stream_token():
            nonlocal prefix_len
            if request._stream_callback:
                # Decode prompt + current output to maintain subword boundaries
                full_ids = request.prompt_token_ids + request.output_token_ids
                current_text = self.tokenizer.decode(
                    full_ids, skip_special_tokens=True
                )
                if len(current_text) > prefix_len:
                    request._stream_callback(current_text[prefix_len:], False)
                    prefix_len = len(current_text)

        _stream_token()

        # === Decode loop ===
        for step in range(1, max_new_tokens):
            # Check stop conditions
            if next_token in stop_token_ids:
                break

            # Check cancellation
            if request.is_cancelled:
                request.status = RequestStatus.CANCELLED
                break

            # Check stop strings
            current_text = self.tokenizer.decode(
                request.output_token_ids, skip_special_tokens=True
            )
            if any(s in current_text for s in stop_strings):
                # Trim output at stop string
                for s in stop_strings:
                    idx = current_text.find(s)
                    if idx != -1:
                        current_text = current_text[:idx]
                        break
                request.output_text = current_text
                break

            # Forward pass with KV cache
            token_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            outputs = self.model(
                token_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Sample
            next_token_id = sample_token(
                next_token_logits, params, request.output_token_ids, generator
            )
            next_token = next_token_id.item()
            request.output_token_ids.append(next_token)

            # Update KV cache tracking
            self.kv_cache_manager.extend_sequence(request.request_id, 1)

            # Stream token
            _stream_token()

        # Finalize
        if not request.output_text:
            request.output_text = self.tokenizer.decode(
                request.output_token_ids, skip_special_tokens=True
            )

        if request.status != RequestStatus.CANCELLED:
            request.status = RequestStatus.COMPLETED
        request.finished_at = time.time()

        # Free KV cache tracking
        self.kv_cache_manager.free_sequence(request.request_id)

        # Signal stream end
        if request._stream_callback:
            request._stream_callback("", True)

        logger.info(
            "Request %s completed: %d tokens in %.2fs (%.1f tok/s)",
            request.request_id,
            request.generation_tokens,
            request.elapsed,
            request.tokens_per_second,
        )

        return request

    @torch.inference_mode()
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[tuple[str, bool]]:
        """Async generator that yields (token_text, is_finished) tuples.

        Uses asyncio.Queue with call_soon_threadsafe for a proper
        async/sync bridge — no busy-wait polling.
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

