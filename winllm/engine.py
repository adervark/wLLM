"""Core inference engine -- prefill and decode loop with streaming.

This module contains the InferenceEngine, which is responsible for:
  1. Loading a model and tokenizer via ModelLoader.
  2. Running the "prefill" step (processing the full prompt in one pass).
  3. Running the "decode" loop (generating tokens one at a time using KV cache).
  4. Streaming generated tokens back to callers in real time.

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
import os
import time
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
        self._device: Optional[torch.device] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.kv_cache_manager: Optional[KVCacheManager] = None
        self.speculative_engine: Optional['SpeculativeEngine'] = None
        self._ready = False

        # --- Performance: pre-allocated decode buffers ---
        self._decode_input_buffer: Optional[torch.Tensor] = None
        # Persistent batch KV buffers to avoid per-step allocation
        self._batch_kv_buffers: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self._batch_attn_mask: Optional[torch.Tensor] = None
        self._batch_buf_max_seq: int = 0
        self._batch_buf_batch_size: int = 0
        self._batch_buf_num_layers: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready

    @staticmethod
    def _configure_cuda_backends():
        """Configure CUDA backends for maximum inference throughput.

        Called once before model loading. These are global PyTorch settings:
          - TF32: Enables TF32 tensor cores on Ampere+ GPUs for ~3x faster
            float32 matmuls with negligible accuracy impact for inference.
          - Expandable segments: Reduces CUDA memory fragmentation on Windows
            where the default allocator is particularly wasteful.
        """
        if not torch.cuda.is_available():
            return

        # TF32 for Ampere+ (compute >= 8.0) -- free 2-3x speedup on float32 ops
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

        # Expandable segments allocator -- reduces fragmentation-induced OOMs
        alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        if 'expandable_segments' not in alloc_conf:
            new_conf = 'expandable_segments:True'
            if alloc_conf:
                new_conf = f'{alloc_conf},{new_conf}'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = new_conf

        logger.info("CUDA backends configured: TF32=on, allocator=%s",
                    os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'))

    def load_model(self):
        """Load model and initialize KV cache manager."""
        # Configure CUDA backends before any allocation
        self._configure_cuda_backends()

        self.model, self.tokenizer = self._loader.load()
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate static decode input buffer (reused every decode step)
        self._decode_input_buffer = torch.zeros(
            (1, 1), dtype=torch.long, device=self._device
        )

        self.kv_cache_manager = KVCacheManager(self.kv_cache_config)

        # Feed actual model dimensions into KV cache for precise memory estimation
        kv_params = self._loader.get_kv_cache_params()
        if kv_params.get("num_layers") and kv_params.get("num_kv_heads") and kv_params.get("head_dim"):
            self.kv_cache_manager.update_model_params(
                num_layers=kv_params["num_layers"],
                num_kv_heads=kv_params["num_kv_heads"],
                head_dim=kv_params["head_dim"],
            )

        self._init_speculative_engine()

        self._ready = True
        logger.info("InferenceEngine ready. GPU: %s", get_gpu_memory_info())

    def _init_speculative_engine(self):
        """Set up speculative decoding if a draft model was loaded."""
        if self._loader.draft_model:
            from .speculative import SpeculativeEngine
            self.speculative_engine = SpeculativeEngine(
                target_model=self.model,
                draft_model=self._loader.draft_model,
                tokenizer=self.tokenizer
            )
            logger.info("Speculative decoding enabled using draft model")

    def unload_model(self):
        """Unload model and free resources."""
        self._ready = False
        self._loader.unload()
        if self.kv_cache_manager:
            self.kv_cache_manager.reset()
        self.model = None
        self.tokenizer = None

        # Free performance buffers to release GPU memory
        self._decode_input_buffer = None
        self._batch_kv_buffers = None
        self._batch_attn_mask = None
        self._batch_buf_max_seq = 0
        self._batch_buf_batch_size = 0
        self._batch_buf_num_layers = 0

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Tokenize input text."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Batched step-level generation (used by the Scheduler)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_step(self, requests: list[GenerationRequest], chunked_prefill_enabled: bool = False, max_num_batched_tokens: int = 512) -> list[GenerationRequest]:
        """Perform a single inference step for a batch of requests.

        This handles both:
          - Prefill for NEW requests (first time the model sees the prompt).
          - Decode for EXISTING requests (generate one more token).
          - Chunked prefill: processing long prompts incrementally.
        """
        if not requests:
            return []

        device = self._resolve_device()

        # Separate requests by prefill progress
        prefill_reqs = [r for r in requests if not r.is_prefill_complete]
        decode_reqs = [r for r in requests if r.is_prefill_complete]

        # Batch prefill when multiple new requests arrive simultaneously
        if prefill_reqs:
            # Prefix-cache hits or chunked mode need per-request handling
            has_prefix_hits = any(r._prefix_past_key_values is not None for r in prefill_reqs)
            if len(prefill_reqs) == 1 or chunked_prefill_enabled or has_prefix_hits:
                for req in prefill_reqs:
                    self._prefill_single_request(req, device, chunked_prefill_enabled, max_num_batched_tokens)
            else:
                # Multiple non-chunked prefills without cache hits: batch into one forward pass
                self._prefill_batch(prefill_reqs, device)

        if decode_reqs:
            if len(decode_reqs) == 1:
                self._decode_single_request(decode_reqs[0], device, batch_size=1)
            else:
                self._decode_batch(decode_reqs, device)

        return requests

    def _resolve_device(self) -> torch.device:
        """Figure out which device the model is actually on."""
        return self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prefill_single_request(self, req: GenerationRequest, device: torch.device, chunked_prefill: bool, max_tokens: int):
        """Run the prefill forward pass for a single new request.

        Processes the prompt in one pass, or in chunks if chunked prefill is enabled. 
        Generates the first output token only after the final chunk completes.
        """
        if not req.prompt_token_ids:
            req.prompt_token_ids = self.tokenize(req.prompt)

        req.started_at = time.time()
        req.status = RequestStatus.RUNNING

        # Fast forward cursor if prefix cache hit
        if req._prefill_cursor == 0 and req._prefix_past_key_values is not None:
            req._prefill_cursor = req._prefix_cache_token_len
            req._past_key_values = req._prefix_past_key_values

        start_idx = req._prefill_cursor
        remaining_tokens = len(req.prompt_token_ids) - start_idx
        
        if chunked_prefill and remaining_tokens > max_tokens:
            end_idx = start_idx + max_tokens
            is_final_chunk = False
        else:
            end_idx = start_idx + remaining_tokens
            is_final_chunk = True

        chunk_ids = req.prompt_token_ids[start_idx:end_idx]
        input_ids = torch.tensor([chunk_ids], device=device)

        outputs = self.model(
            input_ids,
            past_key_values=req._past_key_values,
            use_cache=True
        )

        req._past_key_values = outputs.past_key_values
        req._prefill_cursor = end_idx

        # Only sample if we've processed the entire prompt
        if is_final_chunk:
            next_logits = outputs.logits[:, -1, :]
            next_token_id = sample_token(next_logits, req.sampling_params, req.output_token_ids)
            req.output_token_ids.append(next_token_id.item())

            prompt_text = self.tokenizer.decode(req.prompt_token_ids, skip_special_tokens=True)
            req._stream_text_cursor = len(prompt_text)
            self._emit_stream_token(req)

    def _prefill_batch(self, reqs: list[GenerationRequest], device: torch.device):
        """Batch multiple prefill requests into a single forward pass.

        Uses left-padding to align variable-length prompts so they can be processed
        together. This is 2-5x faster than sequential prefill when multiple requests
        arrive simultaneously (common under API load).
        """
        now = time.time()
        pad_token_id = self.tokenizer.pad_token_id or 0

        # Tokenize any un-tokenized requests
        for req in reqs:
            if not req.prompt_token_ids:
                req.prompt_token_ids = self.tokenize(req.prompt)
            req.started_at = now
            req.status = RequestStatus.RUNNING

        # Build left-padded input batch
        max_len = max(len(req.prompt_token_ids) for req in reqs)
        batch_ids = []
        attention_masks = []
        for req in reqs:
            tokens = req.prompt_token_ids
            pad_len = max_len - len(tokens)
            batch_ids.append([pad_token_id] * pad_len + tokens)
            attention_masks.append([0] * pad_len + [1] * len(tokens))

        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        # Unbatch: extract per-request KV caches and sample first token
        num_layers = len(outputs.past_key_values)
        for i, req in enumerate(reqs):
            prompt_len = len(req.prompt_token_ids)

            # Slice KV cache to only this request's actual tokens (remove padding)
            req._past_key_values = tuple(
                (
                    outputs.past_key_values[layer][0][i:i+1, :, -prompt_len:, :].clone(),
                    outputs.past_key_values[layer][1][i:i+1, :, -prompt_len:, :].clone(),
                )
                for layer in range(num_layers)
            )
            req._prefill_cursor = prompt_len

            # Sample first output token from this request's logits
            # The last non-padding position for this request is at index (max_len - 1)
            # since we left-padded and the model outputs logits for all positions
            next_logits = outputs.logits[i:i+1, -1, :]
            next_token_id = sample_token(next_logits, req.sampling_params, req.output_token_ids)
            req.output_token_ids.append(next_token_id.item())

            prompt_text = self.tokenizer.decode(req.prompt_token_ids, skip_special_tokens=True)
            req._stream_text_cursor = len(prompt_text)
            self._emit_stream_token(req)

    def _decode_single_request(self, req: GenerationRequest, device: torch.device, batch_size: int = 1):
        """Run one decode step for a request that's already being generated.

        Takes the last generated token, feeds it through the model with
        the KV cache, and samples the next token. Uses speculative decoding
        when available and there's only one request in the batch.
        """
        if req.is_cancelled:
            return

        # Speculative decoding: use the fast draft model to propose tokens,
        # then verify them with the target model in one pass.
        # Only works for single-request batches (no batched speculation yet).
        if self.speculative_engine and batch_size == 1:
            tokens_before = len(req.output_token_ids)
            self.speculative_engine.step(req)
            tokens_added = len(req.output_token_ids) - tokens_before
            if tokens_added > 0:
                self.kv_cache_manager.extend_sequence(req.request_id, tokens_added)
            self._emit_stream_token(req)
            return

        # Reuse static input buffer to avoid per-step CUDA allocation
        last_token = req.output_token_ids[-1]
        if self._decode_input_buffer is not None and self._decode_input_buffer.device == device:
            self._decode_input_buffer[0, 0] = last_token
            input_ids = self._decode_input_buffer
        else:
            input_ids = torch.tensor([[last_token]], device=device)

        outputs = self.model(
            input_ids,
            past_key_values=req._past_key_values,
            use_cache=True
        )

        req._past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        next_token_id = sample_token(next_logits, req.sampling_params, req.output_token_ids)
        req.output_token_ids.append(next_token_id.item())

        self.kv_cache_manager.extend_sequence(req.request_id, 1)
        self._emit_stream_token(req)

    def _ensure_batch_kv_buffers(
        self, num_layers: int, batch_size: int, max_seq_len: int,
        num_kv_heads: int, head_dim: int, kv_dtype: torch.dtype, device: torch.device
    ):
        """Ensure persistent KV batch buffers are large enough, expanding only when needed.

        Instead of allocating fresh zero tensors every decode step (the old approach),
        we maintain persistent buffers and only re-allocate when batch_size or max_seq_len
        grows beyond what we've already allocated. This eliminates thousands of CUDA
        malloc/free cycles per generation.
        """
        needs_realloc = (
            self._batch_kv_buffers is None
            or self._batch_buf_num_layers != num_layers
            or batch_size > self._batch_buf_batch_size
            or max_seq_len > self._batch_buf_max_seq
        )
        if needs_realloc:
            # Over-allocate seq dimension by 25% to avoid frequent reallocs
            alloc_seq = max(max_seq_len, int(max_seq_len * 1.25))
            self._batch_kv_buffers = [
                (
                    torch.zeros((batch_size, num_kv_heads, alloc_seq, head_dim), dtype=kv_dtype, device=device),
                    torch.zeros((batch_size, num_kv_heads, alloc_seq, head_dim), dtype=kv_dtype, device=device),
                )
                for _ in range(num_layers)
            ]
            self._batch_attn_mask = torch.zeros((batch_size, alloc_seq + 1), dtype=torch.long, device=device)
            self._batch_buf_max_seq = alloc_seq
            self._batch_buf_batch_size = batch_size
            self._batch_buf_num_layers = num_layers
            logger.debug("Batch KV buffers (re)allocated: layers=%d, batch=%d, seq=%d", num_layers, batch_size, alloc_seq)

    def _decode_batch(self, decode_reqs: list[GenerationRequest], device: torch.device):
        """Run a single forward pass for a batch of requests.

        This is the primary throughput optimization, moving from O(N) model passes
        to O(1) for a batch of size N. Uses persistent KV buffers to avoid
        re-allocating large tensors every step.
        """
        # Filter out cancelled requests before wasting a forward pass on them
        decode_reqs = [r for r in decode_reqs if not r.is_cancelled]
        if not decode_reqs:
            return

        batch_size = len(decode_reqs)
        
        # 1. Prepare batched inputs
        input_ids = torch.tensor(
            [[req.output_token_ids[-1]] for req in decode_reqs], 
            device=device
        )
        
        # --- Batch the past_key_values using persistent buffers ---
        seq_lengths = [len(req.prompt_token_ids) + len(req.output_token_ids) - 1 for req in decode_reqs]
        max_seq_len = max(seq_lengths)
        
        num_layers = len(decode_reqs[0]._past_key_values)
        
        # Extract shapes dynamically from the first valid state to handle any attention architecture
        sample_k = decode_reqs[0]._past_key_values[0][0]
        num_kv_heads = sample_k.shape[1]
        head_dim = sample_k.shape[3]
        kv_dtype = sample_k.dtype
        
        # Ensure persistent buffers are large enough
        self._ensure_batch_kv_buffers(num_layers, batch_size, max_seq_len, num_kv_heads, head_dim, kv_dtype, device)
        
        batched_past_key_values = []
        for layer_idx in range(num_layers):
            buf_k, buf_v = self._batch_kv_buffers[layer_idx]
            # Slice to actual needed size and zero
            k_view = buf_k[:batch_size, :, :max_seq_len, :]
            v_view = buf_v[:batch_size, :, :max_seq_len, :]
            k_view.zero_()
            v_view.zero_()
            
            for i, req in enumerate(decode_reqs):
                k, v = req._past_key_values[layer_idx]
                seq_len = seq_lengths[i]
                k_view[i:i+1, :, -seq_len:, :] = k
                v_view[i:i+1, :, -seq_len:, :] = v
            
            # Use contiguous slices (they already are since we sliced from contiguous buffers)
            batched_past_key_values.append((k_view, v_view))

        # Create attention mask using persistent buffer
        attn_buf = self._batch_attn_mask[:batch_size, :max_seq_len + 1]
        attn_buf.zero_()
        for i, seq_len in enumerate(seq_lengths):
            attn_buf[i, -(seq_len + 1):] = 1
            
        position_ids = attn_buf.sum(dim=1) - 1
        position_ids = position_ids.unsqueeze(-1)
        
        # 4. Forward pass
        outputs = self.model(
            input_ids,
            past_key_values=batched_past_key_values,
            use_cache=True,
            attention_mask=attn_buf,
            position_ids=position_ids
        )
        
        new_past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :] 
        
        next_token_ids = sample_token(
            next_logits, 
            [req.sampling_params for req in decode_reqs],
            [req.output_token_ids for req in decode_reqs]
        )
        
        for i, req in enumerate(decode_reqs):
            new_seq_len = seq_lengths[i] + 1
            req._past_key_values = tuple(
                (
                    new_past_key_values[layer_idx][0][i:i+1, :, -new_seq_len:, :].clone(), 
                    new_past_key_values[layer_idx][1][i:i+1, :, -new_seq_len:, :].clone()
                )
                for layer_idx in range(num_layers)
            )
            
            token_id = next_token_ids[i].item()
            req.output_token_ids.append(token_id)
            
            self.kv_cache_manager.extend_sequence(req.request_id, 1)
            self._emit_stream_token(req)

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _emit_stream_token(self, request: GenerationRequest):
        """Send the latest generated token to the registered callback.

        There are two callback modes:
          - _token_callback: receives raw token IDs (preferred, faster).
          - _stream_callback: receives decoded text deltas (legacy fallback).
        """
        # Preferred path: raw token ID callback (used by the API server)
        if request._token_callback:
            last_id = request.output_token_ids[-1]
            request._token_callback(last_id, False)
            return

        # Legacy path: decoded text callback (used by the chat CLI)
        if not request._stream_callback:
            return

        # Decode only the latest token to avoid O(n²) full-sequence re-decode.
        # Note: single-token decoding may differ slightly from full-sequence
        # decoding for tokenizers with context-dependent spacing (e.g.
        # SentencePiece), but this is acceptable for streaming — the final
        # output_text is always decoded from the full sequence.
        new_text = self.tokenizer.decode(
            request.output_token_ids[-1:], skip_special_tokens=True
        )
        if new_text:
            request._stream_callback(new_text, False)

    # ------------------------------------------------------------------
    # Full single-request generation (blocking)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationRequest:
        """Run full generation for a single request (blocking).

        This uses a manual decode loop for fine-grained control,
        streaming support, and integration with KV cache tracking.
        """
        if not self._ready:
            request.status = RequestStatus.FAILED
            request.error = "Engine not ready -- model not loaded"
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
        """Internal generation implementation.

        Orchestrates the full lifecycle: validate -> prefill -> decode -> finalize.
        """
        params = request.sampling_params
        device = self._resolve_device()

        # --- Step 1: Tokenize the prompt ---
        if not request.prompt_token_ids:
            request.prompt_token_ids = self.tokenize(request.prompt)
        prompt_len = len(request.prompt_token_ids)

        # --- Step 2: Validate that the prompt fits ---
        if not self._validate_prompt(request, prompt_len):
            return request

        max_new_tokens = min(
            params.max_tokens,
            self.model_config.max_model_len - prompt_len,
        )

        # --- Step 3: Allocate KV cache memory ---
        if not self._allocate_kv_cache(request, prompt_len, max_new_tokens):
            return request

        # --- Step 4: Prefill (process entire prompt at once) ---
        input_ids = torch.tensor(
            [request.prompt_token_ids], dtype=torch.long, device=device
        )
        generator = self._make_generator(params, device)
        stop_token_ids, stop_strings = self._get_stop_conditions(params)

        outputs = self.model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Sample the first token
        next_token_id = sample_token(next_token_logits, params, request.output_token_ids, generator)
        next_token = next_token_id.item()
        request.output_token_ids.append(next_token)

        # Set up streaming text prefix tracking.
        # We decode the full sequence (prompt + output) to ensure correct spacing
        # for tokenizers like SentencePiece that handle leading spaces specially.
        initial_prompt_text = self.tokenizer.decode(
            request.prompt_token_ids, skip_special_tokens=True
        )
        prefix_len = len(initial_prompt_text)

        def _stream_token():
            nonlocal prefix_len
            if request._stream_callback:
                full_ids = request.prompt_token_ids + request.output_token_ids
                current_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
                if len(current_text) > prefix_len:
                    request._stream_callback(current_text[prefix_len:], False)
                    prefix_len = len(current_text)

        _stream_token()

        # --- Step 5: Decode loop (generate tokens one at a time) ---
        next_token, past_key_values = self._run_decode_loop(
            request, params, device, generator,
            next_token, past_key_values,
            max_new_tokens, stop_token_ids, stop_strings,
            _stream_token,
        )

        # --- Step 6: Finalize ---
        self._finalize_generation(request)

        return request

    def _validate_prompt(self, request: GenerationRequest, prompt_len: int) -> bool:
        """Check that the prompt fits within the model's context window.

        Returns True if valid, False if the request was marked as failed.
        """
        if prompt_len >= self.model_config.max_model_len:
            request.status = RequestStatus.FAILED
            request.error = f"Prompt too long: {prompt_len} tokens (max {self.model_config.max_model_len})"
            return False
        return True

    def _allocate_kv_cache(self, request: GenerationRequest, prompt_len: int, max_new_tokens: int) -> bool:
        """Reserve KV cache blocks for this request.

        Returns True if allocation succeeded, False if insufficient memory.
        """
        total_tokens_needed = prompt_len + max_new_tokens
        if not self.kv_cache_manager.can_allocate(total_tokens_needed):
            request.status = RequestStatus.FAILED
            request.error = "Insufficient KV cache memory"
            return False
        self.kv_cache_manager.allocate_sequence(request.request_id, prompt_len)
        return True

    def _make_generator(self, params: SamplingParams, device: torch.device) -> Optional[torch.Generator]:
        """Create a random number generator for reproducible sampling."""
        if params.seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(params.seed)
            return gen
        return None

    def _get_stop_conditions(self, params: SamplingParams) -> tuple[set[int], list[str]]:
        """Build the set of stop token IDs and stop strings."""
        eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = {eos_token_id} if eos_token_id is not None else set()
        stop_strings = params.stop if params.stop else []
        return stop_token_ids, stop_strings

    def _run_decode_loop(
        self,
        request: GenerationRequest,
        params: SamplingParams,
        device: torch.device,
        generator: Optional[torch.Generator],
        next_token: int,
        past_key_values,
        max_new_tokens: int,
        stop_token_ids: set[int],
        stop_strings: list[str],
        stream_fn: Callable,
    ) -> tuple[int, object]:
        """Run the autoregressive decode loop, generating tokens one at a time.

        Returns the last generated token and the final past_key_values.
        """
        # Maintain running output text to avoid O(n²) re-decoding for stop-string checks
        running_text = ""

        for step in range(1, max_new_tokens):
            # Check if we hit an end-of-sequence token
            if next_token in stop_token_ids:
                break

            # Check if the request was cancelled externally
            if request.is_cancelled:
                request.status = RequestStatus.CANCELLED
                break

            # Incrementally extend running text for stop-string checking
            if stop_strings:
                new_piece = self.tokenizer.decode(
                    [next_token], skip_special_tokens=True
                )
                running_text += new_piece
                if any(s in running_text for s in stop_strings):
                    for s in stop_strings:
                        idx = running_text.find(s)
                        if idx != -1:
                            request.output_text = running_text[:idx]
                            break
                    break

            # Forward pass: feed the last token, reuse KV cache and static buffer
            if self._decode_input_buffer is not None and self._decode_input_buffer.device == device:
                self._decode_input_buffer[0, 0] = next_token
                token_input = self._decode_input_buffer
            else:
                token_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            outputs = self.model(
                token_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Sample the next token
            next_token_id = sample_token(
                next_token_logits, params, request.output_token_ids, generator
            )
            next_token = next_token_id.item()
            request.output_token_ids.append(next_token)

            # Update KV cache tracking
            self.kv_cache_manager.extend_sequence(request.request_id, 1)

            # Stream token to callback
            stream_fn()

        return next_token, past_key_values

    def _finalize_generation(self, request: GenerationRequest):
        """Wrap up a completed generation: decode text, free memory, log stats."""
        # Decode the full output if not already set (e.g., by stop-string trimming)
        if not request.output_text:
            request.output_text = self.tokenizer.decode(
                request.output_token_ids, skip_special_tokens=True
            )

        if request.status != RequestStatus.CANCELLED:
            request.status = RequestStatus.COMPLETED
        request.finished_at = time.time()

        # Free KV cache blocks for this request
        self.kv_cache_manager.free_sequence(request.request_id)

        # Signal to the stream consumer that generation is done
        if request._stream_callback:
            request._stream_callback("", True)

        logger.info(
            "Request %s completed: %d tokens in %.2fs (%.1f tok/s)",
            request.request_id,
            request.generation_tokens,
            request.elapsed,
            request.tokens_per_second,
        )

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
