"""Prefill: processing the prompt to build the initial KV cache.

Handles three shapes of prefill:
  - single request, full prompt in one pass
  - single request, chunked (long prompts processed incrementally)
  - batched (multiple new requests left-padded into one forward pass)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from ..core.types import RequestStatus
from ..sampling import sample_token

if TYPE_CHECKING:
    from ..core.types import GenerationRequest
    from .runtime import ModelRuntime
    from .streaming import StreamEmitter


class PrefillRunner:
    """Runs prefill forward passes for new requests."""

    def __init__(self, runtime: "ModelRuntime", emitter: "StreamEmitter"):
        self._runtime = runtime
        self._emitter = emitter

    def run(
        self,
        requests: list["GenerationRequest"],
        device: torch.device,
        chunked_prefill_enabled: bool,
        max_num_batched_tokens: int,
    ) -> None:
        """Prefill a group of new requests, batching when profitable."""
        # Prefix-cache hits or chunked mode need per-request handling
        has_prefix_hits = any(r._prefix_past_key_values is not None for r in requests)
        if len(requests) == 1 or chunked_prefill_enabled or has_prefix_hits:
            for req in requests:
                self.prefill_single(req, device, chunked_prefill_enabled, max_num_batched_tokens)
        else:
            # Multiple non-chunked prefills without cache hits: batch into one forward pass
            self.prefill_batch(requests, device)

    def prefill_single(
        self,
        req: "GenerationRequest",
        device: torch.device,
        chunked_prefill: bool,
        max_tokens: int,
    ) -> None:
        """Run the prefill forward pass for a single new request.

        Processes the prompt in one pass, or in chunks if chunked prefill is
        enabled. Generates the first output token only after the final chunk
        completes.
        """
        if not req.prompt_token_ids:
            req.prompt_token_ids = self._runtime.tokenize(req.prompt)

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

        outputs = self._runtime.model(
            input_ids,
            past_key_values=req._past_key_values,
            use_cache=True,
        )

        req._past_key_values = outputs.past_key_values
        req._prefill_cursor = end_idx

        # Only sample if we've processed the entire prompt
        if is_final_chunk:
            next_logits = outputs.logits[:, -1, :]
            next_token_id = sample_token(
                next_logits, req.sampling_params, req.output_token_ids, grammars=[req._grammar]
            )
            req.output_token_ids.append(next_token_id.item())

            prompt_text = self._runtime.decode_tokens(req.prompt_token_ids)
            req._stream_text_cursor = len(prompt_text)
            self._emitter.emit(req)

    def prefill_batch(self, reqs: list["GenerationRequest"], device: torch.device) -> None:
        """Batch multiple prefill requests into a single forward pass.

        Uses left-padding to align variable-length prompts so they can be
        processed together. This is 2-5x faster than sequential prefill when
        multiple requests arrive simultaneously (common under API load).
        """
        now = time.time()
        tokenizer = self._runtime.tokenizer
        pad_token_id = tokenizer.pad_token_id or 0

        # Tokenize any un-tokenized requests
        for req in reqs:
            if not req.prompt_token_ids:
                req.prompt_token_ids = self._runtime.tokenize(req.prompt)
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

        outputs = self._runtime.model(
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

            # Sample first output token from this request's logits.
            # The last non-padding position for every request is the final
            # position, since inputs are left-padded.
            next_logits = outputs.logits[i:i+1, -1, :]
            next_token_id = sample_token(
                next_logits, req.sampling_params, req.output_token_ids, grammars=[req._grammar]
            )
            req.output_token_ids.append(next_token_id.item())

            prompt_text = self._runtime.decode_tokens(req.prompt_token_ids)
            req._stream_text_cursor = len(prompt_text)
            self._emitter.emit(req)
