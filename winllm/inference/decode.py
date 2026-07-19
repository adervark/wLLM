"""Decode: generating one token per step for running requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..core.types import RequestStatus
from ..sampling import sample_token
from .eager import eager_decode_step

if TYPE_CHECKING:
    from ..core.types import GenerationRequest
    from .buffers import DecodeInputBuffer, PersistentBatchCache
    from .runtime import ModelRuntime
    from .streaming import StreamEmitter


class DecodeRunner:
    """Runs decode forward passes for requests that finished prefill."""

    def __init__(
        self,
        runtime: "ModelRuntime",
        emitter: "StreamEmitter",
        input_buffer: "DecodeInputBuffer",
        batch_cache: "PersistentBatchCache",
    ):
        self._runtime = runtime
        self._emitter = emitter
        self._input_buffer = input_buffer
        self._batch_cache = batch_cache

    def run(self, requests: list["GenerationRequest"], device: torch.device) -> None:
        if len(requests) == 1:
            # Single-request decode owns its own per-request cache; drop the
            # batched one so its tensors (and any views into it) can be freed.
            self._batch_cache.release()
            self.decode_single(requests[0], device, batch_size=1)
        else:
            self.decode_batch(requests, device)

    def decode_single(self, req: "GenerationRequest", device: torch.device, batch_size: int = 1) -> None:
        """Run one decode step for a request that's already being generated.

        Takes the last generated token, feeds it through the model with
        the KV cache, and samples the next token. Uses speculative decoding
        when available and there's only one request in the batch.
        """
        if req.is_cancelled:
            return

        # Speculative decoding: use the fast draft model to propose tokens,
        # then verify them with the target model in one pass.
        # Only works for single-request batches (no batched speculation yet),
        # and not for grammar-constrained requests (draft proposals aren't masked).
        speculative = self._runtime.speculative_engine
        if speculative and batch_size == 1 and req._grammar is None:
            tokens_before = len(req.output_token_ids)
            speculative.step(req)
            tokens_added = len(req.output_token_ids) - tokens_before
            if tokens_added > 0 and not self._commit_tokens(req, tokens_added):
                return
            self._emitter.emit(req)
            return

        next_logits, req._past_key_values = eager_decode_step(
            self._runtime.model, req.output_token_ids[-1],
            req._past_key_values, device, self._input_buffer,
        )

        next_token_id = sample_token(
            next_logits, req.sampling_params, req.output_token_ids, grammars=[req._grammar]
        )
        req.output_token_ids.append(next_token_id.item())

        if not self._commit_tokens(req, 1):
            return
        self._emitter.emit(req)

    def _commit_tokens(self, req: "GenerationRequest", count: int) -> bool:
        """Account newly generated tokens in the KV budget.

        When even prefix-cache eviction cannot make room, the request is
        failed so the scheduler frees its blocks — silently continuing would
        let the real cache grow past the budget until CUDA OOM.
        """
        if self._runtime.kv_cache_manager.extend_sequence(req.request_id, count):
            return True
        req.status = RequestStatus.FAILED
        req.error = "KV cache exhausted during decode"
        return False

    def decode_batch(self, decode_reqs: list["GenerationRequest"], device: torch.device) -> None:
        """Run a single forward pass for a batch of requests.

        This is the primary throughput optimization, moving from O(N) model
        passes to O(1) for a batch of size N. The batched KV cache persists
        across steps (see ``PersistentBatchCache``): it is only repacked when
        the resident set changes, and per-request caches are exposed as
        zero-copy views — so a stable batch copies no KV history per step.
        """
        # Filter out cancelled requests before wasting a forward pass on them
        decode_reqs = [r for r in decode_reqs if not r.is_cancelled]
        if not decode_reqs:
            self._batch_cache.release()
            return

        batch_size = len(decode_reqs)

        # Repack the batched cache only when the resident set changed;
        # survivors move via fused gathers, not per-request copies.
        self._batch_cache.refresh(decode_reqs, device)

        # 1. Prepare batched inputs
        input_ids = torch.tensor(
            [[req.output_token_ids[-1]] for req in decode_reqs],
            device=device,
        )

        # Current cache length per request (= prompt + output - 1).
        seq_lengths = [len(req.prompt_token_ids) + len(req.output_token_ids) - 1 for req in decode_reqs]
        padded_len = self._batch_cache.kv[0][0].shape[2]

        # 2. Attention mask + position ids. The cache is left-padded, so each
        #    request's valid span (its tokens + the new one) is right-aligned.
        #    Built as one vectorized comparison instead of a per-row loop.
        lengths_t = torch.tensor(seq_lengths, dtype=torch.long, device=device)
        positions = torch.arange(padded_len + 1, device=device)
        attn_mask = (positions >= (padded_len - lengths_t).unsqueeze(1)).long()

        position_ids = lengths_t.unsqueeze(-1)

        # 3. Forward pass, feeding the persistent cache straight back
        outputs = self._runtime.model(
            input_ids,
            past_key_values=self._batch_cache.kv,
            use_cache=True,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
        self._batch_cache.update(outputs.past_key_values)
        next_logits = outputs.logits[:, -1, :]

        next_token_ids = sample_token(
            next_logits,
            [req.sampling_params for req in decode_reqs],
            [req.output_token_ids for req in decode_reqs],
            grammars=[req._grammar for req in decode_reqs],
        )

        # 4. Commit sampled tokens and stream them
        for i, req in enumerate(decode_reqs):
            req.output_token_ids.append(next_token_ids[i].item())
            if self._commit_tokens(req, 1):
                self._emitter.emit(req)

        # 5. Expose each request's updated cache as a zero-copy view (no clone).
        self._batch_cache.expose_views(decode_reqs, [sl + 1 for sl in seq_lengths])
