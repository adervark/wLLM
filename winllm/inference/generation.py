"""Blocking single-request generation.

Orchestrates the full lifecycle of one request when used without the
scheduler (CLI chat, benchmarks): validate -> prefill -> decode -> finalize.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

import torch

from ..core.types import GenerationRequest, RequestStatus, normalize_eos_ids
from ..sampling import sample_token
from .eager import eager_decode_step

if TYPE_CHECKING:
    from ..config import ModelConfig, SamplingParams
    from .buffers import DecodeInputBuffer
    from .runtime import ModelRuntime
    from .streaming import StreamEmitter

logger = logging.getLogger(__name__)


class _VerifyThrottle:
    """Exponential backoff for graph-verified speculation.

    A verify replay is wasted work whenever every draft gets rejected —
    e.g. templated output ("Item 1: ... Item 2: ...") where the repeated
    surface pattern always continues with a *new* value, so the suffix
    draft is deterministically wrong. After a fully-rejected verify the
    next 2^k opportunities are skipped (capped), driving speculation
    overhead toward zero on such text, while a single acceptance snaps
    straight back to speculating every step.
    """

    def __init__(self, max_backoff: int = 32):
        self._max = max_backoff
        self._backoff = 1
        self._skip = 0

    def should_try(self) -> bool:
        if self._skip > 0:
            self._skip -= 1
            return False
        return True

    def record(self, accepted: int) -> None:
        if accepted > 0:
            self._backoff = 1
        else:
            self._skip = self._backoff
            self._backoff = min(self._backoff * 2, self._max)


class BlockingGenerator:
    """Runs one request end-to-end on the current thread."""

    def __init__(
        self,
        runtime: "ModelRuntime",
        model_config: "ModelConfig",
        emitter: "StreamEmitter",
        input_buffer: "DecodeInputBuffer",
    ):
        self._runtime = runtime
        self._model_config = model_config
        self._emitter = emitter
        self._input_buffer = input_buffer
        # Serializes access to the (single-slot) CUDA graph decoder
        self._graph_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    def validate_prompt(self, request: GenerationRequest, prompt_len: int) -> bool:
        """Check that the prompt fits within the model's context window.

        Returns True if valid, False if the request was marked as failed.
        """
        if prompt_len >= self._model_config.max_model_len:
            request.status = RequestStatus.FAILED
            request.error = f"Prompt too long: {prompt_len} tokens (max {self._model_config.max_model_len})"
            return False
        return True

    def allocate_kv_cache(self, request: GenerationRequest, prompt_len: int, max_new_tokens: int) -> bool:
        """Reserve KV cache blocks for this request.

        Returns True if allocation succeeded, False if insufficient memory.
        """
        kv = self._runtime.kv_cache_manager
        total_tokens_needed = prompt_len + max_new_tokens
        if not kv.can_allocate(total_tokens_needed):
            request.status = RequestStatus.FAILED
            request.error = "Insufficient KV cache memory"
            return False
        kv.allocate_sequence(request.request_id, prompt_len)
        return True

    @staticmethod
    def make_generator(params: "SamplingParams", device: torch.device) -> Optional[torch.Generator]:
        """Create a random number generator for reproducible sampling."""
        if params.seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(params.seed)
            return gen
        return None

    def get_stop_conditions(self, params: "SamplingParams") -> tuple[set[int], list[str]]:
        """Build the set of stop token IDs and stop strings."""
        stop_token_ids = normalize_eos_ids(self._runtime.tokenizer.eos_token_id)
        stop_strings = params.stop if params.stop else []
        return stop_token_ids, stop_strings

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationRequest:
        """Internal generation implementation.

        Orchestrates the full lifecycle: validate -> prefill -> decode -> finalize.
        """
        params = request.sampling_params
        device = self._runtime.resolve_device()

        # --- Step 1: Tokenize the prompt ---
        if not request.prompt_token_ids:
            request.prompt_token_ids = self._runtime.tokenize(request.prompt)
        prompt_len = len(request.prompt_token_ids)

        # --- Step 2: Validate that the prompt fits ---
        if not self.validate_prompt(request, prompt_len):
            return request

        max_new_tokens = min(
            params.max_tokens,
            self._model_config.max_model_len - prompt_len,
        )

        # --- Step 3: Allocate KV cache memory ---
        if not self.allocate_kv_cache(request, prompt_len, max_new_tokens):
            return request

        generator = self.make_generator(params, device)
        stop_token_ids, stop_strings = self.get_stop_conditions(params)

        # Set up incremental streaming. Each newly generated token is decoded
        # on its own (O(1) per token) rather than re-decoding the full sequence
        # every step (which was O(n^2) over a generation). Single-token decoding
        # may differ slightly from full-sequence decoding for tokenizers with
        # context-dependent spacing (e.g. SentencePiece), but the final
        # output_text is always decoded from the full sequence in _finalize.
        tokenizer = self._runtime.tokenizer
        streamed = 0  # number of output tokens already streamed

        def _stream_token():
            nonlocal streamed
            if not request._stream_callback:
                return
            while streamed < len(request.output_token_ids):
                piece = tokenizer.decode(
                    [request.output_token_ids[streamed]], skip_special_tokens=True
                )
                if piece:
                    request._stream_callback(piece, False)
                streamed += 1

        # --- Step 4: Prefill (process entire prompt at once) ---
        # Single-flight claim on the CUDA-graph decoder: concurrent callers
        # (it holds one static cache) fall back to the eager path.
        graph = None
        runtime_graph = self._runtime.graph_decoder
        if runtime_graph is not None and runtime_graph.ok and self._graph_lock.acquire(blocking=False):
            graph = runtime_graph

        spec = None
        try:
            if graph is not None:
                next_token_logits = graph.prefill(request.prompt_token_ids)
                forward_step = graph.decode
                spec_engine = self._runtime.speculative_engine
                if (
                    graph.verify_ok
                    and request._grammar is None
                    and spec_engine is not None
                    and hasattr(spec_engine, "suffix_cache")
                ):
                    # Compose the two accelerators: suffix drafts scored by
                    # one graph-replayed verify forward per step.
                    spec = (graph, spec_engine.suffix_cache, spec_engine)
            else:
                input_ids = torch.tensor(
                    [request.prompt_token_ids], dtype=torch.long, device=device
                )
                outputs = self._runtime.model(input_ids, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                def forward_step(token: int) -> torch.Tensor:
                    nonlocal past_key_values
                    logits, past_key_values = eager_decode_step(
                        self._runtime.model, token, past_key_values,
                        device, self._input_buffer,
                    )
                    return logits

            # Sample the first token
            next_token_id = sample_token(
                next_token_logits, params, request.output_token_ids, generator,
                grammars=[request._grammar],
            )
            next_token = next_token_id.item()
            request.output_token_ids.append(next_token)

            _stream_token()

            # --- Step 5: Decode loop ---
            self._run_decode_loop(
                request, params, generator,
                next_token, forward_step,
                max_new_tokens, stop_token_ids, stop_strings,
                _stream_token, spec=spec,
            )
        finally:
            if graph is not None:
                self._graph_lock.release()

        # --- Step 6: Finalize ---
        self._finalize(request)

        return request

    def _run_decode_loop(
        self,
        request: GenerationRequest,
        params: "SamplingParams",
        generator: Optional[torch.Generator],
        next_token: int,
        forward_step: Callable[[int], torch.Tensor],
        max_new_tokens: int,
        stop_token_ids: set[int],
        stop_strings: list[str],
        stream_fn: Callable,
        spec: Optional[tuple] = None,
    ) -> int:
        """Run the autoregressive decode loop.

        ``forward_step`` runs one decode forward for a token and returns the
        next-token logits — either the eager HF path or a CUDA-graph replay.
        When ``spec`` is provided (graph decoder with a validated verify
        path, suffix cache, telemetry sink), an iteration may commit several
        tokens from one graph-verified draft. Returns the last generated
        token.
        """
        tokenizer = self._runtime.tokenizer
        kv = self._runtime.kv_cache_manager

        # Maintain running output text to avoid O(n²) re-decoding for
        # stop-string checks. The cursor covers multi-token speculative
        # commits; it starts at the current last token so the first sampled
        # token is folded in like every later one.
        running_text = ""
        stop_cursor = len(request.output_token_ids) - 1

        # Assume we run to the token budget; break points below override this.
        request.finish_reason = "length"

        # Backs off speculation while drafts keep getting fully rejected
        throttle = _VerifyThrottle()

        while True:
            # Check if we hit an end-of-sequence token
            if next_token in stop_token_ids:
                request.finish_reason = "stop"
                break

            # Check if the request was cancelled externally
            if request.is_cancelled:
                request.status = RequestStatus.CANCELLED
                request.finish_reason = "cancelled"
                break

            # Fold all not-yet-checked committed tokens into the running text
            if stop_strings:
                while stop_cursor < len(request.output_token_ids):
                    running_text += tokenizer.decode(
                        [request.output_token_ids[stop_cursor]], skip_special_tokens=True
                    )
                    stop_cursor += 1
                if any(s in running_text for s in stop_strings):
                    for s in stop_strings:
                        idx = running_text.find(s)
                        if idx != -1:
                            request.output_text = running_text[:idx]
                            break
                    request.finish_reason = "stop"
                    break

            if len(request.output_token_ids) >= max_new_tokens:
                break  # finish_reason stays "length"

            # Commit one token (plain) or several (graph-verified draft)
            new_count = 0
            if spec is not None and throttle.should_try():
                new_count, accepted = self._speculative_commit(
                    request, params, generator, next_token,
                    max_new_tokens, stop_token_ids, spec,
                )
                if new_count:
                    throttle.record(accepted)
            if new_count == 0:
                next_token_logits = forward_step(next_token)
                next_token_id = sample_token(
                    next_token_logits, params, request.output_token_ids, generator,
                    grammars=[request._grammar],
                )
                request.output_token_ids.append(next_token_id.item())
                new_count = 1

            next_token = request.output_token_ids[-1]

            # Update KV cache tracking
            kv.extend_sequence(request.request_id, new_count)

            # Stream tokens to callback
            stream_fn()

        return next_token

    def _speculative_commit(
        self,
        request: GenerationRequest,
        params: "SamplingParams",
        generator: Optional[torch.Generator],
        next_token: int,
        max_new_tokens: int,
        stop_token_ids: set[int],
        spec: tuple,
    ) -> tuple[int, int]:
        """Try one graph-verified speculative step.

        Returns ``(committed, accepted)``: tokens appended to the output and
        drafts the target agreed with. ``(0, 0)`` means "no usable draft" —
        the caller runs the plain decode step (and doesn't count it against
        the verify throttle, since no verify ran). Mirrors
        SuffixSpeculativeEngine's accept loop: committed tokens are always
        the target's own samples, so speculation never changes the output,
        only how many tokens one forward pass commits. Rollback is a
        position rewind on the static cache (``graph.advance``), which is
        lossless by construction.
        """
        graph, suffix_cache, telemetry = spec
        suffix_cache.sync(
            request.request_id,
            request.prompt_token_ids + request.output_token_ids,
        )
        drafts = suffix_cache.propose(request.request_id)
        budget = max_new_tokens - len(request.output_token_ids) - 1
        drafts = drafts[: min(graph.VERIFY_BUCKET, max(0, budget))]
        if not drafts:
            return 0, 0
        rows = graph.verify([next_token] + drafts)
        if rows is None:  # too close to the context limit for a full bucket
            return 0, 0

        telemetry.drafted_tokens += len(drafts)
        committed = 0
        accepted = 0

        # Without repetition penalty, each row's distribution is independent
        # of which drafts get accepted, so all rows are sampled in one
        # batched call — one device sync (.tolist()) per verify step instead
        # of one per draft. Rows past the first rejection are simply
        # discarded, which cannot change the committed output — unless the
        # request is seeded: a seeded generator must consume exactly one
        # draw per committed token, like plain decode, or the extra draws
        # for discarded rows desync the RNG stream and break speculation-off
        # reproducibility. Greedy (temperature 0) never touches the RNG, so
        # it batches even when seeded. (Grammar requests never reach here:
        # speculation is disabled for them at setup.)
        if params.repetition_penalty == 1.0 and (
            generator is None or params.temperature == 0
        ):
            toks = sample_token(rows, params, None, generator).tolist()
            for i in range(len(drafts)):
                tok = toks[i]
                request.output_token_ids.append(tok)
                committed += 1
                if tok in stop_token_ids or tok != drafts[i]:
                    break
                accepted += 1
                telemetry.accepted_tokens += 1
            else:
                # Every draft accepted — the final row is a free bonus prediction
                request.output_token_ids.append(toks[-1])
                committed += 1
        else:
            # Penalty state depends on tokens accepted so far, and a seeded
            # generator's draws must track committed tokens one-to-one: rows
            # are sampled sequentially against the growing output.
            for i in range(len(drafts)):
                tok = sample_token(
                    rows[i:i + 1, :], params, request.output_token_ids, generator
                ).item()
                request.output_token_ids.append(tok)
                committed += 1
                if tok in stop_token_ids or tok != drafts[i]:
                    break
                accepted += 1
                telemetry.accepted_tokens += 1
            else:
                bonus = sample_token(
                    rows[-1:, :], params, request.output_token_ids, generator
                ).item()
                request.output_token_ids.append(bonus)
                committed += 1

        graph.advance(1 + accepted)
        return committed, accepted

    def _finalize(self, request: GenerationRequest) -> None:
        """Wrap up a completed generation: decode text, free memory, log stats."""
        # Decode the full output if not already set (e.g., by stop-string trimming)
        if not request.output_text:
            request.output_text = self._runtime.decode_tokens(request.output_token_ids)

        if request.status != RequestStatus.CANCELLED:
            request.status = RequestStatus.COMPLETED
        request.finished_at = time.time()

        # Free KV cache blocks for this request
        self._runtime.kv_cache_manager.free_sequence(request.request_id)

        # Signal to the stream consumer that generation is done
        self._emitter.emit_finished(request)

        logger.info(
            "Request %s completed: %d tokens in %.2fs (%.1f tok/s)",
            request.request_id,
            request.generation_tokens,
            request.elapsed,
            request.tokens_per_second,
        )
