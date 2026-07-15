"""CUDA graph capture for the single-request decode hot path.

Per-token decode in a Python/HF loop is dominated by CPU kernel-launch
overhead. A CUDA graph records the whole decode forward once and replays
it with a single launch, eliminating that overhead without custom kernels
(and without torch.compile/Triton, which is unavailable on Windows).

Requirements this class manages:
  - Static shapes/addresses: a preallocated ``StaticCache`` plus fixed
    input/position buffers that are overwritten in place each step.
  - Warmup on a side stream before capture (per torch docs).
  - Graph safety varies by architecture, so ``validate()`` replays a short
    greedy generation against eager execution and disables the graph path
    on any mismatch or capture failure — zero-day models silently fall
    back to the normal loop instead of crashing or corrupting output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


class CUDAGraphDecoder:
    """Graph-accelerated decode for one request at a time (batch size 1).

    Two captured graphs share one StaticCache:
      - decode: the (1, 1) per-token forward (``decode()``).
      - verify: a (1, VERIFY_BUCKET + 1) forward (``verify()``) used for
        speculative drafts — one replay scores a whole draft sequence.
        Rollback after rejection is a position rewind (``advance()``):
        rejected positions keep junk KV, but sequential writes overwrite
        them before any causal query can attend that far, so no crop is
        needed and the rewind is lossless by construction.
    """

    VERIFY_BUCKET = 8  # drafts per verify replay (fixed shape for capture)

    def __init__(self, model, model_config: "ModelConfig", device: torch.device):
        self.ok = False
        self.verify_ok = False
        self._model = model
        self._device = device
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._verify_graph: Optional[torch.cuda.CUDAGraph] = None
        self._cache = None
        self._logits_buf: Optional[torch.Tensor] = None
        self._verify_logits: Optional[torch.Tensor] = None
        self._max_len = model_config.max_model_len
        self._pos = 0

        # Hybrid layouts (conv/ssm/recurrent layers mixed with attention) need
        # architecture-specific caches; StaticCache constructs for them but
        # fails mid-forward, so reject them before touching the GPU.
        layer_types = getattr(getattr(model, "config", None), "layer_types", None) or []
        non_attention = {t for t in layer_types if "attention" not in t}
        if non_attention:
            logger.info(
                "CUDA graphs disabled: hybrid architecture with %s layers "
                "is not StaticCache-compatible", sorted(non_attention),
            )
            return

        if device.type != "cuda":
            logger.info("CUDA graphs disabled: model is not on a CUDA device")
            return

        try:
            from transformers import StaticCache

            self._cache = StaticCache(
                config=model.config,
                max_batch_size=1,
                max_cache_len=model_config.max_model_len,
                device=device,
                dtype=model_config.torch_dtype,
            )
        except Exception as e:
            logger.warning("CUDA graphs disabled: StaticCache unavailable for this model (%s)", e)
            return

        self._input_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        # Verify-path static buffers (fixed VERIFY_BUCKET + 1 shape)
        width = self.VERIFY_BUCKET + 1
        self._verify_input = torch.zeros(1, width, dtype=torch.long, device=device)
        self._verify_pos = torch.zeros(width, dtype=torch.long, device=device)
        self._verify_arange = torch.arange(width, device=device)

        # NOTE (measured 2026-07-06, do not "optimize" this): staging these
        # inputs in pinned host memory and copy_(non_blocking=True)-ing them
        # in looked like it should remove per-step H2D syncs, but it
        # collapsed throughput ~5x (93 -> 19 tok/s) on Windows/WDDM —
        # async memcpys into the graph's private memory pool serialize far
        # worse than the scalar fill kernels the plain assignments below
        # compile to. See documentation/PERFORMANCE_NOTES.md.
        self.ok = True

    # ------------------------------------------------------------------
    # Generation interface (mirrors the eager prefill/decode contract)
    # ------------------------------------------------------------------

    def prefill(self, prompt_token_ids: list[int]) -> torch.Tensor:
        """Run an eager prefill into the static cache; returns last-token logits."""
        self._cache.reset()
        length = len(prompt_token_ids)
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=self._device)
        cache_position = torch.arange(length, device=self._device)
        outputs = self._model(
            input_ids,
            past_key_values=self._cache,
            use_cache=True,
            cache_position=cache_position,
        )
        self._pos = length
        return outputs.logits[:, -1, :]

    def decode(self, token_id: int) -> torch.Tensor:
        """One graph-replayed decode step; returns next-token logits."""
        self._input_buf[0, 0] = token_id
        self._pos_buf[0] = self._pos
        self._pos += 1

        if self._graph is None:
            self._graph, logits = self._capture(self._forward)
            self._logits_buf = logits[:, -1, :]
        # Capture only *records* the kernels; every step (including the first
        # after capture) must replay to actually produce logits.
        self._graph.replay()
        return self._logits_buf

    def verify(self, tokens: list[int]) -> Optional[torch.Tensor]:
        """One graph-replayed verify forward over [last_committed] + drafts.

        Returns logits of shape (len(tokens), vocab): row ``i`` predicts the
        token after consuming ``tokens[:i + 1]``. Does NOT advance the
        position — the caller calls ``advance(1 + accepted)`` after the
        accept loop. Returns None when the request is near the context limit
        or the verify graph is unavailable (caller falls back to decode()).
        """
        width = self.VERIFY_BUCKET + 1
        if len(tokens) > width or self._pos + width > self._max_len:
            return None

        self._verify_input.zero_()
        self._verify_input[0, : len(tokens)] = torch.tensor(
            tokens, dtype=torch.long, device=self._device
        )
        self._verify_pos.copy_(self._verify_arange + self._pos)

        if self._verify_graph is None:
            self._verify_graph, self._verify_logits = self._capture(self._verify_forward)
        self._verify_graph.replay()
        return self._verify_logits[0, : len(tokens), :]

    def advance(self, n: int) -> None:
        """Commit ``n`` cache positions after a verify step (1 + accepted)."""
        self._pos += n

    def _forward(self):
        return self._model(
            self._input_buf,
            past_key_values=self._cache,
            use_cache=True,
            cache_position=self._pos_buf,
        )

    def _verify_forward(self):
        return self._model(
            self._verify_input,
            past_key_values=self._cache,
            use_cache=True,
            cache_position=self._verify_pos,
        )

    def _capture(self, forward_fn):
        """Warm up on a side stream, then record ``forward_fn`` in a graph.

        Returns (graph, logits): every replay rewrites the returned logits
        tensor in place.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                out = forward_fn()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward_fn()
        return graph, out.logits

    # ------------------------------------------------------------------
    # Self-check
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def validate(self, probe_token_ids: list[int], steps: int = 8) -> bool:
        """Teacher-forced comparison of graph logits against eager logits.

        Exact greedy-token identity is the wrong check here: fp16/quantized
        kernels take different code paths under StaticCache, so argmax flips
        at near-ties are benign numeric noise and would reject perfectly good
        architectures. Instead, the same eager-chosen tokens are fed to both
        paths and the logits must mutually agree on what is plausible
        (each path's argmax within the other's top-k) at every step —
        capture corruption produces garbage logits that fail immediately.

        Any exception or divergence releases the decoder, so architectures
        with graph-unsafe forwards degrade to the eager loop.
        """
        try:
            # Eager reference (DynamicCache path, like the normal loop) picks
            # the forced token sequence and records its logits at each step.
            from .eager import eager_decode_step

            ids = torch.tensor([probe_token_ids], dtype=torch.long, device=self._device)
            out = self._model(ids, use_cache=True)
            past = out.past_key_values
            eager_logits = [out.logits[:, -1, :].float()]
            forced_tokens = []
            for _ in range(steps):
                tok = int(torch.argmax(eager_logits[-1], dim=-1).item())
                forced_tokens.append(tok)
                logits, past = eager_decode_step(self._model, tok, past, self._device)
                eager_logits.append(logits.float())

            # Graph path replays the identical token sequence. Clone each
            # step: replay overwrites the shared logits buffer in place.
            graph_logits = [self.prefill(probe_token_ids).float()]
            for tok in forced_tokens:
                graph_logits.append(self.decode(tok).float().clone())

            for step, (ref, test) in enumerate(zip(eager_logits, graph_logits)):
                if not self._logits_agree(ref, test):
                    logger.warning(
                        "CUDA graphs disabled: graph logits diverged from "
                        "eager at probe step %d", step,
                    )
                    self.release()
                    return False

            logger.info(
                "CUDA graph decode validated (%d teacher-forced steps agree with eager)",
                steps,
            )
            # Verify path is validated separately: its failure only disables
            # speculation on the graph path, not graph decode itself.
            self.verify_ok = self._validate_verify(
                probe_token_ids, forced_tokens, eager_logits
            )
            return True
        except Exception as e:
            logger.warning("CUDA graphs disabled: capture/validation failed (%s)", e)
            self.release()
            return False

    def _validate_verify(self, probe_token_ids, forced_tokens, eager_logits) -> bool:
        """Teacher-forced check of the captured verify forward.

        Scores the same forced tokens in one verify() replay and requires
        per-position agreement with the eager reference logits.
        """
        try:
            self.prefill(probe_token_ids)  # reset the cache to the probe context
            rows = self.verify(forced_tokens[: self.VERIFY_BUCKET + 1])
            if rows is None:
                return False
            for i in range(rows.shape[0]):
                if not self._logits_agree(eager_logits[i + 1], rows[i:i + 1, :].float()):
                    logger.info(
                        "CUDA graph verify disabled: logits diverged from eager "
                        "at position %d (graph decode stays enabled)", i,
                    )
                    return False
            logger.info(
                "CUDA graph verify validated (%d positions agree with eager)",
                rows.shape[0],
            )
            return True
        except Exception as e:
            logger.info("CUDA graph verify disabled: %s (graph decode stays enabled)", e)
            self._verify_graph = None
            return False

    @staticmethod
    def _logits_agree(ref: torch.Tensor, test: torch.Tensor, top_k: int = 5) -> bool:
        """True when each distribution's argmax is plausible under the other."""
        if not torch.isfinite(test).all():
            return False
        k = min(top_k, ref.shape[-1])
        ref_top = torch.topk(ref, k, dim=-1).indices
        test_top = torch.topk(test, k, dim=-1).indices
        return bool(
            (test_top == torch.argmax(ref, dim=-1, keepdim=True)).any()
            and (ref_top == torch.argmax(test, dim=-1, keepdim=True)).any()
        )

    def release(self) -> None:
        """Drop the graphs and static cache, marking the decoder unusable."""
        self._graph = None
        self._verify_graph = None
        self._logits_buf = None
        self._verify_logits = None
        self._cache = None
        self.ok = False
        self.verify_ok = False
