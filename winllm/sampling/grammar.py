"""Grammar-constrained decoding (structured output) backed by xgrammar.

xgrammar compiles a JSON grammar / JSON schema into a pushdown automaton
and yields, per step, a packed bitmask of vocabulary tokens that keep the
output well-formed. The mask is applied by the native winllm_sampling
kernel when it's built (one launch), else unpacked with plain torch ops —
either way it works on CUDA without Triton (xgrammar's own GPU kernel
requires Triton, which is unavailable on Windows).

xgrammar is an optional dependency: ``pip install winllm[structured]``.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MISSING_MSG = (
    "Structured output requires the 'xgrammar' package. "
    "Install it with: pip install winllm[structured]"
)


def unpack_bitmask(bitmask: torch.Tensor, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Unpack xgrammar's little-endian int32 bitmask into a bool keep-mask.

    ``bitmask`` has shape (1, ceil(vocab/32)); bit ``i % 32`` of word
    ``i // 32`` says whether token ``i`` is allowed.
    """
    words = bitmask.to(device)
    bits = torch.arange(32, device=device, dtype=torch.int32)
    unpacked = (words.unsqueeze(-1) >> bits) & 1  # (1, words, 32)
    return unpacked.reshape(-1)[:vocab_size].bool()


class GrammarState:
    """Per-request grammar matcher state.

    Masks a logits row in place each step and lazily advances the matcher
    over tokens sampled since the last call, so callers never have to
    remember to "accept" tokens explicitly.
    """

    def __init__(self, xgr, matcher, vocab_size: int):
        self._xgr = xgr
        self._matcher = matcher
        self._vocab_size = vocab_size
        self._bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self._cursor = 0

    def _advance(self, generated_ids: list[int]) -> None:
        while self._cursor < len(generated_ids):
            token = generated_ids[self._cursor]
            if not self.is_terminated:
                accepted = self._matcher.accept_token(token)
                if not accepted:
                    # Shouldn't happen when sampling under the mask; log and
                    # stop constraining rather than corrupting matcher state.
                    logger.warning("Grammar matcher rejected token %d; releasing constraint", token)
                    self._matcher = None
                    return
            self._cursor += 1

    @property
    def is_terminated(self) -> bool:
        return self._matcher is None or self._matcher.is_terminated()

    def apply(self, logits_row: torch.Tensor, generated_ids: list[int]) -> None:
        """Mask ``logits_row`` (shape (vocab,)) so only grammar-legal tokens survive."""
        self._advance(generated_ids or [])
        if self.is_terminated:
            return

        self._matcher.fill_next_token_bitmask(self._bitmask)
        model_vocab = logits_row.shape[-1]
        mask_width = min(self._vocab_size, model_vocab)
        # One fused kernel when the native module is present; the torch
        # unpack chain otherwise (CPU tensors, CI, WINLLM_PURE_PYTHON).
        from .native import fused_sampler

        if not fused_sampler.apply_bitmask(logits_row[:mask_width], self._bitmask):
            keep = unpack_bitmask(self._bitmask, mask_width, logits_row.device)
            logits_row[: keep.shape[0]].masked_fill_(~keep, float("-inf"))
        if model_vocab > self._vocab_size:
            # Padded model vocab beyond the tokenizer's real tokens is never legal
            logits_row[self._vocab_size:] = float("-inf")


class GrammarBackend:
    """Compiles grammars for one tokenizer; hands out per-request states.

    The underlying ``GrammarCompiler`` caches compiled grammars, so repeated
    requests with the same schema reuse the compiled automaton.
    """

    def __init__(self, tokenizer):
        try:
            import xgrammar as xgr
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError(_MISSING_MSG) from e

        self._xgr = xgr
        self._info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        self._compiler = xgr.GrammarCompiler(self._info)

    def state_for(self, response_format: dict) -> Optional["GrammarState"]:
        """Build a fresh matcher for an OpenAI-style ``response_format`` dict.

        Supports {"type": "json_object"} and
        {"type": "json_schema", "json_schema": {"schema": {...}}}.
        Returns None for {"type": "text"}.
        """
        rf_type = response_format.get("type")
        if rf_type in (None, "text"):
            return None
        if rf_type == "json_object":
            compiled = self._compiler.compile_builtin_json_grammar()
        elif rf_type == "json_schema":
            wrapper = response_format.get("json_schema") or {}
            schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
            if schema is None:
                raise ValueError("response_format.json_schema.schema is required")
            compiled = self._compiler.compile_json_schema(json.dumps(schema))
        else:
            raise ValueError(f"Unsupported response_format type: {rf_type!r}")

        matcher = self._xgr.GrammarMatcher(compiled)
        return GrammarState(self._xgr, matcher, self._info.vocab_size)
