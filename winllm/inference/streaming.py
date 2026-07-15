"""Token emission to streaming callbacks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.types import GenerationRequest
    from .runtime import ModelRuntime

logger = logging.getLogger(__name__)


class StreamEmitter:
    """Sends newly generated tokens to a request's registered callback.

    There are two callback modes:
      - _token_callback: receives raw token IDs (preferred, faster).
      - _stream_callback: receives decoded text deltas (legacy fallback).
    """

    def __init__(self, runtime: "ModelRuntime"):
        self._runtime = runtime

    def emit(self, request: "GenerationRequest") -> None:
        """Send all not-yet-streamed generated tokens to the registered callback.

        Drains from ``_emit_cursor`` instead of sending only the last token:
        speculative decoding commits several tokens per step, and each one
        must reach the stream.
        """
        pending = request.output_token_ids[request._emit_cursor:]
        if not pending:
            return

        # Preferred path: raw token ID callback (used by the API server)
        if request._token_callback:
            for token_id in pending:
                request._token_callback(token_id, False)
            request._emit_cursor = len(request.output_token_ids)
            return

        # Legacy path: decoded text callback (used by the chat CLI)
        if not request._stream_callback:
            return

        # Decode only the new tokens to avoid O(n²) full-sequence re-decode.
        # Note: incremental decoding may differ slightly from full-sequence
        # decoding for tokenizers with context-dependent spacing (e.g.
        # SentencePiece), but this is acceptable for streaming — the final
        # output_text is always decoded from the full sequence.
        new_text = self._runtime.tokenizer.decode(pending, skip_special_tokens=True)
        request._emit_cursor = len(request.output_token_ids)
        if new_text:
            request._stream_callback(new_text, False)

    @staticmethod
    def emit_finished(request: "GenerationRequest") -> None:
        """Signal to the stream consumer that generation is done."""
        if request._stream_callback:
            request._stream_callback("", True)
