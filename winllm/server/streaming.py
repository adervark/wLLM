"""SSE token streaming for chat and text completions."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.types import GenerationRequest
    from ..inference.engine import InferenceEngine
    from ..scheduling.scheduler import Scheduler

logger = logging.getLogger(__name__)


class StopStringGate:
    """Withholds trailing text so stop strings never reach the client.

    The scheduler detects stop strings only after a step commits (and
    streams) its tokens, so without a gate the client sees the stop string
    plus any overshoot while the non-streamed ``output_text`` is trimmed.
    Holding back ``len(longest stop) - 1`` characters guarantees every
    occurrence is fully inside some scan window before its text is emitted.
    """

    def __init__(self, stop_strings: list[str]):
        self._stops = stop_strings or []
        self._holdback = max((len(s) for s in self._stops), default=0) - 1
        self._pending = ""
        self.triggered = False

    def _stop_index(self) -> int:
        hits = [i for i in (self._pending.find(s) for s in self._stops) if i != -1]
        return min(hits) if hits else -1

    def push(self, text: str) -> str:
        """Feed newly decoded text; return the part that is safe to emit."""
        if not self._stops:
            return text
        if self.triggered:
            return ""
        self._pending += text
        idx = self._stop_index()
        if idx != -1:
            self.triggered = True
            out, self._pending = self._pending[:idx], ""
            return out
        safe = len(self._pending) - self._holdback
        if safe <= 0:
            return ""
        out, self._pending = self._pending[:safe], self._pending[safe:]
        return out

    def flush(self) -> str:
        """Return whatever held-back text may still be emitted at stream end."""
        if not self._stops or self.triggered:
            return ""
        idx = self._stop_index()
        out = self._pending[:idx] if idx != -1 else self._pending
        self._pending = ""
        return out


async def stream_completion(
    gen_request: "GenerationRequest",
    model_name: str,
    response_type: str,
    engine: "InferenceEngine",
    scheduler: "Scheduler",
    stream_timeout: float,
):
    """Unified SSE stream for both chat and text completions.

    Receives token IDs from the inference loop and performs decoding in
    this async task to keep the GPU loop fast.
    """
    is_chat = response_type == "chat"
    id_prefix = "chatcmpl" if is_chat else "cmpl"
    object_type = "chat.completion.chunk" if is_chat else "text_completion"
    request_id = f"{id_prefix}-{gen_request.request_id}"

    loop = asyncio.get_running_loop()
    token_id_queue: asyncio.Queue[tuple[int, bool] | BaseException] = asyncio.Queue()
    gate = StopStringGate(gen_request.sampling_params.stop)

    def _text_chunk(text: str) -> dict:
        choice = {"index": 0, "finish_reason": None}
        if is_chat:
            choice["delta"] = {"content": text}
        else:
            choice["text"] = text
        return {
            "id": request_id, "object": object_type, "created": int(time.time()),
            "model": model_name, "choices": [choice],
        }

    def token_callback(token_id: int, finished: bool):
        loop.call_soon_threadsafe(token_id_queue.put_nowait, (token_id, finished))

    gen_request._token_callback = token_callback

    # Start generation via scheduler to ensure batching/admission control
    gen_task = loop.create_task(scheduler.submit_streaming(gen_request))

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    token_id_queue.get(), timeout=stream_timeout
                )
            except asyncio.TimeoutError:
                gen_request.cancel()
                error_chunk = {
                    "id": request_id, "object": object_type, "created": int(time.time()),
                    "model": model_name,
                    "error": {"message": f"Token generation timed out after {stream_timeout}s", "type": "timeout"},
                }
                yield {"data": json.dumps(error_chunk)}
                yield {"data": "[DONE]"}
                break

            if isinstance(item, BaseException):
                error_chunk = {
                    "id": request_id, "object": object_type, "created": int(time.time()),
                    "model": model_name,
                    "error": {"message": str(item), "type": type(item).__name__},
                }
                yield {"data": json.dumps(error_chunk)}
                yield {"data": "[DONE]"}
                break

            token_id, finished = item
            if finished:
                # Emit any held-back text the gate cleared before finishing
                tail = gate.flush()
                if tail:
                    yield {"data": json.dumps(_text_chunk(tail))}
                choice = {"index": 0, "finish_reason": gen_request.finish_reason or "stop"}
                choice["delta" if is_chat else "text"] = {} if is_chat else ""
                chunk = {
                    "id": request_id, "object": object_type, "created": int(time.time()),
                    "model": model_name, "choices": [choice],
                }
                yield {"data": json.dumps(chunk)}
                yield {"data": "[DONE]"}
                break
            else:
                # Decode the new token inline -- tokenizer.decode for a single token
                # is a fast CPU-only operation (~1-5us) that doesn't justify the
                # ~50-100us thread pool dispatch overhead of run_in_executor.
                new_text = engine.tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                emit_text = gate.push(new_text) if new_text else ""
                if emit_text:
                    yield {"data": json.dumps(_text_chunk(emit_text))}
    finally:
        await gen_task
