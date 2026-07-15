"""Terminal presentation helpers for streamed model output.

Reasoning models (DeepSeek-R1, QwQ, ...) interleave their chain-of-thought
inside ``<think>...</think>`` tags with the actual answer. When that raw
stream is printed verbatim the reasoning and the answer blur together. This
module segregates the two *as tokens arrive*, rendering thinking under a
muted rule and the answer as live markdown.

The work is presentation-only and terminal-specific, which is why it lives in
``cli`` rather than ``inference/streaming.py`` (that layer stays transport
agnostic and never emits formatting).
"""

from __future__ import annotations

from typing import Callable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class ThinkTagParser:
    """Splits streamed text deltas into (kind, text) segments.

    ``kind`` is ``"thinking"`` for text inside ``<think>...</think>`` and
    ``"answer"`` otherwise. Presentation-free: the callback decides how to
    render each segment. Tags are detected even when split across delta
    boundaries by holding back any trailing substring that could be the
    start of a tag.

    The blank gap models often leave between ``</think>`` and the answer is
    dropped so the answer starts cleanly.
    """

    def __init__(self, on_segment: Callable[[str, str], None]):
        self._on_segment = on_segment
        self._in_thinking = False
        self._buf = ""
        # Whether the answer gap after thinking still needs trimming.
        self._answer_pending = False

    def feed(self, text: str) -> None:
        """Process a streamed text delta, emitting any complete segments."""
        if not text:
            return
        self._buf += text
        while self._buf:
            tag = THINK_CLOSE if self._in_thinking else THINK_OPEN
            idx = self._buf.find(tag)
            if idx != -1:
                self._emit(self._buf[:idx])
                self._buf = self._buf[idx + len(tag):]
                if self._in_thinking:
                    self._answer_pending = True
                self._in_thinking = not self._in_thinking
                continue

            # No complete tag: emit everything except a possible partial tag
            # tail, which we hold back until the next delta completes it.
            hold = self._partial_tail_len(self._buf, tag)
            split = len(self._buf) - hold
            self._emit(self._buf[:split])
            self._buf = self._buf[split:]
            break

    def close(self) -> None:
        """Flush any held-back text."""
        if self._buf:
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, text: str) -> None:
        if not text:
            return
        if self._answer_pending and not self._in_thinking:
            if not text.strip():
                return  # drop the pure-whitespace gap after </think>
            self._answer_pending = False
            text = text.lstrip("\n")
        self._on_segment("thinking" if self._in_thinking else "answer", text)

    @staticmethod
    def _partial_tail_len(buf: str, tag: str) -> int:
        """Length of the longest suffix of ``buf`` that prefixes ``tag``."""
        max_k = min(len(buf), len(tag) - 1)
        for k in range(max_k, 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0


class RichChatRenderer:
    """Streams a chat response to the terminal via rich.

    Thinking text (``<think>...</think>``) streams as muted text under a muted
    "thinking" rule; the answer streams as live-rendered Markdown, so code
    blocks and emphasis appear formatted as tokens arrive. Refresh is capped
    at 10/s to limit re-render flicker.
    """

    def __init__(self, console: Optional[Console] = None):
        if console is None:
            from .console import console as shared_console
            console = shared_console
        self._console = console
        self._parser = ThinkTagParser(self._on_segment)
        self._thinking = ""
        self._answer = ""
        self._closed = False
        self._live = Live(
            console=self._console,
            refresh_per_second=10,
            vertical_overflow="visible",
        )
        self._live.start()

    def feed(self, text: str) -> None:
        if self._closed:
            return
        self._parser.feed(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._parser.close()
        self._live.update(self._renderable(), refresh=True)
        self._live.stop()

    def _on_segment(self, kind: str, text: str) -> None:
        if kind == "thinking":
            self._thinking += text
        else:
            self._answer += text
        self._live.update(self._renderable())

    def _renderable(self) -> Group:
        parts = []
        if self._thinking:
            parts.append(Rule("thinking", style="wllm.border", align="left"))
            parts.append(Text(self._thinking.strip("\n"), style="wllm.muted"))
            parts.append(Rule(style="wllm.border"))
        if self._answer:
            parts.append(Markdown(self._answer, code_theme="github-dark"))
        return Group(*parts)
