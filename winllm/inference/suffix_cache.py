"""Model-free draft source implementing the core of SuffixDecoding.

SuffixDecoding (Oliaro et al., NeurIPS 2025, arXiv:2411.04975) speculates
future tokens without any draft model: language-model output — especially
agentic traces, structured output, and "thinking" transcripts — repeats
itself and its prompt constantly. The longest suffix of the committed text
that already occurred earlier is located, and the tokens that followed that
earlier occurrence are proposed as the draft. Speculation length adapts to
the evidence: ``max_spec = alpha * pattern_len`` (the paper's ``αp`` rule),
so long matches speculate aggressively and weak matches stay cheap.

Two simplifications versus the paper, both driven by wLLM's design:
  - Linear drafts instead of speculation trees: the verifier processes one
    candidate sequence per step, so the greedy (most-recent longest-match)
    path is proposed rather than a frequency-scored tree.
  - Per-request history (prompt + own output) instead of a global
    cross-request tree; history dies with the request.

The index is positions-by-bigram with backward extension, which walks the
same match the paper's suffix tree would return for greedy path selection,
with O(occurrences) lookup and no tree maintenance.
"""

from __future__ import annotations

import os
from collections import OrderedDict


class _RequestHistory:
    """Token history and bigram index for one request."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        # (token_a, token_b) -> positions of token_a where that bigram starts
        self.bigrams: dict[tuple[int, int], list[int]] = {}

    def extend(self, new_tokens: list[int]) -> None:
        start = len(self.tokens)
        self.tokens.extend(new_tokens)
        # Index every bigram that ends inside the new region
        for pos in range(max(0, start - 1), len(self.tokens) - 1):
            key = (self.tokens[pos], self.tokens[pos + 1])
            self.bigrams.setdefault(key, []).append(pos)


class SuffixCache:
    """Tracks per-request token history and proposes draft continuations.

    Args:
        alpha: speculation-length multiplier over the matched pattern length
            (the paper's α; longer matches earn longer speculations).
        max_spec: hard cap on proposed tokens per step.
        max_back_ext: cap on how far a match is extended backwards; bounds
            per-proposal work on pathologically repetitive text.
        max_requests: LRU bound on tracked requests (histories die with the
            request, but the scheduler has no eviction hook into us).
    """

    def __init__(
        self,
        alpha: float = 2.0,
        max_spec: int = 16,
        max_back_ext: int = 64,
        max_requests: int = 64,
        max_occurrences: int = 32,
    ):
        self.alpha = alpha
        self.max_spec = max_spec
        self.max_back_ext = max_back_ext
        self.max_requests = max_requests
        self.max_occurrences = max_occurrences
        self._histories: OrderedDict[str, _RequestHistory] = OrderedDict()

    # ------------------------------------------------------------------
    # History maintenance
    # ------------------------------------------------------------------

    def sync(self, request_id: str, tokens: list[int]) -> None:
        """Bring a request's history up to date with its committed tokens.

        ``tokens`` is the full committed sequence (prompt + output so far);
        only the unseen tail is indexed, so this is cheap to call every step
        regardless of which code path committed the tokens.
        """
        history = self._histories.get(request_id)
        if history is None:
            history = _RequestHistory()
            self._histories[request_id] = history
            if len(self._histories) > self.max_requests:
                self._histories.popitem(last=False)
        self._histories.move_to_end(request_id)

        if len(tokens) > len(history.tokens):
            history.extend(tokens[len(history.tokens):])

    def evict(self, request_id: str) -> None:
        self._histories.pop(request_id, None)

    def has_request(self, request_id: str) -> bool:
        """True if a history is tracked for the request (test/debug probe)."""
        return request_id in self._histories

    # ------------------------------------------------------------------
    # Drafting
    # ------------------------------------------------------------------

    def propose(self, request_id: str) -> list[int]:
        """Propose draft tokens for the request's next decode step.

        Finds the earlier occurrence of the longest suffix of the committed
        text (anchored on its final bigram, extended backwards), and returns
        the tokens that followed it — at most ``alpha * pattern_len``.
        Returns [] when there is no usable match, signalling the caller to
        take the normal decode path at zero extra cost.
        """
        history = self._histories.get(request_id)
        if history is None or len(history.tokens) < 3:
            return []
        tokens = history.tokens
        end = len(tokens)  # suffix to match ends at tokens[end-1]

        anchor = (tokens[-2], tokens[-1])
        # Exclude the trailing bigram itself (starts at end-2). Only the most
        # recent occurrences are scanned: common bigrams (punctuation,
        # newlines) can occur hundreds of times in a long generation, and an
        # unbounded scan would put O(occurrences × max_back_ext) Python work
        # on every decode step. Recency also wins ties anyway.
        occurrences = [p for p in history.bigrams.get(anchor, ()) if p < end - 2]
        occurrences = occurrences[-self.max_occurrences:]
        if not occurrences:
            return []

        # Longest backward extension wins; most recent occurrence breaks ties
        # (recent context is the best predictor of what comes next).
        best_pos, best_ext = -1, -1
        for pos in occurrences:
            ext = 0
            while (
                ext < self.max_back_ext
                and pos - 1 - ext >= 0
                and end - 3 - ext >= 0
                and tokens[pos - 1 - ext] == tokens[end - 3 - ext]
            ):
                ext += 1
            if ext >= best_ext:  # >= so later (more recent) positions win ties
                best_pos, best_ext = pos, ext

        pattern_len = 2 + best_ext
        budget = min(int(self.alpha * pattern_len), self.max_spec)
        continuation_start = best_pos + 2
        return tokens[continuation_start:continuation_start + budget]


# ---------------------------------------------------------------------------
# Optional native (C++) implementation
# ---------------------------------------------------------------------------

PySuffixCache = SuffixCache
"""The pure-Python implementation: always importable, reference for parity
tests, and the fallback when the native module isn't built. The native
implementation lives in native/suffix/ and must mirror it exactly."""

if not os.environ.get("WINLLM_PURE_PYTHON"):
    try:
        from winllm_suffix import SuffixCache  # type: ignore[assignment] # noqa: F811
    except ImportError:
        pass
