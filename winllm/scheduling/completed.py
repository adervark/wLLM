"""Bounded storage for finished requests."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.types import GenerationRequest


class CompletedRequestStore:
    """Keeps finished requests retrievable for a bounded time/count.

    Evicts by both TTL and max count so long-running servers don't leak
    memory through completed-request bookkeeping. Entries are held in an
    OrderedDict in finished order (the oldest is always at the front), so
    eviction pops from the front in O(evicted) instead of sorting everything.
    """

    def __init__(self, max_kept: int, ttl: float):
        self._max_kept = max_kept
        self._ttl = ttl
        # req_id -> (finished_time, request), oldest first
        self._completed: "OrderedDict[str, tuple[float, GenerationRequest]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._completed)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._completed

    def add(self, request: "GenerationRequest") -> None:
        self._completed[request.request_id] = (time.time(), request)
        # Keep finished order even if the same id is re-added.
        self._completed.move_to_end(request.request_id)

    def get(self, request_id: str) -> Optional["GenerationRequest"]:
        entry = self._completed.get(request_id)
        return entry[1] if entry else None

    @property
    def needs_eviction(self) -> bool:
        return len(self._completed) > self._max_kept * 1.5

    def evict(self) -> None:
        """Remove old entries by TTL, then by count — oldest (front) first."""
        now = time.time()

        # 1. TTL eviction: entries are in finished order, so stop at the first
        #    one that hasn't expired yet.
        while self._completed:
            rid = next(iter(self._completed))
            finished_at, _ = self._completed[rid]
            if (now - finished_at) > self._ttl:
                del self._completed[rid]
            else:
                break

        # 2. Count eviction (oldest first)
        while len(self._completed) > self._max_kept:
            self._completed.popitem(last=False)
