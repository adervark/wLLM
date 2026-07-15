"""Scheduling policies: the order in which waiting requests are admitted."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..core.interfaces import SchedulingPolicy

if TYPE_CHECKING:
    from collections import deque

    from ..core.types import GenerationRequest


class FCFSPolicy(SchedulingPolicy):
    """First-come, first-served admission order."""

    def select(self, waiting: "deque[GenerationRequest]") -> Optional["GenerationRequest"]:
        if not waiting:
            return None
        return waiting.popleft()

    def requeue(self, waiting: "deque[GenerationRequest]", request: "GenerationRequest") -> None:
        waiting.appendleft(request)


#: Registry of named policies; new policies plug in here (OCP).
POLICIES: dict[str, type[SchedulingPolicy]] = {
    "fcfs": FCFSPolicy,
}


def get_policy(name: str) -> SchedulingPolicy:
    """Instantiate the policy registered under ``name`` (defaults to FCFS)."""
    return POLICIES.get(name, FCFSPolicy)()
