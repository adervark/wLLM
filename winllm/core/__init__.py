"""Core domain layer: request types and the abstractions (interfaces) that
the rest of the engine is built against.

Nothing in this package depends on concrete backends, schedulers, or
servers — dependencies point inward (Dependency Inversion Principle).
"""

from .types import GenerationRequest, RequestStatus
from .interfaces import (
    InferenceBackend,
    LogitsProcessor,
    SchedulingPolicy,
    StoppingCriterion,
)

__all__ = [
    "GenerationRequest",
    "RequestStatus",
    "InferenceBackend",
    "LogitsProcessor",
    "SchedulingPolicy",
    "StoppingCriterion",
]
