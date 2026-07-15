"""Abstract interfaces (ports) for WinLLM's extension points.

These are the seams where the engine is Open for extension and Closed
for modification (OCP), and what higher-level orchestrators depend on
instead of concrete implementations (DIP):

  - InferenceBackend:  how model weights get loaded/executed
                       (PyTorch, ONNX Runtime, DirectML, ...).
  - LogitsProcessor:   one transform in the sampling pipeline
                       (repetition penalty, temperature, top-k, top-p, ...).
  - SchedulingPolicy:  the order in which waiting requests are admitted
                       (FCFS, priority, ...).
  - StoppingCriterion: one reason a running generation should stop
                       (EOS, max tokens, stop strings, cancellation, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections import deque

    import torch

    from ..config import ModelConfig
    from .types import GenerationRequest


class InferenceBackend(ABC):
    """Loads a model + tokenizer for a specific runtime (PyTorch, ONNX, ...).

    New backends are added by implementing this class and registering it
    in ``winllm.backends.registry`` — no existing code changes required.
    """

    #: Name used in ``ModelConfig.inference_backend`` to select this backend.
    name: str = ""

    @abstractmethod
    def load(self, model_config: "ModelConfig", **load_kwargs: Any) -> tuple[Any, Any]:
        """Load and return ``(model, tokenizer)`` for the given config."""
        raise NotImplementedError


@runtime_checkable
class LogitsProcessor(Protocol):
    """A single transform applied to logits before sampling.

    Implementations must be pure with respect to inputs other than
    ``logits`` (which they may modify in place and/or return).
    """

    def __call__(
        self,
        logits: "torch.Tensor",
        *,
        params: list,
        generated_ids: list,
    ) -> "torch.Tensor": ...


class SchedulingPolicy(ABC):
    """Decides which waiting request is admitted next (OCP seam)."""

    @abstractmethod
    def select(self, waiting: "deque[GenerationRequest]") -> Optional["GenerationRequest"]:
        """Remove and return the next request to admit, or None if empty."""
        raise NotImplementedError

    @abstractmethod
    def requeue(self, waiting: "deque[GenerationRequest]", request: "GenerationRequest") -> None:
        """Put a request back at the front of the queue (admission failed)."""
        raise NotImplementedError


class StoppingCriterion(ABC):
    """One reason a running generation should finish."""

    @abstractmethod
    def should_stop(self, request: "GenerationRequest") -> bool:
        raise NotImplementedError
