"""Backend registry — maps backend names to implementations.

New backends plug in via ``default_registry.register(MyBackend)`` without
touching any existing loading code (Open/Closed Principle).
"""

from __future__ import annotations

import logging
from typing import Type

from ..core.interfaces import InferenceBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry of available inference backends, keyed by name."""

    def __init__(self, fallback: str = "pytorch"):
        self._backends: dict[str, Type[InferenceBackend]] = {}
        self._fallback = fallback

    def register(self, backend_cls: Type[InferenceBackend]) -> Type[InferenceBackend]:
        """Register a backend class under its ``name``. Usable as a decorator."""
        if not backend_cls.name:
            raise ValueError(f"{backend_cls.__name__} must define a non-empty 'name'")
        self._backends[backend_cls.name] = backend_cls
        return backend_cls

    def create(self, name: str) -> InferenceBackend:
        """Instantiate the backend registered under ``name``.

        Unknown names fall back to the default backend so that a typo in
        config degrades gracefully rather than crashing at load time.
        """
        backend_cls = self._backends.get(name)
        if backend_cls is None:
            logger.warning(
                "Unknown inference backend '%s' — falling back to '%s'",
                name, self._fallback,
            )
            backend_cls = self._backends[self._fallback]
        return backend_cls()

    @property
    def available(self) -> list[str]:
        return sorted(self._backends)


#: Process-wide registry with the built-in backends pre-registered.
default_registry = BackendRegistry()


def get_backend(name: str) -> InferenceBackend:
    """Get an instance of the backend registered under ``name``."""
    return default_registry.create(name)


def _register_builtin_backends() -> None:
    # Imported here (not at module top) to avoid a circular import:
    # the backend modules import nothing from this registry.
    from .directml import DirectMLBackend
    from .onnx import OnnxRuntimeBackend
    from .pytorch import PyTorchBackend

    default_registry.register(PyTorchBackend)
    default_registry.register(OnnxRuntimeBackend)
    default_registry.register(DirectMLBackend)


_register_builtin_backends()
