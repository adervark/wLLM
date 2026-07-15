"""Inference backend implementations and their registry.

Each backend is a class implementing ``winllm.core.InferenceBackend``
(Liskov: any backend can stand in for any other). The registry is the
Open/Closed seam: adding a runtime means registering a new class —
no dispatch code is modified.
"""

from ..core.interfaces import InferenceBackend
from .registry import BackendRegistry, default_registry, get_backend
from .pytorch import PyTorchBackend
from .onnx import OnnxRuntimeBackend
from .directml import DirectMLBackend
from .tokenizers import load_tokenizer
from .compat import prepare_transformers_compat

__all__ = [
    "InferenceBackend",
    "BackendRegistry",
    "default_registry",
    "get_backend",
    "PyTorchBackend",
    "OnnxRuntimeBackend",
    "DirectMLBackend",
    "load_tokenizer",
    "prepare_transformers_compat",
]
