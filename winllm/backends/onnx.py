"""ONNX Runtime inference backend (via HuggingFace Optimum)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ..core.interfaces import InferenceBackend
from .compat import prepare_transformers_compat
from .tokenizers import load_tokenizer

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


def detect_exported_onnx(model_name_or_path: str) -> bool:
    """Return True if the path/repo already contains exported ONNX binaries."""
    if os.path.exists(model_name_or_path):
        return any(f.endswith(".onnx") for f in os.listdir(model_name_or_path))
    return "ONNX" in model_name_or_path.upper()


def build_ort_kwargs(model_config: "ModelConfig", is_exported: bool) -> dict:
    """Build the kwargs for ``ORTModelForCausalLM.from_pretrained``.

    Kept as a pure function so the routing rules (export detection,
    LiquidAI repo layout, quantized binary selection, ORT provider)
    are unit-testable without touching the network.
    """
    ort_kwargs = {
        "model_id": model_config.model_name_or_path,
        "export": not is_exported,  # Don't export if it's already an ONNX folder or repo
        "trust_remote_code": model_config.trust_remote_code,
    }

    # Transparently handle LiquidAI's unique ONNX repository structure
    if is_exported and "LiquidAI" in model_config.model_name_or_path:
        ort_kwargs["subfolder"] = "onnx"
        # Map wLLM quantization arguments directly to the pre-compiled ONNX binaries!
        if getattr(model_config.quantization, "value", "none") == "4bit":
            ort_kwargs["file_name"] = "model_q4.onnx"
            logger.info("Auto-selected optimized INT4 (Q4) ONNX binary from LiquidAI repository.")
        elif getattr(model_config.quantization, "value", "none") == "8bit":
            ort_kwargs["file_name"] = "model_q8.onnx"
            logger.info("Auto-selected optimized INT8 (Q8) ONNX binary from LiquidAI repository.")
        else:
            ort_kwargs["file_name"] = "model.onnx"
            logger.info("Auto-selected standard FP32 ONNX binary from LiquidAI repository.")

    # Map device to ORT provider
    if model_config.device == "cpu":
        ort_kwargs["provider"] = "CPUExecutionProvider"
    else:
        ort_kwargs["provider"] = "CUDAExecutionProvider"

    return ort_kwargs


class OnnxRuntimeBackend(InferenceBackend):
    """Loads models via Optimum ONNX Runtime (no MSVC/Triton required)."""

    name = "onnxruntime"

    @staticmethod
    def _import_ort_class():
        """Import ORTModelForCausalLM — isolated for testability."""
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
        except ImportError:
            raise ImportError(
                "ONNX Runtime backend requires 'optimum' and 'onnxruntime-gpu'. "
                "To fix this in your current environment, run:\n"
                "  uv pip install optimum onnxruntime-gpu"
            )
        return ORTModelForCausalLM

    def load(self, model_config: "ModelConfig", **load_kwargs: Any) -> tuple[Any, Any]:
        # Apply 'day zero' architecture compatibility
        prepare_transformers_compat(model_config)

        ort_class = self._import_ort_class()

        logger.info("Loading model with ONNX Runtime backend (Windows-native acceleration)")

        tokenizer = load_tokenizer(
            model_config.model_name_or_path,
            model_config.trust_remote_code,
        )

        is_exported = detect_exported_onnx(model_config.model_name_or_path)
        ort_kwargs = build_ort_kwargs(model_config, is_exported)

        model = ort_class.from_pretrained(**ort_kwargs)
        return model, tokenizer
