"""DirectML inference backend (cross-vendor Windows acceleration)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..core.interfaces import InferenceBackend
from .compat import prepare_transformers_compat
from .tokenizers import load_tokenizer

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


class DirectMLBackend(InferenceBackend):
    """Loads models via torch-directml (DX12 acceleration)."""

    name = "directml"

    def load(self, model_config: "ModelConfig", **load_kwargs: Any) -> tuple[Any, Any]:
        # Apply 'day zero' architecture compatibility
        prepare_transformers_compat(model_config)

        try:
            import torch_directml
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "DirectML backend requires 'torch-directml'. "
                "Install it with: pip install torch-directml"
            )

        logger.info("Loading model with DirectML backend (DX12 acceleration)")

        tokenizer = load_tokenizer(
            model_config.model_name_or_path,
            model_config.trust_remote_code,
        )

        # Load on CPU first then move to DML
        dml_device = torch_directml.device()
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=model_config.torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
        ).to(dml_device)

        return model, tokenizer
