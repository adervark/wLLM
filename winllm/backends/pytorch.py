"""Default PyTorch/Transformers inference backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..core.interfaces import InferenceBackend
from .compat import prepare_transformers_compat
from .tokenizers import load_tokenizer

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


class PyTorchBackend(InferenceBackend):
    """Loads models via ``transformers.AutoModelForCausalLM``."""

    name = "pytorch"

    def load(self, model_config: "ModelConfig", **load_kwargs: Any) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM

        # Apply 'day zero' architecture compatibility
        prepare_transformers_compat(model_config)

        tokenizer = load_tokenizer(
            model_config.model_name_or_path,
            model_config.trust_remote_code,
        )

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        return model, tokenizer
