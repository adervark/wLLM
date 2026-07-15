"""Tokenizer loading shared by all backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool) -> "PreTrainedTokenizerBase":
    """Load a tokenizer, working around the Optimum '-ONNX' repo quirk."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
    except ValueError as e:
        if "TokenizersBackend" in str(e) and "-ONNX" in model_name_or_path:
            base_name = model_name_or_path.replace("-ONNX", "")
            logger.warning(
                "Tokenizer loading failed due to Optimum 'TokenizersBackend' bug. "
                "Falling back to base model: %s", base_name
            )
            return AutoTokenizer.from_pretrained(
                base_name, trust_remote_code=trust_remote_code
            )
        raise
