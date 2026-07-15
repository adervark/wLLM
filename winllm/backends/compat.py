"""'Day zero' architecture compatibility aliasing for HuggingFace Transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


def prepare_transformers_compat(model_config: "ModelConfig") -> None:
    """Apply architecture aliasing and dynamic registration for 'day zero' model support."""
    from transformers import AutoConfig

    target_arch = model_config.force_architecture
    model_name = model_config.model_name_or_path.lower()

    # 1. Automatic 'Day Zero' Aliasing
    # If gemma-4 is detected, alias it to gemma2 architecture
    if not target_arch and ("gemma-4" in model_name or "gemma4" in model_name):
        target_arch = "gemma2"
        logger.info("Detected Gemma 4 model. Applying 'day zero' compatibility alias to 'gemma2' architecture.")

    # 2. Apply Registration
    if target_arch:
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            # We need to map the 'model_type' string found in config.json to a known Config class.
            # Usually the unknown type is 'gemma4' or similar.
            unknown_types = ["gemma4", "gemma-4"] if target_arch == "gemma2" else []

            # If the user forced an architecture, we might not know the unknown type yet,
            # but we can try to guess it from the path or just register a few common variants.
            if model_config.force_architecture:
                # Extract last part of path as a potential model_type
                potential_type = model_name.split("/")[-1].split("-")[0]
                if potential_type not in unknown_types:
                    unknown_types.append(potential_type)

            if target_arch in CONFIG_MAPPING:
                config_class = CONFIG_MAPPING[target_arch]
                for utype in unknown_types:
                    try:
                        AutoConfig.register(utype, config_class)
                        logger.debug("Registered alias: %s -> %s", utype, target_arch)
                    except Exception:
                        pass
            else:
                logger.warning("Target architecture '%s' not found in Transformers CONFIG_MAPPING.", target_arch)
        except Exception as e:
            logger.warning("Architecture aliasing failed: %s", e)
