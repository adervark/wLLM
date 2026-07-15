"""Applies hardware defaults and model profiles to config objects.

This lives here — not on the config dataclasses — so that configs stay
pure data (SRP) and never import detection/registry services (DIP).
Only values still at their dataclass defaults are overwritten, so
explicit user choices always win.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .defaults import HardwareDefaults

if TYPE_CHECKING:
    from ..config import KVCacheConfig, ModelConfig, SchedulerConfig

logger = logging.getLogger(__name__)


def apply_model_defaults(config: "ModelConfig", defaults: HardwareDefaults) -> None:
    """Apply auto-detected hardware defaults to a ModelConfig.

    Only overwrites values the user left at their defaults, then layers
    on any known model-family profile heuristics.
    """
    from ..config import DType, QuantizationType
    from ..models.profiles import apply_model_profile, identify_model_profile

    if not isinstance(defaults, HardwareDefaults):
        return

    # 1. Apply hardware defaults first. A non-"auto" default only comes
    # from the WINLLM_QUANTIZATION env override; otherwise AUTO is left
    # for the loader to resolve against the model's actual size.
    if (
        config.quantization == QuantizationType.AUTO
        and defaults.default_quantization != "auto"
    ):
        try:
            config.quantization = QuantizationType(defaults.default_quantization)
        except ValueError:
            logger.warning(
                "Ignoring invalid default_quantization %r",
                defaults.default_quantization,
            )
    if config.max_model_len == 4096 and defaults.max_model_len is not None:
        config.max_model_len = defaults.max_model_len
    if config.tensor_parallel_size == 1 and defaults.tensor_parallel_size > 1:
        config.tensor_parallel_size = defaults.tensor_parallel_size
    if config.device_map_strategy == "auto":
        config.device_map_strategy = defaults.device_map_strategy
    if config.attention_backend == "auto":
        config.attention_backend = defaults.attention_backend
    config.gpu_memory_utilization = defaults.gpu_memory_utilization
    # Adopt bfloat16 on Ampere+ if the user hasn't explicitly set dtype
    if config.dtype == DType.FLOAT16 and defaults.default_dtype == "bfloat16":
        config.dtype = DType.BFLOAT16

    # 2. Attempt to identify model profile and apply heuristics
    profile = identify_model_profile(config.model_name_or_path)
    if profile:
        apply_model_profile(config, profile)


def apply_scheduler_defaults(config: "SchedulerConfig", defaults: HardwareDefaults) -> None:
    """Apply auto-detected hardware defaults to a SchedulerConfig."""
    if not isinstance(defaults, HardwareDefaults):
        return
    if config.max_batch_size == 4:
        config.max_batch_size = defaults.max_batch_size


def apply_kv_cache_defaults(config: "KVCacheConfig", defaults: HardwareDefaults) -> None:
    """Apply auto-detected hardware defaults to a KVCacheConfig."""
    if not isinstance(defaults, HardwareDefaults):
        return
    config.gpu_memory_fraction = defaults.kv_cache_fraction


def apply_hardware_defaults(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    kv_cache_config: "KVCacheConfig",
    defaults: HardwareDefaults,
) -> None:
    """Apply hardware defaults to all engine configs at once."""
    apply_model_defaults(model_config, defaults)
    apply_scheduler_defaults(scheduler_config, defaults)
    apply_kv_cache_defaults(kv_cache_config, defaults)
