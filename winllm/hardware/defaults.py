"""Auto-tuned hardware defaults and environment-variable overrides."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .info import DeviceInfo

logger = logging.getLogger(__name__)


@dataclass
class HardwareDefaults:
    """Auto-tuned defaults mathematically calculated from hardware."""
    default_quantization: str       # "auto", "4bit", "8bit", "none"
    max_batch_size: int
    max_model_len: int
    device_map_strategy: str        # "auto", "balanced", "sequential"
    tensor_parallel_size: int
    gpu_memory_utilization: float
    kv_cache_fraction: float
    attention_backend: str = "sdpa"  # "sdpa", "flash_attention_2", "eager"
    default_dtype: str = "float16"   # "float16" or "bfloat16"


def apply_env_overrides(defaults: HardwareDefaults) -> HardwareDefaults:
    """Apply environment variable overrides to HardwareDefaults."""
    mapping = {
        "WINLLM_QUANTIZATION": ("default_quantization", str),
        "WINLLM_MAX_BATCH_SIZE": ("max_batch_size", int),
        "WINLLM_MAX_MODEL_LEN": ("max_model_len", int),
        "WINLLM_DEVICE_MAP": ("device_map_strategy", str),
        "WINLLM_TP_SIZE": ("tensor_parallel_size", int),
        "WINLLM_GPU_UTILIZATION": ("gpu_memory_utilization", float),
        "WINLLM_KV_FRACTION": ("kv_cache_fraction", float),
        "WINLLM_ATTENTION_BACKEND": ("attention_backend", str),
        "WINLLM_DTYPE": ("default_dtype", str),
    }

    for env_var, (field_name, type_func) in mapping.items():
        val = os.environ.get(env_var)
        if val is not None:
            try:
                setattr(defaults, field_name, type_func(val))
                logger.info("Overridden %s with %s (via %s)", field_name, val, env_var)
            except ValueError:
                logger.warning("Failed to parse %s from %s as %s", val, env_var, type_func.__name__)

    return defaults


def build_defaults(info: "DeviceInfo") -> HardwareDefaults:
    """Build auto-tuned defaults dynamically from actual hardware."""
    if info.device_type == "cpu" or info.device_count == 0:
        defaults = HardwareDefaults(
            default_quantization="auto",
            max_batch_size=1,
            max_model_len=2048,
            device_map_strategy="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.0,
            kv_cache_fraction=0.3,
            attention_backend="sdpa",
        )
        return apply_env_overrides(defaults)

    # Dynamic Allocation Math
    # Blackwell (sm_120) optimization: higher throughput/BW allow for larger context by default
    is_blackwell = any(g.compute_capability[0] >= 12 for g in info.devices)

    defaults = HardwareDefaults(
        # "auto" defers to load-time resolution, which knows the model's
        # actual weight size — VRAM alone can't decide (a 1B model on an
        # 8 GB card needs no quantization; a 70B on 24 GB does).
        default_quantization="auto",
        max_batch_size=max(1, int(info.total_vram_gb / 1.25)) if is_blackwell else max(1, int(info.total_vram_gb / 1.5)),
        max_model_len=16384 if (is_blackwell and info.total_vram_gb >= 12) else (8192 if info.total_vram_gb >= 24 else (4096 if info.total_vram_gb >= 12 else 2048)),
        device_map_strategy="balanced" if info.device_count > 1 else "auto",
        tensor_parallel_size=info.device_count,
        gpu_memory_utilization=0.90,
        kv_cache_fraction=0.90,
        attention_backend="sdpa",
    )

    # Auto-detect attention backend and dtype based on compute capability
    min_compute = min(g.compute_capability for g in info.devices)
    if min_compute >= (8, 0):
        defaults.attention_backend = "flash_attention_2"
        # Ampere+ natively supports bfloat16 at full throughput with better
        # numerical stability than float16 (larger exponent range, no NaN overflow)
        defaults.default_dtype = "bfloat16"

    return apply_env_overrides(defaults)
