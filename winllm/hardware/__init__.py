"""Hardware detection, auto-tuned defaults, and GPU memory utilities.

Split by responsibility (SRP):
  - info.py:     what hardware is present (detection + classification)
  - defaults.py: what settings that hardware implies (HardwareDefaults)
  - tuning.py:   applying those defaults to config objects
  - memory.py:   GPU memory introspection
  - cuda.py:     process-global CUDA backend configuration
"""

from .info import DeviceInfo, GPUInfo, SystemProfile
from .defaults import HardwareDefaults, apply_env_overrides, build_defaults
from .tuning import (
    apply_hardware_defaults,
    apply_kv_cache_defaults,
    apply_model_defaults,
    apply_scheduler_defaults,
)
from .memory import (
    MemoryUtils,
    get_aggregate_gpu_memory,
    get_all_gpu_memory_info,
    get_gpu_memory_info,
    get_total_gpu_memory,
)

__all__ = [
    "DeviceInfo",
    "GPUInfo",
    "SystemProfile",
    "HardwareDefaults",
    "build_defaults",
    "apply_env_overrides",
    "apply_hardware_defaults",
    "apply_model_defaults",
    "apply_scheduler_defaults",
    "apply_kv_cache_defaults",
    "MemoryUtils",
    "get_aggregate_gpu_memory",
    "get_all_gpu_memory_info",
    "get_gpu_memory_info",
    "get_total_gpu_memory",
]
