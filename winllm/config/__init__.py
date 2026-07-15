"""Configuration dataclasses for WinLLM.

Each module holds the configuration for exactly one subsystem (SRP).
Configs are pure data: they never import services, detect hardware, or
apply tuning themselves — see ``winllm.hardware.tuning`` for that.
"""

from .model import DType, ModelConfig, QuantizationType
from .sampling import SamplingParams
from .scheduling import SchedulerConfig
from .kv_cache import KVCacheConfig
from .server import ServerConfig

__all__ = [
    "DType",
    "ModelConfig",
    "QuantizationType",
    "SamplingParams",
    "SchedulerConfig",
    "KVCacheConfig",
    "ServerConfig",
]
