"""Configuration dataclasses for WinLLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuantizationType(str, Enum):
    """Supported quantization methods."""
    NONE = "none"
    INT8 = "8bit"
    NF4 = "4bit"


class DType(str, Enum):
    """Supported compute dtypes."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name_or_path: str
    quantization: QuantizationType = QuantizationType.NF4
    dtype: DType = DType.FLOAT16
    max_model_len: int = 4096
    trust_remote_code: bool = False
    device: str = "auto"                    # "auto", "cuda", "cuda:0", "cpu"
    gpu_memory_utilization: float = 0.90
    # --- Multi-GPU / scaling ---
    tensor_parallel_size: int = 1           # Number of GPUs for tensor parallelism
    device_map_strategy: str = "auto"       # "auto", "balanced", "balanced_low_0", "sequential"
    cpu_offload: bool = False               # Offload layers to CPU if they don't fit in VRAM

    @property
    def torch_dtype(self):
        import torch
        return {
            DType.FLOAT16: torch.float16,
            DType.BFLOAT16: torch.bfloat16,
            DType.FLOAT32: torch.float32,
        }[self.dtype]

    def apply_hardware_defaults(self, defaults):
        """Apply auto-detected hardware defaults (only overwrites unset/default values)."""
        from .device import HardwareDefaults
        if not isinstance(defaults, HardwareDefaults):
            return

        # Only apply if user hasn't explicitly changed from the original defaults
        if self.quantization == QuantizationType.NF4 and defaults.default_quantization == "none":
            self.quantization = QuantizationType.NONE
        if self.max_model_len == 4096:
            self.max_model_len = defaults.max_model_len
        if self.tensor_parallel_size == 1 and defaults.tensor_parallel_size > 1:
            self.tensor_parallel_size = defaults.tensor_parallel_size
        if self.device_map_strategy == "auto":
            self.device_map_strategy = defaults.device_map_strategy
        self.gpu_memory_utilization = defaults.gpu_memory_utilization


@dataclass
class SamplingParams:
    """Parameters controlling token sampling."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")


@dataclass
class SchedulerConfig:
    """Configuration for the request scheduler."""
    max_batch_size: int = 4
    max_waiting_requests: int = 64
    max_num_seqs: int = 8
    scheduling_policy: str = "fcfs"

    def apply_hardware_defaults(self, defaults):
        """Apply auto-detected hardware defaults."""
        from .device import HardwareDefaults
        if not isinstance(defaults, HardwareDefaults):
            return
        if self.max_batch_size == 4:
            self.max_batch_size = defaults.max_batch_size


@dataclass
class ServerConfig:
    """Configuration for the API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_alias: Optional[str] = None
    api_key: Optional[str] = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class KVCacheConfig:
    """Configuration for KV cache management."""
    block_size: int = 16
    max_blocks_per_seq: int = 256
    gpu_memory_fraction: float = 0.4
    # --- Model-aware estimation (set at runtime) ---
    num_layers: int = 0           # Populated from loaded model
    num_kv_heads: int = 0         # Populated from loaded model
    head_dim: int = 0             # Populated from loaded model
    dtype_bytes: int = 2          # 2 for float16, 4 for float32

    def apply_hardware_defaults(self, defaults):
        """Apply auto-detected hardware defaults."""
        from .device import HardwareDefaults
        if not isinstance(defaults, HardwareDefaults):
            return
        self.gpu_memory_fraction = defaults.kv_cache_fraction
