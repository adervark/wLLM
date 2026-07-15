"""Model loading configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantizationType(str, Enum):
    """Supported quantization methods."""
    AUTO = "auto"   # Resolved at load time by resolve_auto_quantization()
    NONE = "none"
    INT8 = "8bit"
    NF4 = "4bit"
    AWQ = "awq"
    GPTQ = "gptq"


class DType(str, Enum):
    """Supported compute dtypes."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name_or_path: str
    draft_model_name_or_path: Optional[str] = None  # Optional small model for speculative decoding
    # AUTO picks NONE when full-precision weights fit in free VRAM and NF4
    # otherwise; 4-bit costs ~35% decode speed and doubles load time, so it
    # must never be applied to models that fit as-is.
    quantization: QuantizationType = QuantizationType.AUTO
    dtype: DType = DType.FLOAT16
    max_model_len: int = 4096
    trust_remote_code: bool = False
    device: str = "auto"                    # "auto", "cuda", "cuda:0", "cpu"
    gpu_memory_utilization: float = 0.90
    inference_backend: str = "pytorch"      # "pytorch", "onnxruntime", "directml"
    # --- Multi-GPU / scaling ---
    tensor_parallel_size: int = 1           # Number of GPUs for tensor parallelism
    device_map_strategy: str = "auto"       # "auto", "balanced", "balanced_low_0", "sequential"
    cpu_offload: bool = False               # Offload layers to CPU if they don't fit in VRAM
    attention_backend: str = "auto"         # "auto", "sdpa", "flash_attention_2", "eager"
    force_architecture: Optional[str] = None  # Manual override for unknown model types
    # Capture the single-request decode forward in a CUDA graph (experimental).
    # Validated against eager output at load; falls back automatically if the
    # architecture isn't graph-safe.
    enable_cuda_graphs: bool = False
    # Model-free speculative decoding via suffix matching over the request's
    # own prompt/output (SuffixDecoding, arXiv:2411.04975). No draft model,
    # no extra VRAM; ignored when a draft model is configured.
    enable_suffix_decoding: bool = False

    @property
    def torch_dtype(self):
        import torch
        return {
            DType.FLOAT16: torch.float16,
            DType.BFLOAT16: torch.bfloat16,
            DType.FLOAT32: torch.float32,
        }[self.dtype]
