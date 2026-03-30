"""Hardware detection and device abstraction for cross-platform scaling."""

from __future__ import annotations

import logging
import platform
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class SystemProfile(str, Enum):
    """Classification of the hardware environment."""
    CPU = "cpu"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    total_vram_gb: float
    compute_capability: tuple[int, int]   # e.g. (8, 9) for Ada Lovelace
    is_available: bool = True

    @property
    def vram_tier(self) -> str:
        if self.total_vram_gb >= 80:
            return "datacenter"   # A100-80GB, H100, H200
        elif self.total_vram_gb >= 40:
            return "workstation"  # A100-40GB, A6000
        elif self.total_vram_gb >= 16:
            return "desktop"     # RTX 4090 (24GB), 3090, 4080
        else:
            return "laptop"      # RTX 4070M (8GB), 4060M, etc.



@dataclass
class DeviceInfo:
    """Complete hardware environment description."""
    device_type: str                     # "cuda", "cpu"
    device_count: int                    # Number of GPUs (0 for CPU)
    devices: list[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0
    total_cpu_ram_gb: float = 0.0
    platform: str = ""                   # "windows", "linux", "darwin"
    profile: SystemProfile = SystemProfile.CPU
    defaults: Optional['HardwareDefaults'] = None

    @staticmethod
    def detect() -> DeviceInfo:
        """Auto-detect hardware and classify into a profile."""
        import torch

        system_platform = platform.system().lower()
        
        ram_gb = 0.0
        if psutil:
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

        info = DeviceInfo(
            device_type="cpu",
            device_count=0,
            platform=system_platform,
            total_cpu_ram_gb=ram_gb,
            profile=SystemProfile.CPU,
        )

        if not torch.cuda.is_available():
            logger.info("No CUDA GPUs detected — running in CPU-only mode")
            logger.debug("Torch version: %s", torch.__version__)
            try:
                logger.debug("CUDA version info: %s", torch.version.cuda)
            except Exception:
                pass
            info.defaults = _build_defaults(info)
            return info

        info.device_type = "cuda"
        info.device_count = torch.cuda.device_count()

        for i in range(info.device_count):
            props = torch.cuda.get_device_properties(i)
            gpu = GPUInfo(
                index=i,
                name=props.name,
                total_vram_gb=round(props.total_memory / (1024 ** 3), 2),
                compute_capability=(props.major, props.minor),
            )
            info.devices.append(gpu)
            info.total_vram_gb += gpu.total_vram_gb

        info.total_vram_gb = round(info.total_vram_gb, 2)

        # Classify profile
        info.classify_profile()

        # Set dynamic allocation
        info.defaults = _build_defaults(info)

        logger.info(
            "Hardware detected: %d GPU(s), %.1f GB total VRAM, platform=%s",
            info.device_count, info.total_vram_gb, info.platform,
        )
        for gpu in info.devices:
            logger.info(
                "  GPU %d: %s (%.1f GB, compute %d.%d)",
                gpu.index, gpu.name, gpu.total_vram_gb,
                gpu.compute_capability[0], gpu.compute_capability[1],
            )

        return info

    def classify_profile(self):
        """Classify the current hardware into a SystemProfile."""
        if self.device_type == "cpu" or self.device_count == 0:
            self.profile = SystemProfile.CPU
            return

        if self.total_vram_gb >= 40:
            self.profile = SystemProfile.EXTREME
        elif self.total_vram_gb >= 16:
            self.profile = SystemProfile.HIGH
        elif self.total_vram_gb >= 8:
            self.profile = SystemProfile.MEDIUM
        else:
            self.profile = SystemProfile.LOW

    def summary(self) -> dict:
        """JSON-friendly summary."""
        return {
            "device_type": self.device_type,
            "device_count": self.device_count,
            "total_vram_gb": self.total_vram_gb,
            "total_cpu_ram_gb": self.total_cpu_ram_gb,
            "platform": self.platform,
            "profile": self.profile.value,
            "gpus": [
                {"index": g.index, "name": g.name, "vram_gb": g.total_vram_gb, "compute_capability": g.compute_capability}
                for g in self.devices
            ],
            "defaults": {
                "quantization": self.defaults.default_quantization,
                "max_batch_size": self.defaults.max_batch_size,
                "max_model_len": self.defaults.max_model_len,
                "tensor_parallel_size": self.defaults.tensor_parallel_size,
                "attention_backend": self.defaults.attention_backend,
            },
        }



def _apply_env_overrides(defaults: HardwareDefaults) -> HardwareDefaults:
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


@dataclass
class HardwareDefaults:
    """Auto-tuned defaults mathematically calculated from hardware."""
    default_quantization: str       # "4bit", "8bit", "none"
    max_batch_size: int
    max_model_len: int
    device_map_strategy: str        # "auto", "balanced", "sequential"
    tensor_parallel_size: int
    gpu_memory_utilization: float
    kv_cache_fraction: float
    attention_backend: str = "sdpa" # "sdpa", "flash_attention_2", "eager"

def _build_defaults(info: DeviceInfo) -> HardwareDefaults:
    """Build auto-tuned defaults dynamically from actual hardware."""
    if info.device_type == "cpu" or info.device_count == 0:
        defaults = HardwareDefaults(
            default_quantization="4bit",
            max_batch_size=1,
            max_model_len=2048,
            device_map_strategy="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.0,
            kv_cache_fraction=0.3,
            attention_backend="sdpa"
        )
        return _apply_env_overrides(defaults)

    # Dynamic Allocation Math
    defaults = HardwareDefaults(
        default_quantization="4bit" if info.total_vram_gb < 16 else "none",
        max_batch_size=max(1, int(info.total_vram_gb / 1.5)), # Scaled dynamically
        max_model_len=8192 if info.total_vram_gb >= 24 else (4096 if info.total_vram_gb >= 12 else 2048),
        device_map_strategy="balanced" if info.device_count > 1 else "auto",
        tensor_parallel_size=info.device_count,
        gpu_memory_utilization=0.90, # Keep VRAM allowance high for Pooled allocation
        kv_cache_fraction=0.90,      # Pre-allocate 90% of REMAINING free vram to pool
        attention_backend="sdpa"
    )

    # Auto-detect attention backend based on compute capability
    min_compute = min(g.compute_capability for g in info.devices)
    if min_compute >= (8, 0):
        defaults.attention_backend = "flash_attention_2"

    return _apply_env_overrides(defaults)


class MemoryUtils:
    """Utilities for tracking GPU memory usage."""

    @staticmethod
    def get_all_info() -> list[dict[str, float]]:
        """Get memory info for all GPUs."""
        import torch
        if not torch.cuda.is_available():
            return []
        result = []
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            result.append({
                "device": i,
                "name": torch.cuda.get_device_properties(i).name,
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(total - reserved, 2),
            })
        return result

    @staticmethod
    def get_total_vram() -> float:
        """Get total VRAM across all GPUs in bytes."""
        import torch
        if not torch.cuda.is_available():
            return 0.0
        total = 0.0
        for i in range(torch.cuda.device_count()):
            total += torch.cuda.get_device_properties(i).total_memory
        return total

    @staticmethod
    def get_info(device_index: int = 0) -> dict[str, float]:
        """Get current GPU memory usage in GB for a single device."""
        import torch
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}
        total = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        return {
            "total": round(total, 2),
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(total - reserved, 2),
        }

    @staticmethod
    def get_aggregate_info() -> dict[str, float]:
        """Get aggregate GPU memory across all devices."""
        import torch
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "reserved": 0, "free": 0, "device_count": 0}
        total = allocated = reserved = 0.0
        count = torch.cuda.device_count()
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            total += props.total_memory / (1024 ** 3)
            allocated += torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved += torch.cuda.memory_reserved(i) / (1024 ** 3)
        return {
            "total": round(total, 2),
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(total - reserved, 2),
            "device_count": count,
        }


# ------------------------------------------------------------------
# Backward-compatibility aliases
# ------------------------------------------------------------------
# These module-level functions are imported by older parts of the codebase
# (e.g., model_loader.py). New code should use MemoryUtils.xxx() directly.
get_all_gpu_memory_info = MemoryUtils.get_all_info
get_total_gpu_memory = MemoryUtils.get_total_vram
get_gpu_memory_info = MemoryUtils.get_info
get_aggregate_gpu_memory = MemoryUtils.get_aggregate_info
