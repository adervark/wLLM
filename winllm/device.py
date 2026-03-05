"""Hardware detection and device abstraction for cross-platform scaling."""

from __future__ import annotations

import logging
import platform
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


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


class HardwareProfile(str, Enum):
    """Hardware tier classification."""
    CPU_ONLY = "cpu_only"
    LAPTOP = "laptop"           # 1 GPU, ≤12GB VRAM
    DESKTOP = "desktop"         # 1 GPU, 12-24GB VRAM
    WORKSTATION = "workstation" # 1-2 GPUs, 24-48GB each
    DATACENTER = "datacenter"   # 1-8+ GPUs, 40-141GB each


@dataclass
class HardwareDefaults:
    """Auto-tuned defaults for a hardware profile."""
    default_quantization: str       # "4bit", "8bit", "none"
    max_batch_size: int
    max_model_len: int
    device_map_strategy: str        # "auto", "balanced", "sequential"
    tensor_parallel_size: int
    gpu_memory_utilization: float
    kv_cache_fraction: float


# --- Profile defaults ---
PROFILE_DEFAULTS: dict[HardwareProfile, HardwareDefaults] = {
    HardwareProfile.CPU_ONLY: HardwareDefaults(
        default_quantization="4bit",
        max_batch_size=1,
        max_model_len=2048,
        device_map_strategy="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.0,
        kv_cache_fraction=0.3,
    ),
    HardwareProfile.LAPTOP: HardwareDefaults(
        default_quantization="4bit",
        max_batch_size=1,
        max_model_len=4096,
        device_map_strategy="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        kv_cache_fraction=0.4,
    ),
    HardwareProfile.DESKTOP: HardwareDefaults(
        default_quantization="none",
        max_batch_size=8,
        max_model_len=8192,
        device_map_strategy="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        kv_cache_fraction=0.5,
    ),
    HardwareProfile.WORKSTATION: HardwareDefaults(
        default_quantization="none",
        max_batch_size=16,
        max_model_len=16384,
        device_map_strategy="balanced",
        tensor_parallel_size=1,  # Updated to GPU count at runtime
        gpu_memory_utilization=0.90,
        kv_cache_fraction=0.5,
    ),
    HardwareProfile.DATACENTER: HardwareDefaults(
        default_quantization="none",
        max_batch_size=64,
        max_model_len=32768,
        device_map_strategy="balanced",
        tensor_parallel_size=1,  # Updated to GPU count at runtime
        gpu_memory_utilization=0.92,
        kv_cache_fraction=0.6,
    ),
}


@dataclass
class DeviceInfo:
    """Complete hardware environment description."""
    device_type: str                     # "cuda", "cpu"
    device_count: int                    # Number of GPUs (0 for CPU)
    devices: list[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0
    platform: str = ""                   # "windows", "linux", "darwin"
    profile: HardwareProfile = HardwareProfile.CPU_ONLY
    defaults: HardwareDefaults = field(default_factory=lambda: PROFILE_DEFAULTS[HardwareProfile.CPU_ONLY])

    @staticmethod
    def detect() -> DeviceInfo:
        """Auto-detect hardware and classify into a profile."""
        import torch

        system_platform = platform.system().lower()
        info = DeviceInfo(
            device_type="cpu",
            device_count=0,
            platform=system_platform,
        )

        if not torch.cuda.is_available():
            logger.info("No CUDA GPUs detected — running in CPU-only mode")
            info.profile = HardwareProfile.CPU_ONLY
            info.defaults = PROFILE_DEFAULTS[HardwareProfile.CPU_ONLY]
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
        info.profile = _classify_profile(info)
        info.defaults = _build_defaults(info)

        logger.info(
            "Hardware detected: %d GPU(s), %.1f GB total VRAM, profile=%s, platform=%s",
            info.device_count, info.total_vram_gb, info.profile.value, info.platform,
        )
        for gpu in info.devices:
            logger.info(
                "  GPU %d: %s (%.1f GB, compute %d.%d)",
                gpu.index, gpu.name, gpu.total_vram_gb,
                gpu.compute_capability[0], gpu.compute_capability[1],
            )

        return info

    def summary(self) -> dict:
        """JSON-friendly summary."""
        return {
            "device_type": self.device_type,
            "device_count": self.device_count,
            "total_vram_gb": self.total_vram_gb,
            "platform": self.platform,
            "profile": self.profile.value,
            "gpus": [
                {"index": g.index, "name": g.name, "vram_gb": g.total_vram_gb}
                for g in self.devices
            ],
            "defaults": {
                "quantization": self.defaults.default_quantization,
                "max_batch_size": self.defaults.max_batch_size,
                "max_model_len": self.defaults.max_model_len,
                "tensor_parallel_size": self.defaults.tensor_parallel_size,
            },
        }


def _classify_profile(info: DeviceInfo) -> HardwareProfile:
    """Classify hardware into a profile tier."""
    if info.device_count == 0:
        return HardwareProfile.CPU_ONLY

    max_vram = max(g.total_vram_gb for g in info.devices)

    if info.device_count >= 2 and max_vram >= 40:
        return HardwareProfile.DATACENTER
    elif max_vram >= 40:
        return HardwareProfile.DATACENTER
    elif info.device_count >= 2 and max_vram >= 24:
        return HardwareProfile.WORKSTATION
    elif max_vram >= 16:
        return HardwareProfile.DESKTOP
    else:
        return HardwareProfile.LAPTOP


def _build_defaults(info: DeviceInfo) -> HardwareDefaults:
    """Build auto-tuned defaults from profile + actual hardware."""
    defaults = HardwareDefaults(**vars(PROFILE_DEFAULTS[info.profile]))

    # Set tensor parallel to GPU count for multi-GPU setups
    if info.device_count > 1 and info.profile in (
        HardwareProfile.WORKSTATION, HardwareProfile.DATACENTER
    ):
        defaults.tensor_parallel_size = info.device_count

    # Adjust batch size based on total VRAM
    if info.total_vram_gb >= 200:
        defaults.max_batch_size = min(128, defaults.max_batch_size * 2)

    return defaults


def get_all_gpu_memory_info() -> list[dict[str, float]]:
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


def get_total_gpu_memory() -> float:
    """Get total VRAM across all GPUs in bytes."""
    import torch

    if not torch.cuda.is_available():
        return 0.0

    total = 0.0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.get_device_properties(i).total_memory
    return total
