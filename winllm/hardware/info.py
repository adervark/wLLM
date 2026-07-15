"""Hardware detection and classification."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

try:
    import psutil
except ImportError:
    psutil = None

if TYPE_CHECKING:
    from .defaults import HardwareDefaults

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
            return "desktop"      # RTX 4090/5090 (24GB+), 4080/5080
        elif self.total_vram_gb >= 10:
            return "high-end laptop"  # RTX 5070M (12GB)
        else:
            return "laptop"       # RTX 5060M (8GB), 4060M, etc.


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
    defaults: Optional["HardwareDefaults"] = None

    @staticmethod
    def detect() -> DeviceInfo:
        """Auto-detect hardware and classify into a profile."""
        import torch

        from .defaults import build_defaults

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
            info.defaults = build_defaults(info)
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
        info.defaults = build_defaults(info)

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
                "dtype": self.defaults.default_dtype,
            },
        }
