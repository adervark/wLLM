"""Unit tests for hardware detection and device abstraction."""

import pytest
from winllm.device import (
    GPUInfo,
    HardwareProfile,
    HardwareDefaults,
    DeviceInfo,
    PROFILE_DEFAULTS,
    _classify_profile,
    _build_defaults,
)


# ─── GPUInfo ────────────────────────────────────────────────────────────────


class TestGPUInfo:

    def test_datacenter_tier(self):
        gpu = GPUInfo(index=0, name="A100", total_vram_gb=80.0, compute_capability=(8, 0))
        assert gpu.vram_tier == "datacenter"

    def test_workstation_tier(self):
        gpu = GPUInfo(index=0, name="A6000", total_vram_gb=48.0, compute_capability=(8, 6))
        assert gpu.vram_tier == "workstation"

    def test_desktop_tier(self):
        gpu = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9))
        assert gpu.vram_tier == "desktop"

    def test_laptop_tier(self):
        gpu = GPUInfo(index=0, name="RTX 4060M", total_vram_gb=8.0, compute_capability=(8, 9))
        assert gpu.vram_tier == "laptop"


# ─── _classify_profile ──────────────────────────────────────────────────────


class TestClassifyProfile:

    def test_cpu_only(self):
        info = DeviceInfo(device_type="cpu", device_count=0)
        assert _classify_profile(info) == HardwareProfile.CPU_ONLY

    def test_laptop_low_vram(self):
        gpu = GPUInfo(index=0, name="RTX 4060M", total_vram_gb=8.0, compute_capability=(8, 9))
        info = DeviceInfo(device_type="cuda", device_count=1, devices=[gpu])
        assert _classify_profile(info) == HardwareProfile.LAPTOP

    def test_desktop_mid_vram(self):
        gpu = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9))
        info = DeviceInfo(device_type="cuda", device_count=1, devices=[gpu])
        assert _classify_profile(info) == HardwareProfile.DESKTOP

    def test_datacenter_single_high_vram(self):
        gpu = GPUInfo(index=0, name="A100", total_vram_gb=80.0, compute_capability=(8, 0))
        info = DeviceInfo(device_type="cuda", device_count=1, devices=[gpu])
        assert _classify_profile(info) == HardwareProfile.DATACENTER

    def test_workstation_multi_gpu(self):
        gpus = [
            GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9)),
            GPUInfo(index=1, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9)),
        ]
        info = DeviceInfo(device_type="cuda", device_count=2, devices=gpus)
        assert _classify_profile(info) == HardwareProfile.WORKSTATION

    def test_datacenter_multi_gpu_high_vram(self):
        gpus = [
            GPUInfo(index=0, name="A100", total_vram_gb=80.0, compute_capability=(8, 0)),
            GPUInfo(index=1, name="A100", total_vram_gb=80.0, compute_capability=(8, 0)),
        ]
        info = DeviceInfo(device_type="cuda", device_count=2, devices=gpus)
        assert _classify_profile(info) == HardwareProfile.DATACENTER


# ─── _build_defaults ────────────────────────────────────────────────────────


class TestBuildDefaults:

    def test_single_gpu_desktop(self):
        gpu = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9))
        info = DeviceInfo(
            device_type="cuda", device_count=1, devices=[gpu],
            total_vram_gb=24.0, profile=HardwareProfile.DESKTOP,
        )
        defaults = _build_defaults(info)
        assert defaults.tensor_parallel_size == 1
        assert defaults.max_batch_size == PROFILE_DEFAULTS[HardwareProfile.DESKTOP].max_batch_size

    def test_multi_gpu_sets_tensor_parallel(self):
        gpus = [
            GPUInfo(index=0, name="A100", total_vram_gb=80.0, compute_capability=(8, 0)),
            GPUInfo(index=1, name="A100", total_vram_gb=80.0, compute_capability=(8, 0)),
        ]
        info = DeviceInfo(
            device_type="cuda", device_count=2, devices=gpus,
            total_vram_gb=160.0, profile=HardwareProfile.DATACENTER,
        )
        defaults = _build_defaults(info)
        assert defaults.tensor_parallel_size == 2

    def test_very_high_vram_scales_batch_size(self):
        gpus = [
            GPUInfo(index=i, name="H100", total_vram_gb=80.0, compute_capability=(9, 0))
            for i in range(4)
        ]
        info = DeviceInfo(
            device_type="cuda", device_count=4, devices=gpus,
            total_vram_gb=320.0, profile=HardwareProfile.DATACENTER,
        )
        defaults = _build_defaults(info)
        base = PROFILE_DEFAULTS[HardwareProfile.DATACENTER].max_batch_size
        assert defaults.max_batch_size == min(128, base * 2)


# ─── PROFILE_DEFAULTS ───────────────────────────────────────────────────────


class TestProfileDefaults:

    def test_all_profiles_have_defaults(self):
        for profile in HardwareProfile:
            assert profile in PROFILE_DEFAULTS

    def test_cpu_only_no_gpu_util(self):
        d = PROFILE_DEFAULTS[HardwareProfile.CPU_ONLY]
        assert d.gpu_memory_utilization == 0.0
        assert d.tensor_parallel_size == 1

    def test_laptop_uses_4bit(self):
        d = PROFILE_DEFAULTS[HardwareProfile.LAPTOP]
        assert d.default_quantization == "4bit"

    def test_desktop_uses_none_quantization(self):
        d = PROFILE_DEFAULTS[HardwareProfile.DESKTOP]
        assert d.default_quantization == "none"


# ─── HardwareDefaults dataclass ──────────────────────────────────────────────


class TestHardwareDefaults:

    def test_creation(self):
        d = HardwareDefaults(
            default_quantization="4bit",
            max_batch_size=4,
            max_model_len=4096,
            device_map_strategy="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            kv_cache_fraction=0.4,
        )
        assert d.default_quantization == "4bit"
        assert d.max_batch_size == 4
        assert d.kv_cache_fraction == 0.4


# ─── DeviceInfo ──────────────────────────────────────────────────────────────


class TestDeviceInfo:

    def test_summary_cpu_only(self):
        info = DeviceInfo(
            device_type="cpu", device_count=0, platform="windows",
            profile=HardwareProfile.CPU_ONLY,
            defaults=PROFILE_DEFAULTS[HardwareProfile.CPU_ONLY],
        )
        s = info.summary()
        assert s["device_type"] == "cpu"
        assert s["device_count"] == 0
        assert s["profile"] == "cpu_only"
        assert s["gpus"] == []

    def test_summary_with_gpus(self):
        gpu = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9))
        info = DeviceInfo(
            device_type="cuda", device_count=1, devices=[gpu],
            total_vram_gb=24.0, platform="windows",
            profile=HardwareProfile.DESKTOP,
            defaults=PROFILE_DEFAULTS[HardwareProfile.DESKTOP],
        )
        s = info.summary()
        assert s["device_count"] == 1
        assert len(s["gpus"]) == 1
        assert s["gpus"][0]["name"] == "RTX 4090"
        assert s["total_vram_gb"] == 24.0
