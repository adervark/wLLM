"""Unit tests for hardware detection and dynamic device abstraction."""

import pytest
import os
from winllm.device import (
    GPUInfo,
    HardwareDefaults,
    DeviceInfo,
    _build_defaults,
    _apply_env_overrides,
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


# ─── _build_defaults (Dynamic Allocation) ───────────────────────────────────


class TestBuildDefaults:

    def test_cpu_only(self):
        info = DeviceInfo(device_type="cpu", device_count=0)
        defaults = _build_defaults(info)
        assert defaults.max_batch_size == 1
        assert defaults.device_map_strategy == "auto"
        assert defaults.gpu_memory_utilization == 0.0
        assert defaults.tensor_parallel_size == 1

    def test_single_gpu_low_vram(self):
        gpu = GPUInfo(index=0, name="RTX 4060M", total_vram_gb=8.0, compute_capability=(8, 9))
        info = DeviceInfo(device_type="cuda", device_count=1, devices=[gpu], total_vram_gb=8.0)
        defaults = _build_defaults(info)
        assert defaults.default_quantization == "4bit"
        assert defaults.max_batch_size == max(1, int(8.0 / 1.5))  # 5
        assert defaults.tensor_parallel_size == 1
        assert defaults.attention_backend == "flash_attention_2"

    def test_single_gpu_high_vram(self):
        gpu = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9))
        info = DeviceInfo(device_type="cuda", device_count=1, devices=[gpu], total_vram_gb=24.0)
        defaults = _build_defaults(info)
        assert defaults.default_quantization == "none"
        assert defaults.max_batch_size == int(24.0 / 1.5)  # 16
        assert defaults.max_model_len == 8192

    def test_multi_gpu_sets_tensor_parallel(self):
        gpus = [
            GPUInfo(index=0, name="A100", total_vram_gb=80.0, compute_capability=(8, 0)),
            GPUInfo(index=1, name="A100", total_vram_gb=80.0, compute_capability=(8, 0)),
        ]
        info = DeviceInfo(device_type="cuda", device_count=2, devices=gpus, total_vram_gb=160.0)
        defaults = _build_defaults(info)
        assert defaults.tensor_parallel_size == 2
        assert defaults.device_map_strategy == "balanced"
        assert defaults.max_batch_size == int(160.0 / 1.5)  # 106


# ─── DeviceInfo ──────────────────────────────────────────────────────────────


class TestDeviceInfo:

    def test_summary_cpu_only(self):
        info = DeviceInfo(device_type="cpu", device_count=0, platform="windows")
        info.defaults = _build_defaults(info)
        s = info.summary()
        assert s["device_type"] == "cpu"
        assert s["device_count"] == 0
        assert "profile" not in s
        assert s["gpus"] == []

    def test_summary_with_gpus(self):
        gpu = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9))
        info = DeviceInfo(
            device_type="cuda", device_count=1, devices=[gpu],
            total_vram_gb=24.0, platform="windows",
        )
        info.defaults = _build_defaults(info)
        s = info.summary()
        assert s["device_count"] == 1
        assert len(s["gpus"]) == 1
        assert s["gpus"][0]["name"] == "RTX 4090"
        assert s["total_vram_gb"] == 24.0
        assert "profile" not in s


# ─── Env Overrides ───────────────────────────────────────────────────────────


class TestConfigEnhancements:

    def test_apply_env_overrides(self):
        d = HardwareDefaults(
            default_quantization="4bit",
            max_batch_size=4,
            max_model_len=4096,
            device_map_strategy="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            kv_cache_fraction=0.4,
            attention_backend="sdpa",
        )
        
        os.environ["WINLLM_MAX_BATCH_SIZE"] = "128"
        os.environ["WINLLM_ATTENTION_BACKEND"] = "flash_attention_2"
        os.environ["WINLLM_GPU_UTILIZATION"] = "0.95"
        
        try:
            d = _apply_env_overrides(d)
            assert d.max_batch_size == 128
            assert d.attention_backend == "flash_attention_2"
            assert d.gpu_memory_utilization == 0.95
        finally:
            del os.environ["WINLLM_MAX_BATCH_SIZE"]
            del os.environ["WINLLM_ATTENTION_BACKEND"]
            del os.environ["WINLLM_GPU_UTILIZATION"]

    def test_attention_backend_auto_detect(self):
        gpu_old = GPUInfo(index=0, name="RTX 2080", total_vram_gb=8.0, compute_capability=(7, 5))
        info_old = DeviceInfo(device_type="cuda", device_count=1, devices=[gpu_old], platform="windows")
        defaults_old = _build_defaults(info_old)
        assert defaults_old.attention_backend == "sdpa"

        gpu_new = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, compute_capability=(8, 9))
        info_new = DeviceInfo(device_type="cuda", device_count=1, devices=[gpu_new], platform="windows")
        defaults_new = _build_defaults(info_new)
        assert defaults_new.attention_backend == "flash_attention_2"
