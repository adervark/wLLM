"""Unit tests for configuration dataclasses."""

import pytest
from winllm.config import (
    ModelConfig,
    SamplingParams,
    SchedulerConfig,
    ServerConfig,
    KVCacheConfig,
    QuantizationType,
    DType,
)


# ─── SamplingParams validation ─────────────────────────────────────────────


class TestSamplingParamsValidation:
    def test_valid_defaults(self):
        p = SamplingParams()
        assert p.temperature == 0.7
        assert p.top_k == 50
        assert p.top_p == 0.9
        assert p.repetition_penalty == 1.1
        assert p.max_tokens == 512
        assert p.seed is None

    def test_temperature_zero_is_valid(self):
        p = SamplingParams(temperature=0)
        assert p.temperature == 0

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-0.1)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-1)

    def test_top_p_zero_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)

    def test_top_p_above_one_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.1)

    def test_top_p_one_is_valid(self):
        p = SamplingParams(top_p=1.0)
        assert p.top_p == 1.0

    def test_repetition_penalty_below_one_raises(self):
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=0.5)

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=-10)

    def test_custom_stop_sequences(self):
        p = SamplingParams(stop=["<|end|>", "\n\n"])
        assert p.stop == ["<|end|>", "\n\n"]


# ─── ModelConfig ───────────────────────────────────────────────────────────


class TestModelConfig:
    def test_defaults(self):
        c = ModelConfig(model_name_or_path="test-model")
        assert c.quantization == QuantizationType.NF4
        assert c.dtype == DType.FLOAT16
        assert c.max_model_len == 4096
        assert c.device == "auto"
        assert c.gpu_memory_utilization == 0.90
        assert c.tensor_parallel_size == 1
        assert c.device_map_strategy == "auto"
        assert c.cpu_offload is False
        assert c.trust_remote_code is False

    def test_apply_hardware_defaults_changes_quantization(self):
        from winllm.device import HardwareDefaults

        c = ModelConfig(model_name_or_path="unknown-model-xyz")
        hw = HardwareDefaults(
            default_quantization="none",
            max_batch_size=8,
            max_model_len=8192,
            device_map_strategy="balanced",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.92,
            kv_cache_fraction=0.5,
        )
        c.apply_hardware_defaults(hw)

        # NF4 + hw says "none" → should switch to NONE
        assert c.quantization == QuantizationType.NONE
        assert c.max_model_len == 8192
        assert c.tensor_parallel_size == 2
        assert c.gpu_memory_utilization == 0.92
        assert c.device_map_strategy == "balanced"

    def test_apply_hardware_defaults_preserves_user_overrides(self):
        """If user already set a non-default quantization, hw defaults shouldn't clobber it."""
        from winllm.device import HardwareDefaults

        c = ModelConfig(
            model_name_or_path="some-model",
            quantization=QuantizationType.INT8,  # User explicitly chose INT8
            max_model_len=2048,                  # User explicitly set 2048
        )
        hw = HardwareDefaults(
            default_quantization="none",
            max_batch_size=8,
            max_model_len=16384,
            device_map_strategy="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            kv_cache_fraction=0.5,
        )
        c.apply_hardware_defaults(hw)

        # INT8 != NF4, so the "change to NONE" logic should NOT trigger
        assert c.quantization == QuantizationType.INT8
        # max_model_len 2048 != sentinel 4096, so hw default shouldn't override
        assert c.max_model_len == 2048

    def test_apply_with_non_hardware_defaults_is_noop(self):
        c = ModelConfig(model_name_or_path="test")
        c.apply_hardware_defaults("not a HardwareDefaults object")
        assert c.quantization == QuantizationType.NF4  # Unchanged


# ─── SchedulerConfig ───────────────────────────────────────────────────────


class TestSchedulerConfig:
    def test_defaults(self):
        c = SchedulerConfig()
        assert c.max_batch_size == 4
        assert c.max_waiting_requests == 64
        assert c.scheduling_policy == "fcfs"

    def test_apply_hardware_defaults(self):
        from winllm.device import HardwareDefaults

        c = SchedulerConfig()
        hw = HardwareDefaults(
            default_quantization="none",
            max_batch_size=32,
            max_model_len=8192,
            device_map_strategy="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            kv_cache_fraction=0.5,
        )
        c.apply_hardware_defaults(hw)
        assert c.max_batch_size == 32


# ─── KVCacheConfig ──────────────────────────────────────────────────────────


class TestKVCacheConfig:
    def test_defaults(self):
        c = KVCacheConfig()
        assert c.block_size == 16
        assert c.gpu_memory_fraction == 0.4
        assert c.num_layers == 0
        assert c.num_kv_heads == 0

    def test_apply_hardware_defaults(self):
        from winllm.device import HardwareDefaults

        c = KVCacheConfig()
        hw = HardwareDefaults(
            default_quantization="none",
            max_batch_size=8,
            max_model_len=8192,
            device_map_strategy="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            kv_cache_fraction=0.6,
        )
        c.apply_hardware_defaults(hw)
        assert c.gpu_memory_fraction == 0.6


# ─── ServerConfig ──────────────────────────────────────────────────────────


class TestServerConfig:
    def test_defaults(self):
        c = ServerConfig()
        assert c.host == "0.0.0.0"
        assert c.port == 8000
        assert c.model_alias is None
        assert c.api_key is None
        assert c.cors_origins == ["*"]

    def test_custom_values(self):
        c = ServerConfig(host="127.0.0.1", port=9000, api_key="secret")
        assert c.host == "127.0.0.1"
        assert c.port == 9000
        assert c.api_key == "secret"


# ─── Enums ─────────────────────────────────────────────────────────────────


class TestEnums:
    def test_quantization_values(self):
        assert QuantizationType.NONE.value == "none"
        assert QuantizationType.INT8.value == "8bit"
        assert QuantizationType.NF4.value == "4bit"

    def test_dtype_values(self):
        assert DType.FLOAT16.value == "float16"
        assert DType.BFLOAT16.value == "bfloat16"
        assert DType.FLOAT32.value == "float32"
