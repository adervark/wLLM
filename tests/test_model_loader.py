"""Unit tests for model loading, quantization config, and device map resolution."""

from unittest.mock import MagicMock, patch
import pytest
import torch

from winllm.config import ModelConfig, QuantizationType, DType
from winllm.model_loader import (
    ModelLoader,
    _build_quantization_config,
    _extract_model_kv_params,
)


# --- _build_quantization_config ---


class TestBuildQuantizationConfig:
    def test_none_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.NONE)
        result = _build_quantization_config(config)
        assert result is None

    def test_nf4_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.NF4)
        result = _build_quantization_config(config)
        assert result is not None
        assert result.load_in_4bit is True
        assert result.bnb_4bit_quant_type == "nf4"
        assert result.bnb_4bit_use_double_quant is True

    def test_int8_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.INT8)
        result = _build_quantization_config(config)
        assert result is not None
        assert result.load_in_8bit is True

    def test_gptq_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.GPTQ)
        try:
            result = _build_quantization_config(config)
            assert result is not None
        except ImportError:
            # GPTQConfig may not be available in all environments
            pytest.skip("GPTQConfig not available")

    def test_awq_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.AWQ)
        try:
            result = _build_quantization_config(config)
            assert result is not None
        except ImportError:
            pytest.skip("AwqConfig not available")

    def test_nf4_uses_model_dtype(self):
        config = ModelConfig(
            model_name_or_path="test",
            quantization=QuantizationType.NF4,
            dtype=DType.BFLOAT16,
        )
        result = _build_quantization_config(config)
        assert result.bnb_4bit_compute_dtype == torch.bfloat16


# --- _extract_model_kv_params ---


class TestExtractModelKVParams:
    def _make_model_config(self, **kwargs):
        config = MagicMock()
        # Set all attributes to None by default
        for attr in (
            "num_hidden_layers", "n_layer", "num_layers",
            "num_key_value_heads", "num_kv_heads",
            "num_attention_heads", "n_head", "num_heads",
            "head_dim", "hidden_size",
        ):
            setattr(config, attr, None)
        # Then override with provided values
        for k, v in kwargs.items():
            setattr(config, k, v)
        model = MagicMock()
        model.config = config
        return model

    def test_standard_llama_config(self):
        model = self._make_model_config(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        result = _extract_model_kv_params(model)
        assert result["num_layers"] == 32
        assert result["num_kv_heads"] == 8
        assert result["head_dim"] == 128

    def test_gpt_style_config(self):
        model = self._make_model_config(
            n_layer=24,
            num_attention_heads=16,
            hidden_size=1024,
        )
        result = _extract_model_kv_params(model)
        assert result["num_layers"] == 24
        assert result["num_kv_heads"] == 16
        assert result["head_dim"] == 64  # 1024 / 16

    def test_head_dim_computed_from_hidden(self):
        model = self._make_model_config(
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_size=768,
        )
        result = _extract_model_kv_params(model)
        assert result["head_dim"] == 64  # 768 / 12

    def test_missing_all_attrs_returns_empty(self):
        model = self._make_model_config()
        result = _extract_model_kv_params(model)
        assert "num_layers" not in result
        assert "num_kv_heads" not in result

    def test_explicit_head_dim_preferred_over_computed(self):
        model = self._make_model_config(
            num_hidden_layers=32,
            num_attention_heads=16,
            hidden_size=2048,
            head_dim=96,  # Explicitly set, different from 2048/16=128
        )
        result = _extract_model_kv_params(model)
        assert result["head_dim"] == 96


# --- ModelLoader._resolve_device_map ---


class TestResolveDeviceMap:
    def test_cpu_device(self):
        config = ModelConfig(model_name_or_path="test", device="cpu")
        loader = ModelLoader(config)
        assert loader._resolve_device_map() == "cpu"

    def test_auto_device(self):
        config = ModelConfig(model_name_or_path="test", device="auto")
        loader = ModelLoader(config)
        assert loader._resolve_device_map() == config.device_map_strategy

    def test_specific_cuda_device(self):
        config = ModelConfig(model_name_or_path="test", device="cuda:1")
        loader = ModelLoader(config)
        result = loader._resolve_device_map()
        assert result == {"": "cuda:1"}

    def test_unknown_device_fallback(self):
        config = ModelConfig(model_name_or_path="test", device="xpu")
        loader = ModelLoader(config)
        assert loader._resolve_device_map() == config.device_map_strategy


# --- ModelLoader lifecycle ---


class TestModelLoaderLifecycle:
    def test_initial_state(self):
        config = ModelConfig(model_name_or_path="test")
        loader = ModelLoader(config)
        assert loader.model is None
        assert loader.tokenizer is None
        assert loader.draft_model is None

    def test_get_kv_cache_params_when_unloaded(self):
        config = ModelConfig(model_name_or_path="test")
        loader = ModelLoader(config)
        assert loader.get_kv_cache_params() == {}

    def test_unload_clears_state(self):
        config = ModelConfig(model_name_or_path="test")
        loader = ModelLoader(config)
        loader.model = MagicMock()
        loader.tokenizer = MagicMock()
        loader.unload()
        assert loader.model is None
        assert loader.tokenizer is None
