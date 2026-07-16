"""Unit tests for model loading, quantization config, and device map resolution."""

from unittest.mock import MagicMock, patch
import pytest
import torch

from winllm.config import ModelConfig, QuantizationType, DType
from winllm.models.loader import ModelLoader
from winllm.models.quantization import (
    build_quantization_config,
    estimate_weight_gb,
    resolve_auto_quantization,
)
from winllm.models.introspection import extract_kv_params


# --- resolve_auto_quantization ---


class TestResolveAutoQuantization:
    def _resolve(self, weight_gb, free_gb=8.0, **config_kwargs):
        config = ModelConfig(model_name_or_path="test", **config_kwargs)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch(
                 "winllm.hardware.memory.get_aggregate_gpu_memory",
                 return_value={"total": free_gb, "allocated": 0.0,
                               "reserved": 0.0, "free": free_gb,
                               "device_count": 1},
             ), \
             patch(
                 "winllm.models.quantization.estimate_weight_gb",
                 return_value=weight_gb,
             ):
            return resolve_auto_quantization(config)

    def test_model_that_fits_is_not_quantized(self):
        # 2.2 GB weights vs 8*0.9-1.5 = 5.7 GB budget
        assert self._resolve(weight_gb=2.2) == QuantizationType.NONE

    def test_model_that_does_not_fit_gets_4bit(self):
        # 14 GB weights (7B fp16) on an 8 GB card
        assert self._resolve(weight_gb=14.0) == QuantizationType.NF4

    def test_unknown_size_falls_back_to_4bit(self):
        assert self._resolve(weight_gb=None) == QuantizationType.NF4

    def test_no_cuda_resolves_to_none(self):
        config = ModelConfig(model_name_or_path="test")
        with patch("torch.cuda.is_available", return_value=False):
            assert resolve_auto_quantization(config) == QuantizationType.NONE

    def test_zero_visible_devices_resolves_to_none(self):
        # CUDA_VISIBLE_DEVICES="" leaves is_available() True on some boxes
        config = ModelConfig(model_name_or_path="test")
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=0):
            assert resolve_auto_quantization(config) == QuantizationType.NONE

    def test_cpu_device_resolves_to_none(self):
        config = ModelConfig(model_name_or_path="test", device="cpu")
        assert resolve_auto_quantization(config) == QuantizationType.NONE


class TestEstimateWeightGb:
    def test_local_dir_sums_weight_files(self, tmp_path):
        (tmp_path / "model-00001.safetensors").write_bytes(b"\0" * 1024)
        (tmp_path / "model-00002.safetensors").write_bytes(b"\0" * 1024)
        (tmp_path / "tokenizer.json").write_bytes(b"\0" * 4096)  # not weights
        size = estimate_weight_gb(str(tmp_path))
        assert size == pytest.approx(2048 / 1024**3)

    def test_local_dir_without_weights_returns_none(self, tmp_path):
        (tmp_path / "config.json").write_bytes(b"{}")
        assert estimate_weight_gb(str(tmp_path)) is None


# --- build_quantization_config ---


class TestBuildQuantizationConfig:
    def test_auto_default(self):
        assert ModelConfig(model_name_or_path="test").quantization == QuantizationType.AUTO

    def test_auto_builds_no_config(self):
        # AUTO is resolved by the loader before this is called; falling
        # through to unquantized is the safe behavior if it isn't.
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.AUTO)
        assert build_quantization_config(config) is None

    def test_none_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.NONE)
        result = build_quantization_config(config)
        assert result is None

    def test_nf4_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.NF4)
        result = build_quantization_config(config)
        assert result is not None
        assert result.load_in_4bit is True
        assert result.bnb_4bit_quant_type == "nf4"
        assert result.bnb_4bit_use_double_quant is True

    def test_int8_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.INT8)
        result = build_quantization_config(config)
        assert result is not None
        assert result.load_in_8bit is True

    def test_gptq_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.GPTQ)
        try:
            result = build_quantization_config(config)
            assert result is not None
        except ImportError:
            # GPTQConfig may not be available in all environments
            pytest.skip("GPTQConfig not available")

    def test_awq_quantization(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.AWQ)
        try:
            result = build_quantization_config(config)
            assert result is not None
        except ImportError:
            pytest.skip("AwqConfig not available")

    def test_nf4_uses_model_dtype(self):
        config = ModelConfig(
            model_name_or_path="test",
            quantization=QuantizationType.NF4,
            dtype=DType.BFLOAT16,
        )
        result = build_quantization_config(config)
        assert result.bnb_4bit_compute_dtype == torch.bfloat16


# --- extract_kv_params ---


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
        result = extract_kv_params(model)
        assert result["num_layers"] == 32
        assert result["num_kv_heads"] == 8
        assert result["head_dim"] == 128

    def test_gpt_style_config(self):
        model = self._make_model_config(
            n_layer=24,
            num_attention_heads=16,
            hidden_size=1024,
        )
        result = extract_kv_params(model)
        assert result["num_layers"] == 24
        assert result["num_kv_heads"] == 16
        assert result["head_dim"] == 64  # 1024 / 16

    def test_head_dim_computed_from_hidden(self):
        model = self._make_model_config(
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_size=768,
        )
        result = extract_kv_params(model)
        assert result["head_dim"] == 64  # 768 / 12

    def test_missing_all_attrs_returns_empty(self):
        model = self._make_model_config()
        result = extract_kv_params(model)
        assert "num_layers" not in result
        assert "num_kv_heads" not in result

    def test_explicit_head_dim_preferred_over_computed(self):
        model = self._make_model_config(
            num_hidden_layers=32,
            num_attention_heads=16,
            hidden_size=2048,
            head_dim=96,  # Explicitly set, different from 2048/16=128
        )
        result = extract_kv_params(model)
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


# --- ModelLoader._build_load_kwargs ---


class TestBuildLoadKwargs:
    def test_unquantized_passes_dtype_not_torch_dtype(self):
        # transformers 5.x deprecated torch_dtype= and warns on every load
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.NONE)
        loader = ModelLoader(config)
        kwargs = loader._build_load_kwargs(config, None, "auto")
        assert kwargs["dtype"] == config.torch_dtype
        assert "torch_dtype" not in kwargs

    def test_quantized_omits_dtype(self):
        config = ModelConfig(model_name_or_path="test", quantization=QuantizationType.NF4)
        loader = ModelLoader(config)
        kwargs = loader._build_load_kwargs(config, build_quantization_config(config), "auto")
        assert "dtype" not in kwargs
        assert "torch_dtype" not in kwargs


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
