"""Unit tests for the backend abstraction layer."""

from unittest.mock import MagicMock, patch
import pytest

from winllm.backends import (
    DirectMLBackend,
    OnnxRuntimeBackend,
    PyTorchBackend,
    default_registry,
    get_backend,
    load_tokenizer,
)
from winllm.backends.onnx import build_ort_kwargs, detect_exported_onnx
from winllm.config import ModelConfig, QuantizationType


# ─── Tokenizer loading ──────────────────────────────────────────────────────


class TestLoadTokenizer:
    @patch("transformers.AutoTokenizer")
    def test_happy_path(self, MockAutoTokenizer):
        mock_tok = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tok

        result = load_tokenizer("some/model", trust_remote_code=False)
        assert result is mock_tok
        MockAutoTokenizer.from_pretrained.assert_called_once_with(
            "some/model", trust_remote_code=False
        )

    @patch("transformers.AutoTokenizer")
    def test_onnx_fallback(self, MockAutoTokenizer):
        """When loading an -ONNX model fails with TokenizersBackend, fall back to base."""
        mock_tok = MagicMock()
        MockAutoTokenizer.from_pretrained.side_effect = [
            ValueError("TokenizersBackend error"),
            mock_tok,
        ]

        result = load_tokenizer("model-ONNX", trust_remote_code=True)
        assert result is mock_tok
        # First call with ONNX name, second with base name
        assert MockAutoTokenizer.from_pretrained.call_count == 2
        second_call = MockAutoTokenizer.from_pretrained.call_args_list[1]
        assert second_call[0][0] == "model"

    @patch("transformers.AutoTokenizer")
    def test_non_onnx_error_reraises(self, MockAutoTokenizer):
        """Non-ONNX models should not silently catch ValueError."""
        MockAutoTokenizer.from_pretrained.side_effect = ValueError("some other error")

        with pytest.raises(ValueError, match="some other error"):
            load_tokenizer("plain-model", trust_remote_code=False)


# ─── Backend registry dispatch ──────────────────────────────────────────────


class TestBackendDispatch:
    def test_default_uses_pytorch(self):
        backend = get_backend("pytorch")
        assert isinstance(backend, PyTorchBackend)

    def test_onnxruntime_dispatch(self):
        backend = get_backend("onnxruntime")
        assert isinstance(backend, OnnxRuntimeBackend)

    def test_directml_dispatch(self):
        backend = get_backend("directml")
        assert isinstance(backend, DirectMLBackend)

    def test_unknown_backend_defaults_to_pytorch(self):
        """Unknown backend values should fall through to pytorch."""
        backend = get_backend("something_else")
        assert isinstance(backend, PyTorchBackend)

    def test_registry_lists_builtins(self):
        assert {"pytorch", "onnxruntime", "directml"} <= set(default_registry.available)

    @patch.object(PyTorchBackend, "load")
    def test_pytorch_load_invoked(self, mock_load):
        """Loading through the registry should call the backend's load()."""
        mock_load.return_value = (MagicMock(), MagicMock())
        config = ModelConfig(model_name_or_path="test/model", inference_backend="pytorch")
        backend = get_backend(config.inference_backend)
        backend.load(config)
        mock_load.assert_called_once()


# ─── ONNX Runtime specifics ─────────────────────────────────────────────────


class TestOnnxRuntimeBackend:
    def test_import_error_message(self):
        """Verify clear error message when optimum is not installed."""
        with patch.dict("sys.modules", {"optimum": None, "optimum.onnxruntime": None}):
            with pytest.raises(ImportError, match="optimum"):
                OnnxRuntimeBackend._import_ort_class()

    def test_detect_exported_from_repo_name(self):
        assert detect_exported_onnx("LiquidAI/test-ONNX") is True
        assert detect_exported_onnx("meta-llama/Llama-3-8B") is False

    def test_liquidai_q4_routing(self):
        """LiquidAI models should auto-select model_q4.onnx for 4bit quantization."""
        config = ModelConfig(
            model_name_or_path="LiquidAI/test-ONNX",
            inference_backend="onnxruntime",
            quantization=QuantizationType.NF4,
        )
        kwargs = build_ort_kwargs(config, is_exported=True)
        assert kwargs["subfolder"] == "onnx"
        assert kwargs["file_name"] == "model_q4.onnx"
        assert kwargs["export"] is False

    def test_liquidai_q8_routing(self):
        config = ModelConfig(
            model_name_or_path="LiquidAI/test-ONNX",
            inference_backend="onnxruntime",
            quantization=QuantizationType.INT8,
        )
        kwargs = build_ort_kwargs(config, is_exported=True)
        assert kwargs["file_name"] == "model_q8.onnx"

    def test_liquidai_fp32_routing(self):
        config = ModelConfig(
            model_name_or_path="LiquidAI/test-ONNX",
            inference_backend="onnxruntime",
            quantization=QuantizationType.NONE,
        )
        kwargs = build_ort_kwargs(config, is_exported=True)
        assert kwargs["file_name"] == "model.onnx"

    def test_provider_selection(self):
        cpu_config = ModelConfig(model_name_or_path="m", device="cpu")
        gpu_config = ModelConfig(model_name_or_path="m", device="auto")
        assert build_ort_kwargs(cpu_config, is_exported=False)["provider"] == "CPUExecutionProvider"
        assert build_ort_kwargs(gpu_config, is_exported=False)["provider"] == "CUDAExecutionProvider"

    def test_unexported_model_requests_export(self):
        config = ModelConfig(model_name_or_path="meta-llama/Llama-3-8B")
        kwargs = build_ort_kwargs(config, is_exported=False)
        assert kwargs["export"] is True
        assert "file_name" not in kwargs


# ─── DirectML specifics ─────────────────────────────────────────────────────


class TestDirectMLBackend:
    def test_import_error_message(self):
        """Verify clear error when torch-directml is not installed."""
        config = ModelConfig(model_name_or_path="test", inference_backend="directml")
        with patch.dict("sys.modules", {"torch_directml": None}):
            with pytest.raises(ImportError):
                DirectMLBackend().load(config)
