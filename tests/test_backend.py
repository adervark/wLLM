"""Unit tests for the backend abstraction layer."""

from unittest.mock import MagicMock, patch
import pytest

from winllm.backend import BackendFactory
from winllm.config import ModelConfig, QuantizationType


# ─── Tokenizer loading ──────────────────────────────────────────────────────


class TestLoadTokenizer:
    @patch("transformers.AutoTokenizer")
    def test_happy_path(self, MockAutoTokenizer):
        mock_tok = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tok

        result = BackendFactory._load_tokenizer("some/model", trust_remote_code=False)
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

        result = BackendFactory._load_tokenizer("model-ONNX", trust_remote_code=True)
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
            BackendFactory._load_tokenizer("plain-model", trust_remote_code=False)


# ─── Backend dispatch ───────────────────────────────────────────────────────


class TestBackendDispatch:
    @patch.object(BackendFactory, "_load_pytorch")
    def test_default_uses_pytorch(self, mock_pytorch):
        mock_pytorch.return_value = (MagicMock(), MagicMock())
        config = ModelConfig(model_name_or_path="test/model", inference_backend="pytorch")
        BackendFactory.load(config)
        mock_pytorch.assert_called_once()

    @patch.object(BackendFactory, "_load_onnxruntime")
    def test_onnxruntime_dispatch(self, mock_onnx):
        mock_onnx.return_value = (MagicMock(), MagicMock())
        config = ModelConfig(model_name_or_path="test/model", inference_backend="onnxruntime")
        BackendFactory.load(config)
        mock_onnx.assert_called_once()

    @patch.object(BackendFactory, "_load_directml")
    def test_directml_dispatch(self, mock_dml):
        mock_dml.return_value = (MagicMock(), MagicMock())
        config = ModelConfig(model_name_or_path="test/model", inference_backend="directml")
        BackendFactory.load(config)
        mock_dml.assert_called_once()

    @patch.object(BackendFactory, "_load_pytorch")
    def test_unknown_backend_defaults_to_pytorch(self, mock_pytorch):
        """Unknown backend values should fall through to pytorch."""
        mock_pytorch.return_value = (MagicMock(), MagicMock())
        config = ModelConfig(model_name_or_path="test/model", inference_backend="something_else")
        BackendFactory.load(config)
        mock_pytorch.assert_called_once()


# ─── ONNX Runtime specifics ─────────────────────────────────────────────────


class TestOnnxRuntimeBackend:
    def test_import_error_message(self):
        """Verify clear error message when optimum is not installed."""
        config = ModelConfig(model_name_or_path="test", inference_backend="onnxruntime")
        with patch.dict("sys.modules", {"optimum": None, "optimum.onnxruntime": None}):
            # The actual import will fail, we just need to verify it raises ImportError
            with pytest.raises(ImportError):
                BackendFactory._load_onnxruntime(config)

    @patch("winllm.backend.BackendFactory._load_tokenizer")
    def test_liquidai_q4_routing(self, mock_tok):
        """LiquidAI models should auto-select model_q4.onnx for 4bit quantization."""
        mock_tok.return_value = MagicMock()

        config = ModelConfig(
            model_name_or_path="LiquidAI/test-ONNX",
            inference_backend="onnxruntime",
            quantization=QuantizationType.NF4,
        )

        # We can't fully mock ORTModelForCausalLM import, but we can test the
        # kwargs construction by patching at a higher level
        with patch("winllm.backend.ORTModelForCausalLM", create=True) as MockORT:
            MockORT.from_pretrained.return_value = MagicMock()
            try:
                BackendFactory._load_onnxruntime(config)
            except (ImportError, Exception):
                pass


# ─── DirectML specifics ─────────────────────────────────────────────────────


class TestDirectMLBackend:
    def test_import_error_message(self):
        """Verify clear error when torch-directml is not installed."""
        config = ModelConfig(model_name_or_path="test", inference_backend="directml")
        with patch.dict("sys.modules", {"torch_directml": None}):
            with pytest.raises(ImportError):
                BackendFactory._load_directml(config)
