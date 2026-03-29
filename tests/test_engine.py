"""Unit tests for the inference engine with mocked models."""

import time
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import torch

from winllm.engine import InferenceEngine
from winllm.config import ModelConfig, SamplingParams, KVCacheConfig
from winllm.types import GenerationRequest, RequestStatus


# ─── Fixtures ───────────────────────────────────────────────────────────────


def _make_mock_tokenizer():
    """Create a mock tokenizer with essential methods."""
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.decode.return_value = "decoded text"
    tok.eos_token_id = 2
    tok.pad_token = None
    tok.eos_token = "<eos>"
    return tok


def _make_mock_model(vocab_size=100):
    """Create a mock model that returns plausible outputs."""
    model = MagicMock()
    # Make model.parameters() return a real tensor so _resolve_device works
    param = torch.zeros(1)
    model.parameters.return_value = iter([param])

    # Mock forward pass output
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, vocab_size)
    mock_output.past_key_values = ((torch.zeros(1),),)
    model.return_value = mock_output

    model.eval = MagicMock()
    return model


@pytest.fixture
def model_config():
    return ModelConfig(model_name_or_path="test/model")


@pytest.fixture
def engine(model_config):
    return InferenceEngine(model_config)


# ─── Lifecycle ──────────────────────────────────────────────────────────────


class TestEngineLifecycle:
    def test_not_ready_before_load(self, engine):
        assert not engine.is_ready

    def test_unload_on_unloaded_engine(self, engine):
        """Unloading without loading should not crash."""
        engine.unload_model()
        assert not engine.is_ready
        assert engine.model is None
        assert engine.tokenizer is None

    @patch("winllm.engine.ModelLoader")
    def test_load_and_ready(self, MockLoader, model_config):
        mock_model = _make_mock_model()
        mock_tokenizer = _make_mock_tokenizer()

        instance = MockLoader.return_value
        instance.load.return_value = (mock_model, mock_tokenizer)
        instance.draft_model = None
        instance.get_kv_cache_params.return_value = {}

        engine = InferenceEngine(model_config)
        engine.load_model()

        assert engine.is_ready
        assert engine.model is mock_model
        assert engine.tokenizer is mock_tokenizer


# ─── Tokenization ──────────────────────────────────────────────────────────


class TestTokenization:
    def test_tokenize_delegates(self, engine):
        engine.tokenizer = _make_mock_tokenizer()
        result = engine.tokenize("hello")
        engine.tokenizer.encode.assert_called_once_with("hello", add_special_tokens=True)
        assert result == [1, 2, 3, 4, 5]

    def test_decode_tokens_delegates(self, engine):
        engine.tokenizer = _make_mock_tokenizer()
        result = engine.decode_tokens([1, 2, 3])
        engine.tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)
        assert result == "decoded text"


# ─── Validation ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_validate_prompt_valid(self, engine):
        engine.model_config.max_model_len = 4096
        req = GenerationRequest()
        assert engine._validate_prompt(req, 100) is True
        assert req.status != RequestStatus.FAILED

    def test_validate_prompt_too_long(self, engine):
        engine.model_config.max_model_len = 100
        req = GenerationRequest()
        assert engine._validate_prompt(req, 100) is False
        assert req.status == RequestStatus.FAILED
        assert "too long" in req.error.lower()

    def test_validate_prompt_exactly_at_limit(self, engine):
        engine.model_config.max_model_len = 100
        req = GenerationRequest()
        # prompt_len >= max_model_len should fail
        assert engine._validate_prompt(req, 100) is False


# ─── KV Cache Allocation ───────────────────────────────────────────────────


class TestKVCacheAllocation:
    def test_allocate_success(self, engine):
        mock_kv = MagicMock()
        mock_kv.can_allocate.return_value = True
        engine.kv_cache_manager = mock_kv

        req = GenerationRequest()
        result = engine._allocate_kv_cache(req, 100, 50)
        assert result is True
        mock_kv.allocate_sequence.assert_called_once()

    def test_allocate_failure(self, engine):
        mock_kv = MagicMock()
        mock_kv.can_allocate.return_value = False
        engine.kv_cache_manager = mock_kv

        req = GenerationRequest()
        result = engine._allocate_kv_cache(req, 100, 50)
        assert result is False
        assert req.status == RequestStatus.FAILED
        assert "insufficient" in req.error.lower()


# ─── Generator ──────────────────────────────────────────────────────────────


class TestMakeGenerator:
    def test_no_seed_returns_none(self, engine):
        params = SamplingParams(seed=None)
        assert engine._make_generator(params, torch.device("cpu")) is None

    def test_with_seed_returns_generator(self, engine):
        params = SamplingParams(seed=42)
        gen = engine._make_generator(params, torch.device("cpu"))
        assert isinstance(gen, torch.Generator)


# ─── Stop Conditions ───────────────────────────────────────────────────────


class TestStopConditions:
    def test_with_eos_token(self, engine):
        engine.tokenizer = _make_mock_tokenizer()
        engine.tokenizer.eos_token_id = 50256
        params = SamplingParams()
        stop_ids, stop_strings = engine._get_stop_conditions(params)
        assert 50256 in stop_ids
        assert stop_strings == []

    def test_without_eos_token(self, engine):
        engine.tokenizer = _make_mock_tokenizer()
        engine.tokenizer.eos_token_id = None
        params = SamplingParams()
        stop_ids, stop_strings = engine._get_stop_conditions(params)
        assert len(stop_ids) == 0

    def test_with_stop_strings(self, engine):
        engine.tokenizer = _make_mock_tokenizer()
        params = SamplingParams(stop=["<|end|>", "\n\n"])
        _, stop_strings = engine._get_stop_conditions(params)
        assert stop_strings == ["<|end|>", "\n\n"]


# ─── Generate on unready engine ─────────────────────────────────────────────


class TestGenerateUnready:
    def test_generate_fails_when_not_ready(self, engine):
        req = GenerationRequest(prompt="hello")
        result = engine.generate(req)
        assert result.status == RequestStatus.FAILED
        assert "not ready" in result.error.lower()


# ─── Stream token emission ──────────────────────────────────────────────────


class TestEmitStreamToken:
    def test_token_callback_path(self, engine):
        """_token_callback is the preferred path (sends raw token IDs)."""
        received = []

        def cb(token_id, finished):
            received.append((token_id, finished))

        req = GenerationRequest(output_token_ids=[10, 20, 30])
        req._token_callback = cb
        engine._emit_stream_token(req)

        assert received == [(30, False)]

    def test_no_callback_noop(self, engine):
        """No callbacks set should not crash."""
        req = GenerationRequest(output_token_ids=[10])
        engine._emit_stream_token(req)  # Should not raise

    def test_stream_callback_path(self, engine):
        """_stream_callback gets decoded text deltas."""
        engine.tokenizer = _make_mock_tokenizer()
        engine.tokenizer.decode.return_value = "Hello world"

        received = []

        def cb(text, finished):
            received.append((text, finished))

        req = GenerationRequest(
            prompt_token_ids=[1, 2],
            output_token_ids=[3],
        )
        req._stream_callback = cb
        req._stream_text_cursor = 5  # "Hello" already sent

        engine._emit_stream_token(req)

        # Should have sent " world" (chars 5-11)
        assert len(received) == 1
        assert received[0][0] == " world"
        assert received[0][1] is False


# ─── Resolve device ─────────────────────────────────────────────────────────


class TestResolveDevice:
    def test_resolve_from_model_params(self, engine):
        engine.model = _make_mock_model()
        device = engine._resolve_device()
        assert isinstance(device, torch.device)

    def test_resolve_fallback_no_params(self, engine):
        """Model with no parameters should fall back to cpu/cuda detection."""
        engine.model = MagicMock()
        engine.model.parameters.return_value = iter([])  # Empty
        device = engine._resolve_device()
        assert isinstance(device, torch.device)
