"""Unit tests for the FastAPI API server."""

import time
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from winllm.api_server import (
    ChatMessage,
    ChatCompletionRequest,
    CompletionRequest,
    UsageInfo,
    ChatCompletionChoice,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionResponse,
    ModelInfo,
    ModelListResponse,
)
from winllm.config import SamplingParams


# ─── Pydantic model tests ──────────────────────────────────────────────────


class TestChatMessage:
    def test_construction(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"


class TestChatCompletionRequest:
    def test_defaults(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")]
        )
        assert req.model == ""
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.max_tokens == 512
        assert req.stream is False
        assert req.stop is None
        assert req.repetition_penalty == 1.1
        assert req.seed is None

    def test_custom_values(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="test")],
            temperature=0.0,
            top_p=0.5,
            max_tokens=100,
            stream=True,
            stop=["<end>"],
            seed=42,
        )
        assert req.temperature == 0.0
        assert req.stream is True
        assert req.stop == ["<end>"]
        assert req.seed == 42


class TestCompletionRequest:
    def test_defaults(self):
        req = CompletionRequest(prompt="hello world")
        assert req.model == ""
        assert req.temperature == 0.7
        assert req.stream is False

    def test_custom(self):
        req = CompletionRequest(
            prompt="test", model="m", temperature=0.0, stream=True
        )
        assert req.model == "m"
        assert req.stream is True


class TestUsageInfo:
    def test_defaults(self):
        u = UsageInfo()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_custom(self):
        u = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.total_tokens == 30


class TestChatCompletionResponse:
    def test_serialization(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        d = resp.model_dump()
        assert d["id"] == "chatcmpl-123"
        assert d["object"] == "chat.completion"
        assert len(d["choices"]) == 1
        assert d["choices"][0]["message"]["role"] == "assistant"
        assert d["usage"]["total_tokens"] == 6


class TestCompletionResponse:
    def test_serialization(self):
        resp = CompletionResponse(
            id="cmpl-456",
            created=1234567890,
            model="test",
            choices=[CompletionChoice(text="hello world", finish_reason="stop")],
            usage=UsageInfo(),
        )
        d = resp.model_dump()
        assert d["object"] == "text_completion"
        assert d["choices"][0]["text"] == "hello world"


class TestModelInfo:
    def test_defaults(self):
        m = ModelInfo(id="test-model")
        assert m.object == "model"
        assert m.owned_by == "winllm"


class TestModelListResponse:
    def test_structure(self):
        resp = ModelListResponse(
            data=[ModelInfo(id="m1"), ModelInfo(id="m2")]
        )
        assert resp.object == "list"
        assert len(resp.data) == 2


# ─── Response format consistency ────────────────────────────────────────────


class TestOpenAICompatibility:
    """Verify response formats match OpenAI's API contract."""

    def test_chat_completion_response_has_required_fields(self):
        """OpenAI requires: id, object, created, model, choices, usage."""
        resp = ChatCompletionResponse(
            id="chatcmpl-test",
            created=int(time.time()),
            model="test",
            choices=[ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=""),
                finish_reason="stop",
            )],
            usage=UsageInfo(),
        )
        d = resp.model_dump()
        for key in ["id", "object", "created", "model", "choices", "usage"]:
            assert key in d, f"Missing required field: {key}"

    def test_completion_response_has_required_fields(self):
        resp = CompletionResponse(
            id="cmpl-test",
            created=int(time.time()),
            model="test",
            choices=[CompletionChoice(text="hi", finish_reason="stop")],
            usage=UsageInfo(),
        )
        d = resp.model_dump()
        for key in ["id", "object", "created", "model", "choices", "usage"]:
            assert key in d

    def test_chat_choice_has_message_and_finish_reason(self):
        choice = ChatCompletionChoice(
            message=ChatMessage(role="assistant", content="test"),
            finish_reason="length",
        )
        d = choice.model_dump()
        assert "message" in d
        assert "finish_reason" in d
        assert d["finish_reason"] == "length"

    def test_completion_choice_has_text_and_finish_reason(self):
        choice = CompletionChoice(text="output", finish_reason="stop")
        d = choice.model_dump()
        assert "text" in d
        assert "finish_reason" in d
