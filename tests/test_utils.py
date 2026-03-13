"""Unit tests for shared utility functions."""

import pytest
from winllm.utils import format_chat_prompt


class _MockTokenizerWithTemplate:
    """Mock tokenizer with a working apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Simple template: join role:content lines
        parts = [msg["role"] + ": " + msg["content"] for msg in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


class _MockTokenizerNoTemplate:
    """Mock tokenizer that always fails on apply_chat_template."""

    def apply_chat_template(self, *args, **kwargs):
        raise Exception("No chat template available")


class _PydanticLikeMessage:
    """Simulates a Pydantic ChatMessage model with .role / .content attrs."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


class TestFormatChatPromptWithTemplate:

    def test_single_user_message(self):
        tok = _MockTokenizerWithTemplate()
        msgs = [{"role": "user", "content": "Hello"}]
        result = format_chat_prompt(tok, msgs)
        assert "user: Hello" in result
        assert "assistant:" in result

    def test_multi_turn(self):
        tok = _MockTokenizerWithTemplate()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = format_chat_prompt(tok, msgs)
        assert "system: You are helpful." in result
        assert "user: Hi" in result
        assert "assistant: Hello!" in result
        assert "user: How are you?" in result
        assert result.endswith("assistant:")


class TestFormatChatPromptFallback:

    def test_fallback_format(self):
        tok = _MockTokenizerNoTemplate()
        msgs = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]
        result = format_chat_prompt(tok, msgs)
        assert "System: Be brief." in result
        assert "User: Hello" in result
        assert result.endswith("Assistant:")

    def test_fallback_multi_turn(self):
        tok = _MockTokenizerNoTemplate()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey there!"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = format_chat_prompt(tok, msgs)
        lines = result.split("\n")
        assert lines[0] == "User: Hi"
        assert lines[1] == "Assistant: Hey there!"
        assert lines[2] == "User: What is 2+2?"
        assert lines[3] == "Assistant:"


class TestFormatChatPromptPydanticMessages:

    def test_pydantic_like_messages(self):
        tok = _MockTokenizerNoTemplate()
        msgs = [
            _PydanticLikeMessage("user", "Hello from pydantic"),
        ]
        result = format_chat_prompt(tok, msgs)
        assert "User: Hello from pydantic" in result
        assert result.endswith("Assistant:")

    def test_mixed_dict_and_pydantic(self):
        tok = _MockTokenizerNoTemplate()
        msgs = [
            {"role": "system", "content": "System msg"},
            _PydanticLikeMessage("user", "User msg"),
        ]
        result = format_chat_prompt(tok, msgs)
        assert "System: System msg" in result
        assert "User: User msg" in result


class TestFormatChatPromptEdgeCases:

    def test_empty_messages(self):
        tok = _MockTokenizerNoTemplate()
        result = format_chat_prompt(tok, [])
        assert result == "Assistant:"

    def test_empty_content(self):
        tok = _MockTokenizerNoTemplate()
        msgs = [{"role": "user", "content": ""}]
        result = format_chat_prompt(tok, msgs)
        assert "User: " in result
