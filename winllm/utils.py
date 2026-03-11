"""Shared utility functions for WinLLM."""

from __future__ import annotations

from typing import Union


def format_chat_prompt(
    tokenizer,
    messages: list[dict[str, str]],
) -> str:
    """Format chat messages into a prompt string.

    Uses the tokenizer's chat template if available, otherwise
    falls back to a simple role-prefixed format.

    Args:
        tokenizer: A HuggingFace tokenizer instance.
        messages: List of message dicts with "role" and "content" keys.
                  Also accepts Pydantic models with .role / .content attrs.

    Returns:
        Formatted prompt string ready for model input.
    """
    # Normalise to plain dicts (handles Pydantic ChatMessage objects)
    message_dicts = []
    for m in messages:
        if isinstance(m, dict):
            message_dicts.append(m)
        else:
            message_dicts.append({"role": m.role, "content": m.content})

    # Try the tokenizer's built-in chat template first
    try:
        return tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        pass

    # Fallback: simple role-prefixed format
    parts = []
    for msg in message_dicts:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)
