"""Pydantic request/response models (OpenAI API format)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[list[str]] = None
    repetition_penalty: float = 1.1
    seed: Optional[int] = None
    # {"type": "text" | "json_object" | "json_schema", ...} (OpenAI format)
    response_format: Optional[dict] = None


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[list[str]] = None
    repetition_penalty: float = 1.1
    seed: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "winllm"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
