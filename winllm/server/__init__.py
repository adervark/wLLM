"""OpenAI-compatible REST API server.

  - schemas.py:   request/response models (the wire contract)
  - streaming.py: SSE token streaming
  - app.py:       FastAPI application factory
"""

from .app import create_app
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)

__all__ = [
    "create_app",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    "ModelInfo",
    "ModelListResponse",
    "UsageInfo",
]
