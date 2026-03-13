"""OpenAI-compatible REST API server for WinLLM."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .config import ModelConfig, SamplingParams, SchedulerConfig, ServerConfig, KVCacheConfig
from .engine import GenerationRequest, InferenceEngine
from .model_loader import get_gpu_memory_info
from .scheduler import Scheduler
from .utils import format_chat_prompt

logger = logging.getLogger(__name__)


# ─── Pydantic request/response models (OpenAI format) ─────────────────────


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


# ─── Server setup ─────────────────────────────────────────────────────────


def create_app(
    model_config: ModelConfig,
    server_config: Optional[ServerConfig] = None,
    scheduler_config: Optional[SchedulerConfig] = None,
    kv_cache_config: Optional[KVCacheConfig] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    server_config = server_config or ServerConfig()
    scheduler_config = scheduler_config or SchedulerConfig()
    kv_cache_config = kv_cache_config or KVCacheConfig()

    # --- State (created early so lifespan can reference them) ---
    engine = InferenceEngine(model_config, kv_cache_config)
    scheduler = Scheduler(engine, scheduler_config)
    model_display_name = server_config.model_alias or model_config.model_name_or_path

    # --- Lifespan (replaces deprecated @app.on_event) ---

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info("Starting WinLLM server...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, engine.load_model)
        logger.info("Server ready! Model: %s", model_display_name)
        yield
        # Shutdown
        logger.info("Shutting down WinLLM server...")
        engine.unload_model()

    app = FastAPI(
        title="WinLLM",
        description="Windows-native LLM inference engine — OpenAI-compatible API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Helper ---

    def _build_sampling_params(
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[list[str]],
        repetition_penalty: float,
        seed: Optional[int],
    ) -> SamplingParams:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

    def _format_chat_prompt(messages: list[ChatMessage]) -> str:
        """Format chat messages into a prompt string."""
        if engine.tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return format_chat_prompt(engine.tokenizer, messages)

    # --- Routes ---

    @app.get("/v1/models")
    async def list_models():
        return ModelListResponse(
            data=[ModelInfo(id=model_display_name, created=int(time.time()))]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        prompt = _format_chat_prompt(req.messages)
        sampling_params = _build_sampling_params(
            req.temperature, req.top_p, req.max_tokens,
            req.stop, req.repetition_penalty, req.seed,
        )

        gen_request = GenerationRequest(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        if req.stream:
            return EventSourceResponse(
                _stream_response(gen_request, model_display_name, "chat")
            )

        # Non-streaming
        result = await scheduler.submit(gen_request)

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        return ChatCompletionResponse(
            id=f"chatcmpl-{result.request_id}",
            created=int(result.created_at),
            model=model_display_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=result.output_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(result.prompt_token_ids),
                completion_tokens=result.generation_tokens,
                total_tokens=result.total_tokens,
            ),
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        sampling_params = _build_sampling_params(
            req.temperature, req.top_p, req.max_tokens,
            req.stop, req.repetition_penalty, req.seed,
        )

        gen_request = GenerationRequest(
            prompt=req.prompt,
            sampling_params=sampling_params,
        )

        if req.stream:
            return EventSourceResponse(
                _stream_response(gen_request, model_display_name, "completion")
            )

        result = await scheduler.submit(gen_request)

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        return CompletionResponse(
            id=f"cmpl-{result.request_id}",
            created=int(result.created_at),
            model=model_display_name,
            choices=[
                CompletionChoice(
                    text=result.output_text,
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(result.prompt_token_ids),
                completion_tokens=result.generation_tokens,
                total_tokens=result.total_tokens,
            ),
        )

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if engine.is_ready else "loading",
            "model": model_display_name,
            "gpu": get_gpu_memory_info(),
            "scheduler": scheduler.get_status(),
        }

    # --- Streaming helpers ---

    async def _stream_response(
        gen_request: GenerationRequest,
        model_name: str,
        response_type: str,
    ):
        """Unified SSE stream for both chat and text completions.

        Uses asyncio.Queue with call_soon_threadsafe for a proper
        async/sync bridge. Propagates errors as SSE error events.

        Args:
            response_type: "chat" for chat completions, "completion" for text completions.
        """
        import json

        stream_timeout = server_config.stream_token_timeout
        is_chat = response_type == "chat"
        id_prefix = "chatcmpl" if is_chat else "cmpl"
        object_type = "chat.completion.chunk" if is_chat else "text_completion"
        request_id = f"{id_prefix}-{gen_request.request_id}"

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[tuple[str, bool] | BaseException] = asyncio.Queue()

        def callback(text: str, finished: bool):
            loop.call_soon_threadsafe(token_queue.put_nowait, (text, finished))

        gen_request._stream_callback = callback

        def _run_generate():
            try:
                engine.generate(gen_request)
            except BaseException as exc:
                loop.call_soon_threadsafe(token_queue.put_nowait, exc)

        # Start generation in background
        gen_task = loop.run_in_executor(None, _run_generate)

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        token_queue.get(), timeout=stream_timeout
                    )
                except asyncio.TimeoutError:
                    # Token generation stalled — cancel and notify client
                    gen_request.cancel()
                    error_chunk = {
                        "id": request_id,
                        "object": object_type,
                        "created": int(time.time()),
                        "model": model_name,
                        "error": {
                            "message": f"Token generation timed out after {stream_timeout}s",
                            "type": "timeout",
                        },
                    }
                    yield {"data": json.dumps(error_chunk)}
                    yield {"data": "[DONE]"}
                    break

                # If the generation thread raised, send an error event
                if isinstance(item, BaseException):
                    error_chunk = {
                        "id": request_id,
                        "object": object_type,
                        "created": int(time.time()),
                        "model": model_name,
                        "error": {
                            "message": str(item),
                            "type": type(item).__name__,
                        },
                    }
                    yield {"data": json.dumps(error_chunk)}
                    yield {"data": "[DONE]"}
                    break

                text, finished = item
                if finished:
                    # Final chunk with finish_reason
                    choice = {"index": 0, "finish_reason": "stop"}
                    choice["delta" if is_chat else "text"] = {} if is_chat else ""
                    chunk = {
                        "id": request_id,
                        "object": object_type,
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [choice],
                    }
                    yield {"data": json.dumps(chunk)}
                    yield {"data": "[DONE]"}
                    break
                else:
                    choice = {"index": 0, "finish_reason": None}
                    if is_chat:
                        choice["delta"] = {"content": text}
                    else:
                        choice["text"] = text
                    chunk = {
                        "id": request_id,
                        "object": object_type,
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [choice],
                    }
                    yield {"data": json.dumps(chunk)}
        finally:
            await gen_task

    return app
