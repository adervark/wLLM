"""OpenAI-compatible REST API server for WinLLM."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
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

    app = FastAPI(
        title="WinLLM",
        description="Windows-native LLM inference engine — OpenAI-compatible API",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- State ---
    engine = InferenceEngine(model_config, kv_cache_config)
    scheduler = Scheduler(engine, scheduler_config)
    model_display_name = server_config.model_alias or model_config.model_name_or_path

    # --- Startup / Shutdown ---

    @app.on_event("startup")
    async def startup():
        logger.info("Starting WinLLM server...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.load_model)
        logger.info("Server ready! Model: %s", model_display_name)

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down WinLLM server...")
        engine.unload_model()

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
        """Format chat messages into a prompt string.

        Uses the tokenizer's chat template if available, otherwise
        falls back to a simple format.
        """
        if engine.tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Try using the tokenizer's built-in chat template
        try:
            message_dicts = [{"role": m.role, "content": m.content} for m in messages]
            prompt = engine.tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception:
            # Fallback: simple format
            parts = []
            for msg in messages:
                if msg.role == "system":
                    parts.append(f"System: {msg.content}")
                elif msg.role == "user":
                    parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"Assistant: {msg.content}")
            parts.append("Assistant:")
            return "\n".join(parts)

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
                _stream_chat_response(gen_request, model_display_name)
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
                _stream_completion_response(gen_request, model_display_name)
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

    async def _stream_chat_response(gen_request: GenerationRequest, model_name: str):
        """SSE stream for chat completions."""
        import json
        import queue

        request_id = f"chatcmpl-{gen_request.request_id}"
        token_queue: queue.Queue[tuple[str, bool]] = queue.Queue()

        def callback(text: str, finished: bool):
            token_queue.put((text, finished))

        gen_request._stream_callback = callback

        # Start generation in background
        loop = asyncio.get_event_loop()
        gen_task = loop.run_in_executor(None, engine.generate, gen_request)

        while True:
            try:
                text, finished = token_queue.get(timeout=0.05)
                if finished:
                    # Send final chunk with finish_reason
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield {"data": json.dumps(chunk)}
                    yield {"data": "[DONE]"}
                    break
                else:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }],
                    }
                    yield {"data": json.dumps(chunk)}
            except queue.Empty:
                if gen_task.done():
                    yield {"data": "[DONE]"}
                    break
                await asyncio.sleep(0.01)

        await gen_task

    async def _stream_completion_response(gen_request: GenerationRequest, model_name: str):
        """SSE stream for text completions."""
        import json
        import queue

        request_id = f"cmpl-{gen_request.request_id}"
        token_queue: queue.Queue[tuple[str, bool]] = queue.Queue()

        def callback(text: str, finished: bool):
            token_queue.put((text, finished))

        gen_request._stream_callback = callback

        loop = asyncio.get_event_loop()
        gen_task = loop.run_in_executor(None, engine.generate, gen_request)

        while True:
            try:
                text, finished = token_queue.get(timeout=0.05)
                if finished:
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }],
                    }
                    yield {"data": json.dumps(chunk)}
                    yield {"data": "[DONE]"}
                    break
                else:
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "text": text,
                            "finish_reason": None,
                        }],
                    }
                    yield {"data": json.dumps(chunk)}
            except queue.Empty:
                if gen_task.done():
                    yield {"data": "[DONE]"}
                    break
                await asyncio.sleep(0.01)

        await gen_task

    return app
