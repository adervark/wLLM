"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from .. import __version__
from ..config import KVCacheConfig, ModelConfig, SamplingParams, SchedulerConfig, ServerConfig
from ..core.types import GenerationRequest
from ..hardware.memory import get_gpu_memory_info
from ..inference.engine import InferenceEngine
from ..models.chat_template import format_chat_prompt
from ..scheduling.scheduler import Scheduler
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
from .metrics import PROMETHEUS_CONTENT_TYPE, render_prometheus
from .streaming import stream_completion

logger = logging.getLogger(__name__)


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
        version=__version__,
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

    # --- Helpers ---

    def _build_sampling_params(
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[list[str]],
        repetition_penalty: float,
        seed: Optional[int],
        response_format: Optional[dict] = None,
    ) -> SamplingParams:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
            repetition_penalty=repetition_penalty,
            seed=seed,
            response_format=response_format,
        )

    def _validate_response_format(response_format: Optional[dict]) -> Optional[dict]:
        """Vet a response_format dict; 400 on bad type, 400 if xgrammar missing."""
        if not response_format:
            return None
        rf_type = response_format.get("type")
        if rf_type in (None, "text"):
            return None
        if rf_type not in ("json_object", "json_schema"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format type: {rf_type!r}",
            )
        if rf_type == "json_schema":
            wrapper = response_format.get("json_schema") or {}
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("schema"), dict):
                raise HTTPException(
                    status_code=400,
                    detail="response_format.json_schema.schema must be a JSON schema object",
                )
        try:
            import xgrammar  # noqa: F401
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="Structured output requires xgrammar: pip install winllm[structured]",
            )
        return response_format

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
            response_format=_validate_response_format(req.response_format),
        )

        gen_request = GenerationRequest(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        if req.stream:
            return EventSourceResponse(
                stream_completion(
                    gen_request, model_display_name, "chat",
                    engine, scheduler, server_config.stream_token_timeout,
                )
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
                    finish_reason=result.finish_reason or "stop",
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
                stream_completion(
                    gen_request, model_display_name, "completion",
                    engine, scheduler, server_config.stream_token_timeout,
                )
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
                    finish_reason=result.finish_reason or "stop",
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

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            render_prometheus(scheduler), media_type=PROMETHEUS_CONTENT_TYPE
        )

    return app
