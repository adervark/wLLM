"""Model and tokenizer loading orchestration.

The loader's single job is sequencing a load: build quantization config,
resolve the device map, hand off to the selected backend (DIP — it only
knows the InferenceBackend interface), and report memory usage. The
mechanics of each runtime live in ``winllm.backends``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..config import ModelConfig, QuantizationType
from ..hardware.memory import get_aggregate_gpu_memory
from .introspection import extract_kv_params
from .quantization import build_quantization_config, resolve_auto_quantization

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages a HuggingFace model with optional quantization.

    Supports single-GPU, multi-GPU (device_map sharding),
    tensor parallelism, and CPU offloading.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.draft_model: Optional[PreTrainedModel] = None

    def load(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model and tokenizer. Returns (model, tokenizer)."""
        # Resolve AUTO first so every consumer downstream (backends, logs,
        # the draft load) sees a concrete quantization method.
        if self.config.quantization == QuantizationType.AUTO:
            self.config.quantization = resolve_auto_quantization(self.config)

        logger.info(
            "Loading model '%s' (quantization=%s, dtype=%s, device_map=%s, tp=%d)",
            self.config.model_name_or_path,
            self.config.quantization.value,
            self.config.dtype.value,
            self.config.device_map_strategy,
            self.config.tensor_parallel_size,
        )

        mem_before = get_aggregate_gpu_memory()
        logger.info("GPU memory before loading: %s", mem_before)

        t0 = time.perf_counter()

        # --- Build quantization config ---
        quantization_config = build_quantization_config(self.config)

        # --- Resolve device map ---
        device_map = self._resolve_device_map()

        # --- Load model ---
        load_kwargs = self._build_load_kwargs(self.config, quantization_config, device_map)

        if self.config.tensor_parallel_size > 1:
            logger.info("Tensor parallelism enabled: %d GPUs", self.config.tensor_parallel_size)
        if self.config.cpu_offload:
            logger.info("CPU offload enabled -- excess layers will spill to RAM")

        from ..backends import get_backend
        backend = get_backend(self.config.inference_backend)
        self.model, self.tokenizer = backend.load(self.config, **load_kwargs)
        if hasattr(self.model, "eval"):
            self.model.eval()

        # Ensure pad_token is set (many models don't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        elapsed = time.perf_counter() - t0
        mem_after = get_aggregate_gpu_memory()

        logger.info("Model loaded in %.1fs", elapsed)
        logger.info("GPU memory after loading: %s", mem_after)
        logger.info(
            "Model VRAM usage: ~%.2f GB (across %d GPU(s))",
            mem_after["allocated"] - mem_before["allocated"],
            mem_after.get("device_count", 1),
        )

        # Extract model architecture info for KV cache estimation
        kv_params = extract_kv_params(self.model)
        logger.info("Model KV params: %s", kv_params)

        # --- Load draft model for speculative decoding (if specified) ---
        if self.config.draft_model_name_or_path:
            self._load_draft_model(backend, quantization_config, device_map)

        return self.model, self.tokenizer

    def _build_load_kwargs(self, config: ModelConfig, quantization_config, device_map) -> dict:
        """Assemble ``from_pretrained`` kwargs from a model config.

        Shared by the target and draft loads so the draft honors the same
        quantization, dtype, attention backend, and device placement.
        """
        load_kwargs: dict = {
            "pretrained_model_name_or_path": config.model_name_or_path,
            "trust_remote_code": config.trust_remote_code,
            "device_map": device_map,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = config.torch_dtype

        # Optimized Attention Backend (SDPA, Flash Attention 2)
        if config.attention_backend and config.attention_backend != "auto":
            load_kwargs["attn_implementation"] = config.attention_backend

        # Tensor parallelism (requires transformers >= 4.45)
        if config.tensor_parallel_size > 1:
            load_kwargs["tp_plan"] = "auto"

        # CPU offload for models that don't fit entirely in VRAM
        if config.cpu_offload:
            load_kwargs["offload_folder"] = "offload_weights"

        return load_kwargs

    def _load_draft_model(self, backend, quantization_config, device_map) -> None:
        """Load the speculative draft model through the same backend pipeline.

        The draft reuses the target's quantization, dtype, attention backend,
        and device map (so it isn't silently loaded full-precision/eager), but
        never shards via tensor parallelism — it's small by design.
        """
        draft_name = self.config.draft_model_name_or_path
        logger.info(
            "Loading draft model '%s' for speculative decoding via %s backend",
            draft_name, self.config.inference_backend,
        )
        draft_config = replace(
            self.config,
            model_name_or_path=draft_name,
            draft_model_name_or_path=None,
            tensor_parallel_size=1,
        )
        draft_kwargs = self._build_load_kwargs(draft_config, quantization_config, device_map)
        # The shared tokenizer is the target's; the draft's is discarded.
        self.draft_model, _ = backend.load(draft_config, **draft_kwargs)
        if hasattr(self.draft_model, "eval"):
            self.draft_model.eval()

    def get_kv_cache_params(self) -> dict:
        """Get KV cache dimensions from the loaded model."""
        if self.model is None:
            return {}
        return extract_kv_params(self.model)

    def _resolve_device_map(self) -> str | dict:
        """Determine the device_map argument for from_pretrained."""
        device = self.config.device

        if device == "cpu":
            return "cpu"
        elif device == "auto":
            return self.config.device_map_strategy
        elif device.startswith("cuda:"):
            return {"": device}  # Pin to a specific GPU
        else:
            return self.config.device_map_strategy

    def unload(self):
        """Unload model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.draft_model is not None:
            del self.draft_model
            self.draft_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

        logger.info("Model unloaded. GPU memory: %s", get_aggregate_gpu_memory())
