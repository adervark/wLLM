"""Model and tokenizer loading with quantization and multi-GPU support."""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import ModelConfig, QuantizationType, KVCacheConfig
from .device import get_all_gpu_memory_info, get_gpu_memory_info, get_aggregate_gpu_memory

logger = logging.getLogger(__name__)


def _build_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Build bitsandbytes quantization config."""
    if model_config.quantization == QuantizationType.NF4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_config.torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif model_config.quantization == QuantizationType.INT8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


# GPU memory functions are now in device.py — re-exported here for backward compatibility.
# get_gpu_memory_info and get_aggregate_gpu_memory are imported above from .device


def _extract_model_kv_params(model: PreTrainedModel) -> dict:
    """Extract KV cache dimensions from a loaded model's config."""
    config = model.config
    result = {}

    # Number of decoder layers
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        val = getattr(config, attr, None)
        if val is not None:
            result["num_layers"] = val
            break

    # Number of KV heads (may differ from attention heads in GQA)
    for attr in ("num_key_value_heads", "num_kv_heads"):
        val = getattr(config, attr, None)
        if val is not None:
            result["num_kv_heads"] = val
            break
    else:
        # Fallback to full attention heads
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            val = getattr(config, attr, None)
            if val is not None:
                result["num_kv_heads"] = val
                break

    # Head dimension
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_heads = getattr(config, "num_attention_heads", None)
        if hidden_size and num_heads:
            head_dim = hidden_size // num_heads
    if head_dim:
        result["head_dim"] = head_dim

    return result


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

        # --- Load tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # --- Build quantization config ---
        quantization_config = _build_quantization_config(self.config)

        # --- Resolve device map ---
        device_map = self._resolve_device_map()

        # --- Load model ---
        load_kwargs: dict = {
            "pretrained_model_name_or_path": self.config.model_name_or_path,
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": device_map,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = self.config.torch_dtype

        # Tensor parallelism (requires transformers >= 4.45)
        if self.config.tensor_parallel_size > 1:
            try:
                load_kwargs["tp_plan"] = "auto"
                logger.info("Tensor parallelism enabled: %d GPUs", self.config.tensor_parallel_size)
            except Exception as e:
                logger.warning("Tensor parallelism not supported: %s", e)

        # CPU offload
        if self.config.cpu_offload:
            load_kwargs["offload_folder"] = "offload_weights"
            logger.info("CPU offload enabled — excess layers will spill to RAM")

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()

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
        kv_params = _extract_model_kv_params(self.model)
        logger.info("Model KV params: %s", kv_params)

        # --- Load draft model if specified ---
        if self.config.draft_model_name_or_path:
            logger.info("Loading draft model '%s' for speculative decoding", self.config.draft_model_name_or_path)
            # Draft models are usually loaded without quantization and on the same device
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.config.draft_model_name_or_path,
                torch_dtype=self.config.torch_dtype,
                device_map=device_map,
                trust_remote_code=self.config.trust_remote_code
            )
            self.draft_model.eval()

        return self.model, self.tokenizer

    def get_kv_cache_params(self) -> dict:
        """Get KV cache dimensions from the loaded model."""
        if self.model is None:
            return {}
        return _extract_model_kv_params(self.model)

    def _resolve_device_map(self) -> str | dict:
        """Determine the device_map argument for from_pretrained."""
        device = self.config.device

        if device == "cpu":
            return "cpu"
        elif device == "auto":
            return self.config.device_map_strategy
        elif device.startswith("cuda:"):
            # Specific GPU
            return {"": device}
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

        logger.info("Model unloaded. GPU memory: %s", get_aggregate_gpu_memory())
