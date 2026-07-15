"""Quantization configuration builders for HuggingFace Transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..config import QuantizationType

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)

# VRAM held back for KV cache and activations when judging whether
# full-precision weights fit.
_AUTO_HEADROOM_GB = 1.5

# Weights are loaded in fp16/bf16 regardless of checkpoint dtype.
_LOADED_BYTES_PER_PARAM = 2

_WEIGHT_FILE_PATTERNS = ("*.safetensors", "*.bin")


def resolve_auto_quantization(model_config: "ModelConfig") -> QuantizationType:
    """Resolve QuantizationType.AUTO to a concrete method.

    bnb 4-bit is a memory tool, not a speed tool: it roughly doubles load
    time and costs ~35% decode throughput, paying off only when the model
    would otherwise not fit. So AUTO quantizes only when the estimated
    fp16/bf16 weight size (plus KV/activation headroom) exceeds free VRAM,
    and falls back to 4-bit when the size can't be estimated.
    """
    import torch

    # bitsandbytes requires CUDA. The device_count check matters:
    # with CUDA_VISIBLE_DEVICES="" is_available() can still be True here.
    if (
        model_config.device == "cpu"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() == 0
    ):
        logger.info("Auto quantization: no CUDA device -> none")
        return QuantizationType.NONE

    from ..hardware.memory import get_aggregate_gpu_memory

    free_gb = get_aggregate_gpu_memory()["free"]
    budget_gb = free_gb * model_config.gpu_memory_utilization - _AUTO_HEADROOM_GB
    weight_gb = estimate_weight_gb(model_config.model_name_or_path)

    if weight_gb is None:
        logger.info(
            "Auto quantization: weight size unknown -> 4bit (conservative)"
        )
        return QuantizationType.NF4
    if weight_gb <= budget_gb:
        logger.info(
            "Auto quantization: ~%.1f GB weights fit in %.1f GB budget -> none",
            weight_gb, budget_gb,
        )
        return QuantizationType.NONE
    logger.info(
        "Auto quantization: ~%.1f GB weights exceed %.1f GB budget -> 4bit",
        weight_gb, budget_gb,
    )
    return QuantizationType.NF4


def estimate_weight_gb(model_name_or_path: str) -> Optional[float]:
    """Estimate loaded weight size in GB, or None if it can't be determined.

    Sources, in order: a local model directory, the local HF hub cache,
    then the Hub's safetensors metadata (one small request — acceptable
    given the weights themselves are about to be fetched). Checkpoint file
    sizes stand in for loaded size, which overestimates fp32 checkpoints;
    that errs toward quantizing, never toward OOM.
    """
    path = Path(model_name_or_path)
    if path.is_dir():
        return _dir_weight_gb(path)

    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            model_name_or_path,
            local_files_only=True,
            allow_patterns=list(_WEIGHT_FILE_PATTERNS),
        )
        size = _dir_weight_gb(Path(snapshot))
        if size is not None:
            return size
    except Exception:
        pass

    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(model_name_or_path)
        params = sum(meta.parameter_count.values())
        if params:
            return params * _LOADED_BYTES_PER_PARAM / 1024**3
    except Exception:
        pass
    return None


def _dir_weight_gb(path: Path) -> Optional[float]:
    total = sum(
        f.stat().st_size
        for pattern in _WEIGHT_FILE_PATTERNS
        for f in path.glob(pattern)
    )
    return total / 1024**3 if total else None


def build_quantization_config(model_config: "ModelConfig"):
    """Build the appropriate quantization configuration for HF Transformers.

    Handles all supported quantization methods:
      - NF4 (4-bit via bitsandbytes with double quantization)
      - INT8 (8-bit via bitsandbytes)
      - GPTQ (4-bit, uses exllama kernel)
      - AWQ (4-bit with fused attention)

    Returns None for unquantized models.
    """
    from transformers import BitsAndBytesConfig

    if model_config.quantization == QuantizationType.NF4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_config.torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if model_config.quantization == QuantizationType.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)

    if model_config.quantization == QuantizationType.GPTQ:
        from transformers import GPTQConfig
        return GPTQConfig(bits=4, disable_exllama=False)

    if model_config.quantization == QuantizationType.AWQ:
        from transformers import AwqConfig
        return AwqConfig(bits=4, fuse_max_seq_len=model_config.max_model_len, do_fuse=True)

    return None
