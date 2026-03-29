"""Shared utilities for CLI commands."""

from __future__ import annotations

from ..config import QuantizationType, ModelConfig

# Shared quantization string → enum mapping
QUANT_MAP = {
    "auto": None,
    "none": QuantizationType.NONE,
    "4bit": QuantizationType.NF4,
    "8bit": QuantizationType.INT8,
    "awq": QuantizationType.AWQ,
    "gptq": QuantizationType.GPTQ,
}


def build_model_config(args) -> ModelConfig:
    """Build a ModelConfig from parsed CLI args (shared by all commands)."""
    quantization = QUANT_MAP.get(args.quantization)

    kwargs = {"model_name_or_path": args.model}
    if quantization is not None:
        kwargs["quantization"] = quantization
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if hasattr(args, "gpu_memory_utilization") and args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    if hasattr(args, "attention_backend") and args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if hasattr(args, "draft_model") and args.draft_model:
        kwargs["draft_model_name_or_path"] = args.draft_model

    kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    kwargs["device_map_strategy"] = args.device_map_strategy
    kwargs["cpu_offload"] = args.cpu_offload
    kwargs["device"] = args.device
    if hasattr(args, "backend"):
        kwargs["inference_backend"] = args.backend

    return ModelConfig(**kwargs)


def apply_auto_config(model_config, scheduler_config, kv_cache_config):
    """Detect hardware and apply optimal defaults."""
    from ..device import DeviceInfo
    hw = DeviceInfo.detect()

    print(f"\n  Detected: {hw.device_count} GPU(s), {hw.total_vram_gb} GB total VRAM")
    print(f"     Profile:  {hw.profile.value}")
    for gpu in hw.devices:
        print(f"     GPU {gpu.index}: {gpu.name} ({gpu.total_vram_gb} GB)")

    model_config.apply_hardware_defaults(hw.defaults)
    scheduler_config.apply_hardware_defaults(hw.defaults)
    kv_cache_config.apply_hardware_defaults(hw.defaults)

    return hw
