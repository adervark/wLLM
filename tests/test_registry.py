"""Unit tests for config registry mapping."""
import pytest
from winllm.config import ModelConfig, QuantizationType
from winllm.registry import identify_model_profile

def test_identify_model_profile_llama():
    profile = identify_model_profile("meta-llama/Llama-2-7b-chat-hf")
    assert profile is not None
    assert profile.family == "llama"
    assert profile.recommended_quantization == "4bit"

def test_identify_model_profile_unknown():
    profile = identify_model_profile("some-unknown-model-123")
    assert profile is None

def test_model_config_auto_registry():
    # Simulate hardware detection defaults
    from winllm.device import HardwareDefaults
    mock_hw_defaults = HardwareDefaults(
        default_quantization="none",
        max_batch_size=8,
        max_model_len=8192,
        device_map_strategy="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        kv_cache_fraction=0.5,
    )
    
    # User just provided a path, no other args
    config = ModelConfig(
        model_name_or_path="Qwen/Qwen1.5-1.8B-Chat"
    )
    
    # Base instantiation
    assert config.quantization == QuantizationType.NF4
    assert config.max_model_len == 4096
    
    # Apply defaults
    config.apply_hardware_defaults(mock_hw_defaults)
    
    # HW default changes quant from NF4 -> NONE.
    # Registry detects "qwen" but apply_model_profile only overrides if quant is still NF4.
    # Since HW already changed it to NONE, the model profile does NOT override back.
    assert config.quantization == QuantizationType.NONE
    
    # HW default changes ctx len
    assert config.max_model_len == 8192


def test_identify_mistral():
    profile = identify_model_profile("mistralai/Mistral-7B-Instruct-v0.2")
    assert profile is not None
    assert profile.family == "mistral"


def test_identify_qwen():
    profile = identify_model_profile("Qwen/Qwen2-7B-Instruct")
    assert profile is not None
    assert profile.family == "qwen"


def test_identify_gemma():
    profile = identify_model_profile("google/gemma-2b")
    assert profile is not None
    assert profile.family == "gemma"
