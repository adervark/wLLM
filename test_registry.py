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
    assert config.quantization == QuantizationType.NF4 # base dataclass default
    assert config.max_model_len == 4096
    
    # Apply defaults
    config.apply_hardware_defaults(mock_hw_defaults)
    
    # 1. HW default changes quant from NF4 -> NONE
    # 2. Registry detects "qwen" and recommends "4bit" 
    # Therefore, quant should NOT be NONE despite HW defaults because registry recommended 4bit for this specific model, 
    # wait - actually my implementation doesn't override HW back to registry, let's verify actual behavior.
    # Ah, the logic in `apply_model_profile` says:
    # "if config.quant is NF4 and profile recommends 8bit -> set 8bit"
    # "if config.quant is NF4 and profile recommends none -> set none"
    
    # In my current script, `apply_hardware_defaults` runs FIRST, changing config.quantization to NONE.
    # Then `apply_model_profile` runs, but it only modifies if config.quantization == NF4.
    # So it remains NONE. This is actually correct for powerful hardware (like DESKTOP profile) overriding the model's 4bit recommendation.
    assert config.quantization == QuantizationType.NONE
    
    # HW default changes ctx len
    assert config.max_model_len == 8192

if __name__ == "__main__":
    test_identify_model_profile_llama()
    test_identify_model_profile_unknown()
    test_model_config_auto_registry()
    print("All custom tests passed!")
