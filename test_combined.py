"""Combined tests for registry lookup and stream yielding simulations."""
import pytest
from transformers import AutoTokenizer
from winllm.config import ModelConfig, QuantizationType
from winllm.registry import identify_model_profile

# --- Registry Tests ---

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
    
    config = ModelConfig(
        model_name_or_path="Qwen/Qwen1.5-1.8B-Chat"
    )
    
    assert config.quantization == QuantizationType.NF4
    assert config.max_model_len == 4096
    
    config.apply_hardware_defaults(mock_hw_defaults)
    
    assert config.quantization == QuantizationType.NONE
    assert config.max_model_len == 8192

# --- Streaming Simulation ---

def run_stream_simulation():
    print("\n--- Running Stream Simulation ---")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
    tokens = tokenizer.encode('give it some space ', add_special_tokens=False)

    print("Tokens:", tokens)

    prefix_len = 0
    output_ids = []

    for token in tokens:
        output_ids.append(token)
        current_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        if len(current_text) > prefix_len:
            print(f"Yielding ({token}): `{current_text[prefix_len:]}` (Full text: `{current_text}`)")
            prefix_len = len(current_text)


if __name__ == "__main__":
    test_identify_model_profile_llama()
    test_identify_model_profile_unknown()
    test_model_config_auto_registry()
    print("All custom registry tests passed!")
    
    run_stream_simulation()
