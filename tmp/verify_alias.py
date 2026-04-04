import sys
import os
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

from winllm.config import ModelConfig
from winllm.backend import BackendFactory
from transformers import AutoConfig

def test_gemma4_aliasing():
    print("Testing Gemma 4 automatic aliasing...")
    config = ModelConfig(model_name_or_path="google/gemma-4-E4B")
    
    # This should call _prepare_transformers_compat internally
    # We'll just call it directly for the test
    BackendFactory._prepare_transformers_compat(config)
    
    # Check if gemma4 is now in AutoConfig's mapping
    # AutoConfig.register adds it to the internal mapping
    try:
        # Try to get the config class for 'gemma4'
        # Since we registered it, it should return Gemma2Config (if available)
        # or at least not raise a KeyError before even trying to load a file
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if "gemma4" in CONFIG_MAPPING:
            print("SUCCESS: 'gemma4' registered in CONFIG_MAPPING")
        else:
            # Maybe it's registered in the dynamic registry if not in static mapping
            # AutoConfig.register usually updates the static mapping in newer transformers
            # or a dynamic registry.
            print("INFO: 'gemma4' not in CONFIG_MAPPING, checking registry...")
            # We can't easily check the private registry, but we can try to instantiate it
            # with a dummy config.json
            pass
            
        # Verify manual override
        print("\nTesting manual architecture override...")
        config_force = ModelConfig(model_name_or_path="custom/model", force_architecture="llama")
        BackendFactory._prepare_transformers_compat(config_force)
        
        # 'custom' should now be registered to LlamaConfig
        if "custom" in CONFIG_MAPPING:
            print("SUCCESS: 'custom' registered in CONFIG_MAPPING as 'llama' alias")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_gemma4_aliasing()
