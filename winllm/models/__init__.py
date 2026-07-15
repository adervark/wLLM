"""Model loading and model-knowledge services.

  - loader.py:        orchestrates getting weights into memory
  - quantization.py:  builds HF quantization configs
  - introspection.py: extracts KV-cache dimensions from a loaded model
  - profiles.py:      known model-family heuristics
  - chat_template.py: chat-message → prompt formatting
"""

from .loader import ModelLoader
from .quantization import build_quantization_config
from .introspection import extract_kv_params
from .profiles import KNOWN_MODELS, ModelProfile, apply_model_profile, identify_model_profile
from .chat_template import format_chat_prompt

__all__ = [
    "ModelLoader",
    "build_quantization_config",
    "extract_kv_params",
    "ModelProfile",
    "KNOWN_MODELS",
    "identify_model_profile",
    "apply_model_profile",
    "format_chat_prompt",
]
