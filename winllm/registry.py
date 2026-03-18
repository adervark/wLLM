"""Model registry and auto-configuration for plug-and-play inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelProfile:
    """Pre-tuned configuration profile for a known model family."""
    family: str
    recommended_quantization: str = "4bit"
    max_context_window: int = 8192
    rope_scaling: bool = False
    
    # Heuristics for auto-detecting family from repo ID
    match_keywords: list[str] = None
    
    def __post_init__(self):
        if self.match_keywords is None:
            self.match_keywords = [self.family.lower()]

# Define known model families and their optimal baseline parameters
KNOWN_MODELS: list[ModelProfile] = [
    ModelProfile(
        family="llama",
        match_keywords=["llama-3", "llama3", "llama-2", "llama"],
        recommended_quantization="4bit",
        max_context_window=8192,
        rope_scaling=True,
    ),
    ModelProfile(
        family="mistral",
        match_keywords=["mistral", "mixtral"],
        recommended_quantization="4bit",
        max_context_window=32768,
        rope_scaling=False,  # Mistral handles SWA natively
    ),
    ModelProfile(
        family="qwen",
        match_keywords=["qwen1.5", "qwen2", "qwen"],
        recommended_quantization="4bit",
        max_context_window=32768,
        rope_scaling=True,
    ),
    ModelProfile(
        family="gemma",
        match_keywords=["gemma"],
        recommended_quantization="4bit",  # Gemma can be sensitive to quantization, but 4bit is safest for VRAM
        max_context_window=8192,
        rope_scaling=False,
    )
]

def identify_model_profile(model_name_or_path: str) -> ModelProfile | None:
    """Attempt to identify the model family from the path or repo name."""
    name_lower = model_name_or_path.lower()
    
    for profile in KNOWN_MODELS:
        if any(keyword in name_lower for keyword in profile.match_keywords):
            logger.info("Auto-detected model family: %s", profile.family)
            return profile
            
    logger.info("Could not auto-detect model family; falling back to generic defaults.")
    return None

def apply_model_profile(config, profile: ModelProfile):
    """Apply the known profile defaults to the ModelConfig if not explicitly set."""
    from .config import QuantizationType
    
    # Only apply if user hasn't forced a specific quantization
    if config.quantization == QuantizationType.NF4 and profile.recommended_quantization == "8bit":
         config.quantization = QuantizationType.INT8
    elif config.quantization == QuantizationType.NF4 and profile.recommended_quantization == "none":
         config.quantization = QuantizationType.NONE
