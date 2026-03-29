"""Backend abstraction layer for different inference engines."""

from __future__ import annotations
import logging
from typing import Optional, Any, Tuple
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

logger = logging.getLogger(__name__)

class BackendFactory:
    """Factory for loading models with different inference backends."""

    @staticmethod
    def _load_tokenizer(model_name_or_path: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code
            )
        except ValueError as e:
            if "TokenizersBackend" in str(e) and "-ONNX" in model_name_or_path:
                base_name = model_name_or_path.replace("-ONNX", "")
                logger.warning(
                    "Tokenizer loading failed due to Optimum 'TokenizersBackend' bug. "
                    "Falling back to base model: %s", base_name
                )
                return AutoTokenizer.from_pretrained(
                    base_name, trust_remote_code=trust_remote_code
                )
            raise

    @staticmethod
    def load(model_config: 'ModelConfig', **load_kwargs) -> Tuple[Any, PreTrainedTokenizerBase]:
        """Load a model using the specified backend."""
        backend = getattr(model_config, "inference_backend", "pytorch")

        if backend == "onnxruntime":
            return BackendFactory._load_onnxruntime(model_config, **load_kwargs)
        elif backend == "directml":
            return BackendFactory._load_directml(model_config, **load_kwargs)
        else:
            return BackendFactory._load_pytorch(model_config, **load_kwargs)

    @staticmethod
    def _load_pytorch(model_config, **load_kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Default PyTorch/Transformers loader."""
        from transformers import AutoModelForCausalLM
        
        tokenizer = BackendFactory._load_tokenizer(
            model_config.model_name_or_path,
            model_config.trust_remote_code
        )
        
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        return model, tokenizer

    @staticmethod
    def _load_onnxruntime(model_config, **load_kwargs) -> Tuple[Any, PreTrainedTokenizerBase]:
        """Load via Optimum ONNX Runtime (No MSVC/Triton required)."""
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
        except ImportError:
            raise ImportError(
                "ONNX Runtime backend requires 'optimum' and 'onnxruntime-gpu'. "
                "To fix this in your current environment, run:\n"
                "  uv pip install optimum onnxruntime-gpu"
            )

        logger.info("Loading model with ONNX Runtime backend (Windows-native acceleration)")
        
        tokenizer = BackendFactory._load_tokenizer(
            model_config.model_name_or_path,
            model_config.trust_remote_code
        )

        # Check if model is already an ONNX directory locally or an ONNX HF Repo
        import os
        is_exported = False
        if os.path.exists(model_config.model_name_or_path):
            is_exported = any(f.endswith(".onnx") for f in os.listdir(model_config.model_name_or_path))
        elif "ONNX" in model_config.model_name_or_path.upper():
            is_exported = True

        # ORT uses different kwargs, filter them
        ort_kwargs = {
            "model_id": model_config.model_name_or_path,
            "export": not is_exported,  # Don't export if it's already an ONNX folder or repo
            "trust_remote_code": model_config.trust_remote_code,
        }
        
        # Transparently handle LiquidAI's unique ONNX repository structure
        if is_exported and "LiquidAI" in model_config.model_name_or_path:
            ort_kwargs["subfolder"] = "onnx"
            # Map wLLM quantization arguments directly to the pre-compiled ONNX binaries!
            if getattr(model_config.quantization, "value", "none") == "4bit":
                ort_kwargs["file_name"] = "model_q4.onnx"
                logger.info("Auto-selected optimized INT4 (Q4) ONNX binary from LiquidAI repository.")
            elif getattr(model_config.quantization, "value", "none") == "8bit":
                ort_kwargs["file_name"] = "model_q8.onnx"
                logger.info("Auto-selected optimized INT8 (Q8) ONNX binary from LiquidAI repository.")
            else:
                ort_kwargs["file_name"] = "model.onnx"
                logger.info("Auto-selected standard FP32 ONNX binary from LiquidAI repository.")
        
        # Map device to ORT provider
        if model_config.device == "cpu":
            ort_kwargs["provider"] = "CPUExecutionProvider"
        else:
            ort_kwargs["provider"] = "CUDAExecutionProvider"

        model = ORTModelForCausalLM.from_pretrained(**ort_kwargs)
        return model, tokenizer

    @staticmethod
    def _load_directml(model_config, **load_kwargs) -> Tuple[Any, PreTrainedTokenizerBase]:
        """Load via torch-directml (Cross-vendor Windows acceleration)."""
        try:
            import torch_directml
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "DirectML backend requires 'torch-directml'. "
                "Install it with: pip install torch-directml"
            )

        logger.info("Loading model with DirectML backend (DX12 acceleration)")
        
        tokenizer = BackendFactory._load_tokenizer(
            model_config.model_name_or_path,
            model_config.trust_remote_code
        )
        
        # Load on CPU first then move to DML
        dml_device = torch_directml.device()
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=model_config.torch_dtype,
            trust_remote_code=model_config.trust_remote_code
        ).to(dml_device)
        
        return model, tokenizer
