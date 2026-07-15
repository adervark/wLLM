"""WinLLM — A Windows-native LLM inference and serving engine.

Public API is exposed lazily so that ``import winllm`` stays cheap
(no torch/transformers import until actually needed).
"""

__version__ = "1.0.1"

_LAZY_EXPORTS = {
    # Configuration
    "ModelConfig": "winllm.config",
    "SamplingParams": "winllm.config",
    "SchedulerConfig": "winllm.config",
    "ServerConfig": "winllm.config",
    "KVCacheConfig": "winllm.config",
    "QuantizationType": "winllm.config",
    "DType": "winllm.config",
    # Core types
    "GenerationRequest": "winllm.core",
    "RequestStatus": "winllm.core",
    # Engine / scheduling / serving
    "InferenceEngine": "winllm.inference",
    "Scheduler": "winllm.scheduling",
    "create_app": "winllm.server",
    # Hardware
    "DeviceInfo": "winllm.hardware",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'winllm' has no attribute '{name}'")
    import importlib
    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
