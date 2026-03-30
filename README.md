# WinLLM 

A Windows-native LLM inference and serving engine inspired by [vLLM](https://github.com/vllm-project/vllm).

Built with PyTorch + CUDA + HuggingFace Transformers. Provides an OpenAI-compatible API server with quantization support for efficient inference on consumer GPUs.

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions` and `/v1/completions`
- **Multi-backend inference** — PyTorch (default), ONNX Runtime, and DirectML
- **4-bit / 8-bit quantization** via bitsandbytes (NF4, INT8), plus AWQ and GPTQ support
- **KV cache management** — memory-aware block scheduling prevents OOM
- **Streaming** — Server-Sent Events for real-time token output
- **Continuous batching** — serves multiple concurrent requests with iteration-level scheduling
- **Speculative decoding** — accelerates generation using small draft models
- **Prefix caching** — reuses KV cache for shared prompt prefixes across requests
- **Interactive CLI** — chat directly in your terminal

## Why wLLM? (vs. Ollama)

While tools like Ollama are fantastic for consumer-friendly local generation using `.gguf` archives, **wLLM is engineered for developers, rapid prototyping, and highly concurrent API serving**:

- **Zero-Day Model Support**: Instead of waiting for community quantizations or manually converting models to `.gguf`, wLLM uses HuggingFace natively. Any new PyTorch model uploaded to the hub can be served instantly.
- **Infinitely Hackable**: Written entirely in pure Python and PyTorch. If you want to modify the batching scheduler, add new logit processors, or inject memory optimizations, you don't need to write C/C++ or recompile `llama.cpp`.
- **Server-Grade Continuous Batching**: Built as a Windows-native, lightweight equivalent to vLLM. It features iteration-level continuous batching, O(1) memory pool tracking, zero-fragmentation tensor batching, and decoupled async generators, designed to handle concurrent API requests on consumer GPUs without hanging.
- **Out-of-the-Box Speculative Decoding**: Accelerate massive models by leveraging smaller draft models to verify tokens in a single forward pass, integrated natively within PyTorch.
- **Multi-Backend Flexibility**: Choose between PyTorch, ONNX Runtime, or DirectML for the inference backend that best suits your hardware and model.

## Quick Start

### Install

**The Easiest Way (Windows):**
Simply double-click the `install.bat` file in the root directory. It will automatically detect/create a virtual environment, install PyTorch with CUDA 12.4 support, and install the wLLM engine.

**Manual Installation (using `uv`):**
```bash
# Create a virtual environment
uv venv .venv
.venv\Scripts\activate

# Install with CUDA support
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124
```

### Chat in Terminal

```bash
wllm chat --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit
```

### Start API Server

```bash
wllm serve --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit --port 8000
```

### Send a Request

```bash
curl http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"Phi-3-mini\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}"
```

### Run Benchmark

```bash
wllm benchmark --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit
```

## Supported Models

Any HuggingFace `AutoModelForCausalLM` model works. Recommended for 8GB VRAM:

| Model | Size | Quantization | VRAM Usage |
|-------|------|-------------|------------|
| Phi-3-mini-4k-instruct | 3.8B | float16 | ~7.5 GB |
| Phi-3-mini-4k-instruct | 3.8B | 4bit | ~2.5 GB |
| Llama-3.2-3B-Instruct | 3B | 4bit | ~2 GB |
| Mistral-7B-Instruct-v0.3 | 7B | 4bit | ~4.5 GB |
| Qwen2.5-7B-Instruct | 7B | 4bit | ~4.5 GB |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List loaded model |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/completions` | POST | Text completion |
| `/health` | GET | Server health & GPU stats |

## Documentation

WinLLM comes with comprehensive documentation optimized for reading on GitHub or as an Obsidian vault:

- **[Architecture Details](documentation/Architecture.md)** — A visual guide to the software architecture and internal workflows.
- **[Commands Reference](documentation/COMMANDS.md)** — Detailed CLI usage, options, and auto-configuration guide.
- **[Complete Walkthrough](documentation/WALKTHROUGH.md)** — From installation to architecture deep-dive.
- **[Deep Dive (Genesys)](documentation/Genesys.md)** — First-principles guide to how LLM inference works.
- **[Changelog](documentation/CHANGELOG.md)** — Version history and release notes.

## Architecture

```
Request → API Server → Scheduler (Waiting Queue)
                           ↓
                   Inference Loop (Continuous Batching)
                           ↕               ↕
                   KV Cache Manager   Speculative Engine
                           ↓
                    BackendFactory → Model (PyTorch / ONNX / DirectML)
```

## Testing

The test suite includes **231 tests** covering all core modules:

```bash
# Using uv (recommended)
uv run pytest tests/ -v

# Or using the built-in batch file
setup_and_test.bat
```

Individual component tests can be run via:
- `pytest tests/test_engine.py` — Core inference engine
- `pytest tests/test_scheduler.py` — Request scheduling & prefix hashing
- `pytest tests/test_kv_cache.py` — KV cache block management & prefix caching
- `pytest tests/test_sampler.py` — Token sampling pipeline
- `pytest tests/test_backend.py` — Multi-backend model loading
- `pytest tests/test_speculative.py` — Speculative decoding
- `pytest tests/test_types.py` — Request types & lifecycle
- `pytest tests/test_api_server.py` — API response format validation
- `pytest tests/test_config.py` — Config validation & hardware defaults
- `pytest tests/test_device.py` — Hardware detection & profiling
- `pytest tests/test_registry.py` — Model family identification
- `pytest tests/test_cli.py` — CLI argument parsing
- `pytest tests/test_utils.py` — Chat prompt formatting

## License

MIT
