# WinLLM 

A Windows-native LLM inference and serving engine inspired by [vLLM](https://github.com/vllm-project/vllm).

Built with PyTorch + CUDA + HuggingFace Transformers. Provides an OpenAI-compatible API server with quantization support for efficient inference on consumer GPUs.

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions` and `/v1/completions`
- **4-bit / 8-bit quantization** via bitsandbytes (NF4, INT8)
- **KV cache management** — memory-aware scheduling prevents OOM
- **Streaming** — Server-Sent Events for real-time token output
- **Continuous batching** — serves multiple concurrent requests with iteration-level scheduling
- **Speculative decoding** — accelerates generation using small draft models
- **torch.compile support** — reduces Python overhead via graph optimization
- **Interactive CLI** — chat directly in your terminal

## Quick Start

### Install

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
winllm chat --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit
```

### Start API Server

```bash
winllm serve --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit --port 8000
```

### Send a Request

```bash
curl http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"Phi-3-mini\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}"
```

### Run Benchmark

```bash
winllm benchmark --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit
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
- **[Deep Dive (Genesys)](documentation/Genesys.md)** — A first-principles guide explaining how LLM inference engines work under the hood.
- **[Commands Reference](documentation/COMMANDS.md)** — Detailed CLI usage, options, and auto-configuration guide.

## Architecture

Request → API Server → Scheduler (Waiting Queue)
                           ↓
                   Inference Loop (Continuous Batching)
                           ↕               ↕
                   KV Cache Manager   Speculative Engine
                           ↓
                    Model (PyTorch) ← [torch.compile]

## Testing

To run the full test suite and ensure everything works:

```bash
# Using uv (recommended)
uv run pytest tests/ -v

# Or using the built-in batch file
setup_and_test.bat
```

Individual component tests can be run via:
- `pytest tests/test_device.py` (Hardware detection)
- `pytest tests/test_sampler.py` (Logits processing)
- `pytest tests/test_kv_cache.py` (Memory management)

## License

MIT
