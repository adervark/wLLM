# WinLLM: Complete Walkthrough & Guide

Welcome to **WinLLM**! This guide will walk you through what WinLLM is, how to get it running on your machine, and how to use it—whether you're on a standard laptop or a multi-GPU datacenter.

## 1. What is WinLLM?

WinLLM is a Windows-native Inference Engine for Large Language Models (LLMs). It allows you to run open-source AI models (like Llama 3, Mistral, Phi-3, or LiquidAI LFM) locally with an API that is compatible with OpenAI.

### Key Features:
- **Device-Agnostic**: It automatically detects your hardware (CPU, single GPU, or multiple GPUs) and optimizes settings to run efficiently.
- **OpenAI-Compatible API**: You can use WinLLM as a drop-in replacement for OpenAI in your existing apps.
- **Dynamic Auto-Tuning**: All settings (batch size, quantization, context length) are calculated mathematically from your actual VRAM.
- **Multi-Backend**: Supports **PyTorch** (default), **ONNX Runtime** (Windows-native acceleration), and **DirectML** (cross-vendor DX12 acceleration).
- **Performance Optimized**: Built-in support for **Continuous Batching**, **Speculative Decoding**, and **SDPA/Flash Attention**.

---

## 2. Getting Started: Setup and Installation

Follow these steps to get WinLLM running on your Windows machine.

### Prerequisites
- Python 3.10+ installed
- [uv](https://github.com/astral-sh/uv) package manager (Recommended for fast installations)
- Git (optional, for cloning the repo)

### Installation Steps

1. **Navigate to the directory**:
   ```cmd
   cd S:\Code\wLLM
   ```

2. **Create a virtual environment** to keep dependencies isolated:
   ```cmd
   uv venv
   ```

3. **Activate the virtual environment**:
   ```cmd
   .venv\Scripts\activate
   ```

4. **Install PyTorch with CUDA support** (for NVIDIA GPUs):
   ```cmd
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   > **Important**: You must install all three PyTorch packages (`torch`, `torchvision`, `torchaudio`) from the same CUDA index URL. Installing only `torch` and later adding other packages can silently downgrade your GPU-accelerated PyTorch to a CPU-only version.

   *(Note: If you are running on a CPU-only machine, standard PyTorch is fine, but GPU is highly recommended.)*

5. **Install WinLLM and its development dependencies**:
   ```cmd
   uv pip install -e .[dev]
   ```

6. **Verify the installation by running the tests**:
   ```cmd
   uv run pytest tests/ -v
   ```

### Optional: ONNX Runtime Backend

If you want to use the ONNX Runtime backend for Windows-native acceleration:
```cmd
uv pip install optimum onnxruntime-gpu
```

> **Note**: The ONNX Runtime GPU backend requires CUDA 12.x and cuDNN 9.x installed system-wide. Without these, it will fall back to CPU execution. For most users, the default PyTorch backend is recommended.

---

## 3. Running Models (The Easy Way)

WinLLM is designed to be "plug-and-play". You don't need to manually configure complicated settings—WinLLM does it for you based on your hardware!

### Step 1: Detect Your Hardware
Before running a model, you can check what hardware WinLLM detects on your system:
```bash
wllm detect
```
This will show your GPU(s), VRAM, compute capability, and the dynamically calculated defaults (quantization, batch size, context length, attention backend).

### Step 2: Start the Server (Serve a Model)
Here are a few ways to start running models depending on your machine:

**For a Laptop (e.g., RTX 4070 Mobile, 8GB VRAM)**
Runs a small, efficient model quantized to 4-bit to save memory.
```bash
wllm serve -m microsoft/Phi-3-mini-4k-instruct -q 4bit
```

**For a Desktop (e.g., RTX 4090, 24GB VRAM)**
Auto-detects optimal settings for a standard 8B size model.
```bash
wllm serve -m meta-llama/Llama-3.1-8B-Instruct --auto-config
```

**For Multi-GPU Datacenters**
Spreads the model across 4 GPUs using Tensor Parallelism (`-tp 4`).
```bash
wllm serve -m meta-llama/Llama-3.1-70B-Instruct -tp 4 --auto-config
```

**For Computers without a GPU (CPU-only)**
Forces the engine to run strictly on the CPU.
```bash
wllm serve -m microsoft/Phi-3-mini-4k-instruct --device cpu
```

**When the model is too big for your GPU**
Offloads layers to your system RAM/CPU.
```bash
wllm serve -m mistralai/Mistral-7B-v0.3 --cpu-offload -q none
```

**Small models (≤ 3B parameters)**
Skip quantization entirely for maximum speed — they fit in VRAM uncompressed.
```bash
wllm chat -m LiquidAI/LFM2.5-1.2B-Thinking -q none --trust-remote-code
```

---

## 4. Inference Backends

WinLLM supports three inference backends, selectable via `--backend`:

### PyTorch (Default)
The standard backend using HuggingFace Transformers. Supports all quantization methods (NF4, INT8, AWQ, GPTQ), multi-GPU sharding, tensor parallelism, and speculative decoding.
```bash
wllm chat -m meta-llama/Llama-3.1-8B-Instruct
```

### ONNX Runtime
Uses Optimum's `ORTModelForCausalLM` for hardware-optimized inference. Best for models that have been pre-exported to ONNX format (e.g., LiquidAI's `-ONNX` variants). Requires `optimum` and `onnxruntime-gpu`.
```bash
wllm chat -m LiquidAI/LFM2.5-1.2B-Thinking-ONNX --backend onnxruntime --trust-remote-code
```
> **Note**: When using ONNX with LiquidAI models, WinLLM automatically selects the correct pre-quantized binary (`model_q4.onnx`, `model_q8.onnx`, or `model.onnx`) based on your `--quantization` setting.

### DirectML
Uses `torch-directml` for cross-vendor GPU acceleration via DX12. Works with AMD, Intel, and NVIDIA GPUs. Requires `torch-directml`.
```bash
wllm chat -m microsoft/Phi-3-mini-4k-instruct --backend directml
```

---

## 5. Performance Optimizations

WinLLM includes several optimization strategies to maximize hardware utilization:

### A. Unquantized Inference (Small Models)
For models ≤ 3B parameters, skipping quantization (`-q none`) removes the overhead of BitsAndBytes dequantization and allows pure FP16 tensor operations. This often results in **2-5x faster** token generation.

### B. SDPA / Flash Attention (Attention Backend)
WinLLM automatically selects the optimal attention kernel:
- **`flash_attention_2`** on Ampere+ GPUs (compute capability ≥ 8.0)
- **`sdpa`** (Scaled Dot-Product Attention) on all other GPUs

Override with `--attention-backend sdpa` or `--attention-backend flash_attention_2`.

### C. Continuous Batching (High Throughput)
Unlike traditional servers that process one request at a time, WinLLM uses an **Iteration-Level Scheduler**.
- New requests can join the active batch instantly without waiting for the previous request to finish.
- Dramatically increases throughput when multiple users are connected.

### D. Speculative Decoding (Advanced Speedup)
Uses a smaller "draft" model to propose multiple tokens at once, which the larger "target" model verifies in a single step.
- **Enable with**: `--draft-model <path_to_smaller_model>`
- **Benefit**: Can double generation speed for single requests.

---

## 6. How WinLLM Works (Architecture)

If you're curious about what happens under the hood when you send a prompt, here is the simplified flow:

1. **Request arrives via FastAPI**. The user asks a question through the OpenAI-compatible endpoint.
2. **Scheduler**. Placed in an async queue to handle multiple users at once.
3. **Backend Factory**: Selects the inference backend (PyTorch, ONNX Runtime, or DirectML).
4. **Inference Engine**:
   - **ModelLoader → BackendFactory**: Loads the AI model weights onto your GPU(s) or CPU using the selected backend.
   - **KVCacheManager**: Manages VRAM memory for past tokens dynamically.
   - **Sampler**: Decides the next best word (token) using settings like Temperature and Top-P.
5. **Generation Loop**: Does the heavy mathematical lifting.
6. **Streaming Output**: Words are sent back to you piece by piece (via SSE streams) just like ChatGPT!

### Dynamic Defaults Explained
`device.py` detects your hardware and calculates optimal settings mathematically using `_build_defaults()` to prevent "Out Of Memory" (OOM) errors:

- **Quantization**: `4bit` if total VRAM < 16 GB, `none` otherwise
- **Batch size**: `max(1, int(total_vram_gb / 1.5))` — scales continuously with your GPU
- **Context length**: 2048 (< 12 GB) → 4096 (12–24 GB) → 8192 (≥ 24 GB)
- **Attention backend**: `flash_attention_2` on Ampere+ GPUs (compute ≥ 8.0), `sdpa` otherwise
- **Multi-GPU**: Auto-selects `balanced` device map and sets tensor parallelism to your GPU count

All defaults can be overridden with `WINLLM_*` environment variables (e.g., `WINLLM_MAX_BATCH_SIZE=16`).

---

## 7. Exploring the Codebase

If you want to contribute or modify WinLLM, here is a quick overview of the inner workings:

### Project Structure
```
S:\Code\wLLM\
+-- pyproject.toml
+-- README.md
+-- compile_onnx.py         # Standalone script to export models to ONNX format
+-- winllm/
|   +-- __init__.py / __main__.py
|   +-- config.py           # All config dataclasses + apply_hardware_defaults()
|   +-- device.py           # HW detection, dynamic allocation, env overrides
|   +-- registry.py         # Model family detection and auto-tuning
|   +-- model_loader.py     # Quantization, multi-GPU device_map, tensor parallelism
|   +-- backend.py          # Multi-backend factory (PyTorch / ONNX Runtime / DirectML)
|   +-- kv_cache.py         # Block-based KV cache allocation, prefix caching, memory tracking
|   +-- sampler.py          # Token sampling pipeline (temperature, top-k, top-p, rep penalty)
|   +-- engine.py           # Core inference: prefill, decode loop, streaming
|   +-- scheduler.py        # Async request queue, continuous batching, prefix caching
|   +-- api_server.py       # FastAPI OpenAI-compatible REST server
|   +-- speculative.py      # Speculative decoding (draft/verify/accept)
|   +-- cli.py              # CLI entry point: serve / chat / benchmark / list / detect / remove
|   +-- types.py            # Shared data types (GenerationRequest, RequestStatus)
|   +-- utils.py            # Chat prompt formatting utilities
|   +-- commands/           # CLI command implementations
|       +-- serve.py / chat.py / benchmark.py / list.py / detect.py / remove.py / common.py
+-- tests/                  # 226 unit tests covering all modules
|   +-- test_config.py      # Config validation, hardware defaults, user override preservation
|   +-- test_device.py      # Hardware detection, VRAM tiering, env variable overrides
|   +-- test_kv_cache.py    # Block allocation, prefix caching lifecycle, reset completeness
|   +-- test_registry.py    # Model family identification (llama, mistral, qwen, gemma)
|   +-- test_sampler.py     # Logit processing, full sampling pipeline, edge cases
|   +-- test_utils.py       # Chat prompt formatting (template + fallback paths)
|   +-- test_types.py       # GenerationRequest lifecycle, cancellation, timing properties
|   +-- test_engine.py      # InferenceEngine with mocked models (load/generate/stream)
|   +-- test_backend.py     # BackendFactory dispatch, tokenizer fallback, ONNX routing
|   +-- test_scheduler.py   # Prefix hashing, SchedulerStats calculations
|   +-- test_speculative.py # Speculative decoding: draft, verify, accept/reject, EOS
|   +-- test_api_server.py  # Pydantic models, OpenAI response format compliance
|   +-- test_cli.py         # CLI argument parsing, subcommand registration, choices
|   +-- benchmark_throughput.py  # Performance comparison tool
+-- documentation/
    +-- Architecture.md     # System architecture diagrams
    +-- COMMANDS.md         # CLI reference
    +-- WALKTHROUGH.md      # This file
    +-- CHANGELOG.md        # Version history
    +-- Genesys.md          # Project genesis and design philosophy (first-principles guide)
```

### Key Files in Detail

| File | Purpose |
|------|---------|
| `config.py` | Centralizes all dataclasses: `ModelConfig`, `SchedulerConfig`, `KVCacheConfig`, `SamplingParams`. Each has `apply_hardware_defaults()` to merge auto-detected settings. |
| `device.py` | Runs `DeviceInfo.detect()` and `_build_defaults()` to calculate optimal parameters from your actual VRAM. All defaults can be overridden via `WINLLM_*` env vars. |
| `registry.py` | Auto-detects model family (Llama, Mistral, Qwen, Gemma) and applies family-specific tuning (context window, RoPE scaling, quantization). |
| `model_loader.py` | Builds quantization configs (NF4, INT8, AWQ, GPTQ), handles multi-GPU distribution (`device_map`), CPU offloading, and model architecture introspection for KV cache sizing. |
| `backend.py` | `BackendFactory` — abstracts model loading across PyTorch, ONNX Runtime, and DirectML. Handles ONNX tokenizer fallback and LiquidAI auto-routing. |
| `kv_cache.py` | Calculates model-aware per-token cache cost, manages block allocation/extension/freeing, prefix cache promotion and matching for reusing shared prompts. |
| `engine.py` | Core inference loop: `generate_step()` for batched prefill/decode, decomposed into helpers (`_validate_prompt`, `_allocate_kv_cache`, `_run_decode_loop`, `_finalize_generation`). Handles streaming via `_stream_text_cursor`. |
| `scheduler.py` | Background inference loop with continuous batching. Dynamic KV admission control, prefix caching via `_get_prefix_hashes`, automatic `_evict_completed` to prevent memory leaks. |
| `speculative.py` | Three-phase speculative decoding: `_draft_proposals` generates candidates, `_verify_proposals` checks them, `_accept_or_reject` resolves mismatches with bonus sampling. |
| `types.py` | Core data types: `GenerationRequest` tracks request state, tokens, KV cache pointers (`_prefix_cache_token_len`, `_stream_text_cursor`, `_draft_past_key_values`), and `RequestStatus` enum. |
| `cli.py` | Parses CLI arguments and dispatches to command handlers in `commands/`. |
| `api_server.py` | FastAPI server with OpenAI-compatible endpoints, SSE streaming, Pydantic request/response models. |

Enjoy building locally and taking full control of your LLMs with **WinLLM**!
