# WinLLM: Complete Walkthrough & Guide

Welcome to **WinLLM**! This guide will walk you through what WinLLM is, how to get it running on your machine, and how to use it—whether you're on a standard laptop or a multi-GPU datacenter.

## 1. What is WinLLM?

WinLLM is a Windows-native Inference Engine for Large Language Models (LLMs). It allows you to run open-source AI models (like Llama 3, Mistral, or Phi-3) locally with an API that is compatible with OpenAI.

### Key Features:
- **Device-Agnostic**: It automatically detects your hardware (CPU, single GPU, or multiple GPUs) and optimizes settings to run efficiently.
- **OpenAI-Compatible API**: You can use WinLLM as a drop-in replacement for OpenAI in your existing apps.
- **Dynamic Auto-Tuning**: All settings (batch size, quantization, context length) are calculated mathematically from your actual VRAM.
- **Performance Optimized**: Built-in support for **Continuous Batching**, **Speculative Decoding**, and **Graph Compilation** (`torch.compile`).

---

## 1. Getting Started: Setup and Installation

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
   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   *(Note: If you are running on a CPU-only machine, standard PyTorch is fine, but GPU is highly recommended.)*

5. **Install WinLLM and its development dependencies**:
   ```cmd
   uv pip install -e .[dev]
   ```

6. **Verify the installation by running the tests**:
   ```cmd
   uv run pytest tests/ -v
   ```

---

## 2. Running Models (The Easy Way)

WinLLM is designed to be "plug-and-play". You don't need to manually configure complicated settings—WinLLM does it for you based on your hardware!

### Step 1: Detect Your Hardware
Before running a model, you can check what hardware WinLLM detects on your system:
```bash
winllm detect
```
This will show your GPU(s), VRAM, compute capability, and the dynamically calculated defaults (quantization, batch size, context length, attention backend).

### Step 2: Start the Server (Serve a Model)
Here are a few ways to start running models depending on your machine:

**For a Laptop (e.g., RTX 4070 Mobile, 8GB VRAM)**
Runs a small, efficient model quantified to 4-bit to save memory.
```bash
winllm serve -m microsoft/Phi-3-mini-4k-instruct -q 4bit
```

**For a Desktop (e.g., RTX 4090, 24GB VRAM)**
Auto-detects optimal settings for a standard 8B size model.
```bash
winllm serve -m meta-llama/Llama-3.1-8B-Instruct --auto-config
```

**For Multi-GPU Datacenters**
Spreads the model across 4 GPUs using Tensor Parallelism (`-tp 4`).
```bash
winllm serve -m meta-llama/Llama-3.1-70B-Instruct -tp 4 --auto-config
```

**For Computers without a GPU (CPU-only)**
Forces the engine to run strictly on the CPU.
```bash
winllm serve -m microsoft/Phi-3-mini-4k-instruct --device cpu
```

**When the model is too big for your GPU**
Offloads layers to your system RAM/CPU.
```bash
winllm serve -m mistralai/Mistral-7B-v0.3 --cpu-offload -q none
```

---

## 3. Performance Optimizations

WinLLM includes three major optimization stages to maximize hardware utilization:

### A. torch.compile (Latency Reduction)
Integrated PyTorch 2.0+ graph compilation which fuses kernels and reduces Python overhead in the decode loop.
- **Enable with**: `--compile`
- **Benefit**: ~10-15% faster token generation.
- **Note**: Requires a "warm-up" on the first prompt which may take 15-30 seconds.

### B. Continuous Batching (High Throughput)
Unlike traditional servers that process one request at a time, WinLLM uses an **Iteration-Level Scheduler**.
- New requests can join the active batch instantly without waiting for the previous request to finish.
- Dramatically increases throughput when multiple users are connected.

### C. Speculative Decoding (Advanced Speedup)
Uses a smaller "draft" model to propose multiple tokens at once, which the larger "target" model verifies in a single step.
- **Enable with**: `--draft-model <path_to_smaller_model>`
- **Benefit**: Can double generation speed for single requests.

---

## 3. How WinLLM Works (Architecture)

If you're curious about what happens under the hood when you send a prompt, here is the simplified flow:

1. **Request arrives via FastAPI**. The user asks a question through the OpenAI-compatible endpoint.
2. **Scheduler**. Placed in an async queue to handle multiple users at once.
3. **Inference Engine**:
   - **ModelLoader**: Loads the AI model weights onto your GPU(s) or CPU.
   - **KVCacheManager**: Manages VRAM memory for past tokens dynamically.
   - **Sampler**: Decides the next best word (token) using settings like Temperature and Top-P.
4. **PyTorch Engine**: Does the heavy mathematical lifting.
5. **Streaming Output**: Words are sent back to you piece by piece (via SSE streams) just like ChatGPT!

### Dynamic Defaults Explained
`device.py` detects your hardware and calculates optimal settings mathematically using `_build_defaults()` to prevent "Out Of Memory" (OOM) errors:

- **Quantization**: `4bit` if total VRAM < 16 GB, `none` otherwise
- **Batch size**: `max(1, int(total_vram_gb / 1.5))` — scales continuously with your GPU
- **Context length**: 2048 (< 12 GB) → 4096 (12–24 GB) → 8192 (≥ 24 GB)
- **Attention backend**: `flash_attention_2` on Ampere+ GPUs (compute ≥ 8.0), `sdpa` otherwise
- **Multi-GPU**: Auto-selects `balanced` device map and sets tensor parallelism to your GPU count

All defaults can be overridden with `WINLLM_*` environment variables (e.g., `WINLLM_MAX_BATCH_SIZE=16`).

---

## 4. Exploring the Codebase

If you want to contribute or modify WinLLM, here is a quick overview of the inner workings:

### Project Structure
```
S:\Code\wLLM\
+-- pyproject.toml
+-- README.md
+-- winllm/
|   +-- __init__.py / __main__.py
|   +-- config.py           # All config dataclasses + apply_hardware_defaults()
|   +-- device.py           # HW detection, dynamic allocation, env overrides
|   +-- registry.py         # Model family detection and auto-tuning
|   +-- model_loader.py     # Quantization, multi-GPU device_map, tensor parallelism
|   +-- kv_cache.py         # Block-based KV cache allocation and memory tracking
|   +-- sampler.py          # Token sampling pipeline (temperature, top-k, top-p)
|   +-- engine.py           # Core inference: prefill, decode loop, streaming
|   +-- scheduler.py        # Async request queue, continuous batching, prefix caching
|   +-- api_server.py       # FastAPI OpenAI-compatible REST server
|   +-- speculative.py      # Speculative decoding (draft/verify/accept)
|   +-- cli.py              # CLI entry point: serve / chat / benchmark / list / detect
|   +-- types.py            # Shared data types (GenerationRequest, RequestStatus)
|   +-- utils.py            # Chat prompt formatting utilities
|   +-- commands/           # CLI command implementations
|       +-- serve.py / chat.py / benchmark.py / list.py / detect.py / common.py
+-- tests/
|   +-- test_config.py      # Config validation and defaults
|   +-- test_device.py      # Hardware detection and profiling
|   +-- test_kv_cache.py    # KV cache allocation and block management
|   +-- test_registry.py    # Model family identification
|   +-- test_sampler.py     # Logits processing and sampling
|   +-- test_utils.py       # Chat prompt formatting
|   +-- benchmark_throughput.py  # Performance comparison tool
+-- documentation/
    +-- Architecture.md     # System architecture diagrams
    +-- COMMANDS.md         # CLI reference
    +-- WALKTHROUGH.md      # This file
    +-- CHANGELOG.md        # Version history
```

### Key Files in Detail

| File | Purpose |
|------|---------|
| `device.py` | Runs `DeviceInfo.detect()` and `_build_defaults()` to calculate optimal parameters from your actual VRAM. |
| `registry.py` | Auto-detects model family (Llama, Mistral, Qwen, Gemma) and applies family-specific tuning. |
| `config.py` | Centralizes all dataclasses: `ModelConfig`, `SchedulerConfig`, `KVCacheConfig`, `SamplingParams`. |
| `model_loader.py` | Builds quantization configs, handles multi-GPU distribution (`device_map`), CPU offloading, and model architecture introspection. |
| `kv_cache.py` | Calculates model-aware per-token cache usage mathematically, preventing OOM errors by capping allocations dynamically. |
| `engine.py` | Core inference loop. Orchestrates prefill and decode via decomposed helpers: `_validate_prompt`, `_run_decode_loop`, `_finalize_generation`, etc. |
| `scheduler.py` | Background inference loop with continuous batching. Handles KV admission, prefix caching (`_try_promote_prefix_cache`), and request lifecycle. |
| `speculative.py` | Three-phase speculative decoding: `_draft_proposals` generates candidates, `_verify_proposals` checks them, `_accept_or_reject` resolves mismatches. |
| `types.py` | Shared data types: `GenerationRequest` (tracks request state, tokens, KV cache) and `RequestStatus` enum. |
| `cli.py` | Parses CLI arguments and dispatches to command handlers in `commands/`. |

Enjoy building locally and taking full control of your LLMs with **WinLLM**!
