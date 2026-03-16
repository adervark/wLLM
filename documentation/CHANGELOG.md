# Changelog

All notable changes to WinLLM are documented here.

---

## [0.3.0] — 2026-03-16

### Performance Optimization and Continuous Batching

#### Added
- **Continuous Batching** (`scheduler.py`, `engine.py`) — Completely refactored the request scheduler to use a centralized `InferenceLoop`. Multiple requests are now admitted into a single batch and processed concurrently, dramatically increasing throughput.
- **`torch.compile` Support** (`engine.py`, `config.py`, `cli.py`) — Integrated PyTorch 2.0+ graph compilation. Use the `--compile` flag to fuse kernels and reduce Python overhead (requires warm-up on first prompt).
- **Speculative Decoding** (`speculative.py`, `model_loader.py`, `engine.py`) — Support for using a smaller "draft" model to accelerate generation of a larger "target" model. Enabled via the `--draft-model` flag.
- **Comparative Benchmarking** (`tests/benchmark_throughput.py`) — New script to measure and compare TPS (Tokens Per Second) and TTFT (Time To First Token) with different optimization settings.

#### Fixed
- **Sampler logic** (`sampler.py`) — Fixed broken `apply_top_k` implementation that was causing incorrect token filtering.
- **Request Metadata Tracking** (`engine.py`) — Improved state management for batched requests, ensuring correct prefix decoding and streaming across iterations.

---

## [0.2.0] — 2026-03-14

### Device-Agnostic Dynamic Allocation

The hardware detection system has been fundamentally rewritten. Instead of classifying GPUs into static named profiles (`laptop`, `desktop`, etc.) with hard-coded defaults, all parameters are now **calculated mathematically** from actual hardware capabilities.

#### Added
- **`HardwareDefaults` dataclass** (`device.py`) — A clean container for all auto-tuned parameters: quantization, batch size, context length, device map strategy, tensor parallelism, GPU memory utilization, KV cache fraction, and attention backend.
- **`_build_defaults()` function** (`device.py`) — Replaces the old `PROFILE_DEFAULTS` lookup table. Dynamically computes optimal settings using formulas like `max_batch_size = max(1, int(total_vram_gb / 1.5))`.
- **Environment variable overrides** (`device.py`) — All hardware defaults can be overridden via environment variables:
  | Variable | Controls |
  |---|---|
  | `WINLLM_QUANTIZATION` | Default quantization mode |
  | `WINLLM_MAX_BATCH_SIZE` | Max concurrent requests |
  | `WINLLM_MAX_MODEL_LEN` | Max context length |
  | `WINLLM_DEVICE_MAP` | Device map strategy |
  | `WINLLM_TP_SIZE` | Tensor parallel size |
  | `WINLLM_GPU_UTILIZATION` | GPU memory utilization fraction |
  | `WINLLM_KV_FRACTION` | KV cache VRAM fraction |
  | `WINLLM_ATTENTION_BACKEND` | Attention implementation |
- **Attention backend auto-detection** (`device.py`) — Automatically selects `flash_attention_2` on GPUs with compute capability ≥ 8.0 (Ampere+), falling back to `sdpa` otherwise.
- **`--attention-backend` CLI flag** (`cli.py`) — New option on `serve`, `chat`, and `benchmark` commands. Choices: `auto`, `sdpa`, `flash_attention_2`, `eager`.
- **`attention_backend` config field** (`config.py`) — Added to `ModelConfig` and `HardwareDefaults`.
- **`kv_cache_fraction` field** (`device.py`, `config.py`) — Controls what fraction of remaining VRAM is pre-allocated for the KV cache pool (default: 90%).
- **GPU memory utility functions** (`device.py`) — Added `get_all_gpu_memory_info()`, `get_total_gpu_memory()`, `get_aggregate_gpu_memory()` for aggregate multi-GPU memory queries. These are re-exported from `model_loader.py` for backward compatibility.

#### Added — Model Registry (`registry.py` — NEW MODULE)
- **`ModelProfile` dataclass** — Pre-tuned configuration profiles for known model families, including recommended quantization, max context window, and RoPE scaling hints.
- **`KNOWN_MODELS` list** — Built-in profiles for: Llama (2/3), Mistral/Mixtral, Qwen (1.5/2), Gemma.
- **`identify_model_profile()`** — Auto-detects model family from the HuggingFace repo name using keyword matching.
- **`apply_model_profile()`** — Applies family-specific defaults (e.g., quantization, context window) to `ModelConfig` when `--auto-config` is used.
- **Integration with `ModelConfig.apply_hardware_defaults()`** — The hardware defaults pipeline now automatically runs the model registry to fine-tune settings per model family.

#### Changed
- **`device.py`** — Removed `_classify_profile()` function and `PROFILE_DEFAULTS` dictionary. Replaced with continuous mathematical allocation via `_build_defaults()`.
- **KV cache block cap** (`kv_cache.py`) — Dynamic cap scaling: `max(2048, int(total_vram_gb * 50))` instead of a fixed 2048 cap.
- **`DeviceInfo.summary()`** (`device.py`) — Now includes `attention_backend` in JSON output.
- **`KVCacheConfig`** (`config.py`) — Added `apply_hardware_defaults()` method that accepts `kv_cache_fraction` from `HardwareDefaults`.

---

## [0.1.1] — 2026-03-12

### High-Priority Bug Fixes

#### Fixed
- **Deprecated FastAPI lifecycle hooks** (`api_server.py`) — Replaced `@app.on_event("startup")` / `@app.on_event("shutdown")` with the modern `@asynccontextmanager` `lifespan` pattern.
- **Streaming timeout & cancellation** (`api_server.py`, `engine.py`) — Added `asyncio.TimeoutError` handling in the SSE stream. If token generation stalls beyond `stream_token_timeout`, the request is cancelled and an error chunk is yielded to the client.
- **Request cancellation support** (`engine.py`) — `GenerationRequest` now includes a thread-safe `_cancelled` event. The decode loop checks `is_cancelled` each step for cooperative cancellation.
- **Scheduler memory leak** (`scheduler.py`) — Added `_evict_completed()` method that clears old completed requests by TTL (`completed_request_ttl`) and max count (`max_completed_requests`).

---

## [0.1.0] — 2026-02-27

### Initial Release

- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`)
- Manual prefill + decode loop with KV cache reuse
- Token sampling pipeline: repetition penalty → temperature → top-k → top-p → multinomial
- 4-bit NF4 and 8-bit INT8 quantization via bitsandbytes
- Multi-GPU support: `device_map` sharding and `tp_plan` tensor parallelism
- Hardware detection and classification
- Asyncio-based request scheduler with semaphore concurrency control
- SSE streaming with threadsafe async queue bridge
- Interactive terminal chat (`winllm chat`)
- Throughput benchmark (`winllm benchmark`)
- Model cache listing (`winllm list`)
- Hardware detection display (`winllm detect`)
