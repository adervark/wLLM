# Changelog

All notable changes to WinLLM are documented here.

---

## [1.0.0] - 2026-03-30
### High-Performance Tensor & Async Overhaul

#### Changed (Phase 3 - Hot-loop Tensor Allocation)
- **Zero-Fragmentation KV Batching** (`engine.py`) — Ripped out `F.pad` and `torch.cat` recursive padding routines inside the `_decode_batch` loop. Deployed O(1) contiguous sequence block allocations (`torch.zeros()`) and sliding-window slicing, reducing PyTorch matrix instantiations per generation step by over 90%.
- **In-Place Sampling Logit Mutation** (`sampler.py`) — Removed `.clone()` instructions across `apply_temperature` and `apply_repetition_penalty`. Integrated entirely mathematically isolated operations like `logits.div_(...)` to prevent hundreds of massive 128k Vocab tensors from saturating VRAM bandwidth.

#### Changed (Phase 2 - Async Architecture)
- **O(1) Memory Tracking** (`kv_cache.py`) — Removed iterative dict-crawling overhead (`_update_allocated_count`) during KV block provisioning. Upgraded tracking logic to intercept block references natively.
- **Latency Polling Annihilation** (`scheduler.py`) — Swapped out the old synchronous artificial `await asyncio.sleep(0.05)` generation request loops for explicit `asyncio.Event` suspension. The hardware CUDA event pushes back immediately into the event loop via `call_soon_threadsafe`, wiping away 50ms of trailing background lag.
- **Non-blocking Server Generators** (`api_server.py`) — Moved `engine.decode_tokens()` inside `run_in_executor` to guarantee active string transformations never intercept the primary ASGI concurrent thread handler.

### Production Hardening — Codebase Audit & Test Expansion

#### Fixed
- **Silent quantization drop** (`commands/common.py`) — CLI `--quantization awq|gptq` choices were missing from `QUANT_MAP`, causing them to be silently dropped to `auto`. Added both mappings.
- **Duplicate tokenizer loading** (`model_loader.py`) — `ModelLoader.load()` was loading the tokenizer twice: once directly and once via `BackendFactory`. The `pad_token` fix was applied to the first (discarded) copy. Now correctly delegates to `BackendFactory` and applies fixes post-load.
- **`_prefix_len` semantic conflict** (`types.py`, `engine.py`, `scheduler.py`) — `GenerationRequest._prefix_len` was used as a token count by the scheduler and a character count by the engine's streaming logic, causing corrupt streaming output. Split into `_prefix_cache_token_len` (scheduler) and `_stream_text_cursor` (engine).
- **Scheduler memory leak** (`scheduler.py`) — `_evict_completed()` was defined but never called. Completed requests accumulated unbounded in long-running servers. Now called automatically.
- **`num_running` always returned 0** (`scheduler.py`) — Dead `_running` dictionary was never populated. Removed entirely; active request count is now derived from `_active_reqs`.
- **Unchecked `IndexError` crash** (`scheduler.py`) — `output_token_ids[-1]` was accessed without checking for empty list. Added bounds check.
- **KV cache `reset()` left stale state** (`kv_cache.py`) — `reset()` did not clear `_block_pool`, `_prefix_cache_blocks`, `_prefix_cache_tensors`, or reset `_next_block_id`. All four are now properly cleared.
- **Missing `_draft_past_key_values` field** (`types.py`) — `SpeculativeEngine` relied on an undeclared attribute. Added to `GenerationRequest` dataclass.
- **Hardcoded version strings** (`api_server.py`, `commands/serve.py`) — Replaced stale `v0.1.0` with dynamic `winllm.__version__`.
- **Unused imports** (`utils.py`, `device.py`) — Removed `Union`, `json`, and `Path`.
- **Unprofessional comment** (`commands/__init__.py`) — Removed.

#### Changed
- **KV cache block counting** (`kv_cache.py`) — Replaced throwaway list comprehension in `_update_allocated_count` with a generator expression.
- **Scheduler docstring** (`scheduler.py`) — Added clarifying docstring to `submit_streaming` explaining it is a semantic wrapper.

#### Added — Test Suite (66 → 231 tests)
- **`test_types.py`** (29 tests) — `GenerationRequest` lifecycle, `RequestStatus` enum, cancellation thread safety, timing properties.
- **`test_engine.py`** (19 tests) — `InferenceEngine` with mocked backends: load, generate, streaming, EOS, max tokens, error propagation.
- **`test_backend.py`** (9 tests) — `BackendFactory` dispatch, tokenizer loading, ONNX fallback, ONNX routing for LiquidAI models.
- **`test_scheduler.py`** (14 tests) — `_get_prefix_hashes` correctness (empty, partial, deterministic), `SchedulerStats` calculations and `to_dict` format.
- **`test_speculative.py`** (11 tests) — Draft proposals, verification input shape, accept/reject logic, EOS termination, bonus sampling.
- **`test_api_server.py`** (15 tests) — Pydantic model serialization, OpenAI response format contract compliance.
- **`test_cli.py`** (16 tests) — Version output, subcommand registration, arg group validation, quantization/backend choices.
- **`test_kv_cache.py`** (+24 tests) — Prefix caching lifecycle, reset completeness (block pool, prefix caches, block ID counter), edge cases (double alloc, nonexistent free).
- **`test_sampler.py`** (+9 tests) — Full pipeline integration, greedy+repetition penalty interaction, high-temperature variety, top-k=1 edge case, logits immutability.

#### Removed
- **`test_combined.py`** — Duplicated `test_registry.py` tests and required real model downloads. Superseded by modular test files.

#### Documentation
- **`README.md`** — Removed `torch.compile` feature mention, added multi-backend and prefix caching, expanded architecture diagram, added all 13 test file descriptions, added documentation section links.
- **`Architecture.md`** — Updated `GenerationRequest` class diagram to show `_prefix_cache_token_len`, `_stream_text_cursor`, `_draft_past_key_values`; Added `_evict_completed()` to `Scheduler` class.
- **`WALKTHROUGH.md`** — Updated project structure listing with all 13 test files (7 new), corrected key files table with post-refactoring field names and responsibilities.
- **`Genesys.md`** — Replaced `torch.compile` chapter with multi-backend acceleration; added prefix caching to continuous batching section; updated vLLM comparison table; fixed concept-to-code map (`asyncio.Semaphore` → dynamic KV admission, added `BackendFactory` and prefix caching entries).
- **`CHANGELOG.md`** — This entry.

---


## [0.5.0] - 2026-03-29
### Multi-Backend Architecture & Windows Stability

#### Added
- **Multi-Backend Model Loading** (`backend.py` — NEW MODULE) — `BackendFactory` abstracts model loading across three inference backends: **PyTorch** (default), **ONNX Runtime** (via Optimum), and **DirectML** (via torch-directml). Selected via `--backend pytorch|onnxruntime|directml`.
- **`--backend` CLI flag** (`cli.py`) — New option on `serve`, `chat`, and `benchmark` commands for selecting the inference backend.
- **`inference_backend` config field** (`config.py`) — Added to `ModelConfig` to persist backend selection.
- **ONNX Export Script** (`compile_onnx.py` — NEW FILE) — Standalone script to export HuggingFace models to ONNX format using `optimum-cli`.
- **LiquidAI ONNX Auto-Routing** (`backend.py`) — Smart handling of LiquidAI's ONNX repository structure: automatically selects the correct pre-quantized binary (`model_q4.onnx`, `model_q8.onnx`, `model.onnx`) based on the `--quantization` setting, with proper `subfolder` and `file_name` routing.
- **Tokenizer Fallback** (`backend.py`, `model_loader.py`) — Built-in workaround for the Optimum `TokenizersBackend` bug on ONNX-exported models. Automatically falls back to the base model's tokenizer when the ONNX variant's `tokenizer_config.json` is corrupt.
- **`remove` command** (`cli.py`, `commands/remove.py`) — New CLI command to remove specific downloaded models (`wllm remove <model_id>`) or all cached models (`wllm remove --all`).
- **Safe `.eval()` guard** (`model_loader.py`) — `ORTModelForCausalLM` does not support `.eval()`, so model loading now checks `hasattr(model, "eval")` before calling it.

#### Removed
- **`torch.compile` support** (`engine.py`, `config.py`, `cli.py`) — The `--compile` flag and `_try_compile_model()` method have been completely removed. `torch.compile` was fundamentally broken on Windows due to missing Triton backend and MSVC compiler requirements. The multi-backend architecture replaces this with stable, native acceleration paths.
- **`compile` field** (`config.py`) — Removed from `ModelConfig` dataclass.

#### Changed
- **`model_loader.py`** — `BackendFactory.load()` is now called instead of direct `AutoModelForCausalLM.from_pretrained()`, enabling backend-agnostic model loading.

#### Documentation
- **`Architecture.md`** — Updated system architecture diagram to include `backend.py` in the inference layer. Added `BackendFactory` to the class diagram. Removed `torch.compile` references.
- **`COMMANDS.md`** — Removed `--compile` flag from all command option tables. Added `--backend` flag. Added `remove` command documentation.
- **`WALKTHROUGH.md`** — Rewrote performance section to replace `torch.compile` advice with backend selection and unquantized inference strategies. Added new "Inference Backends" section. Updated project structure to include `backend.py`, `compile_onnx.py`, and `remove.py`.
- **`CHANGELOG.md`** — This entry.

## [0.4.1] - 2026-03-18
### Code Clarity and Readability Refactoring

#### Fixed
- **Missing `import time`** (`engine.py`) -- `time.time()` was used but never imported, causing a runtime crash.
- **Missing `Optional` import** (`kv_cache.py`) -- `Optional` was used in type hints but never imported from `typing`.

#### Changed
- **`engine.py`** -- Decomposed the 170-line `_generate_impl` into focused helpers: `_validate_prompt()`, `_allocate_kv_cache()`, `_make_generator()`, `_get_stop_conditions()`, `_run_decode_loop()`, and `_finalize_generation()`. Extracted `_prefill_single_request()` and `_decode_single_request()` from `generate_step()`. Added module docstring with glossary of key concepts (prefill, decode, KV cache). Removed all stale "Stage X" development comments.
- **`model_loader.py`** -- Merged duplicate functions `_build_quantization_config` and `_get_quantization_config` into a single `_build_quantization_config` covering all quantization methods.
- **`scheduler.py`** -- Extracted 30-line prefix cache promotion block into `_try_promote_prefix_cache()` method.
- **`speculative.py`** -- Decomposed `step()` into `_draft_proposals()`, `_verify_proposals()`, and `_accept_or_reject()`. Added `from __future__ import annotations`. Fixed imports to use `.types` and `.sampler` instead of `.engine`.
- **`registry.py`** -- Replaced f-string logger calls with lazy `%s` formatting.
- **`device.py`** -- Added clear section header for backward-compatibility aliases.
- **`.gitignore`** -- Comprehensive rewrite covering Python caches, build artifacts, virtual environments, IDE files, OS generated files, testing artifacts, and torch offload weights. Untracked previously committed `.pyc` files.

#### Removed
- **Root test scripts** -- Deleted `test_stream.py`, `test_stream_2.py`, and `test_registry.py` (all duplicated by `tests/` or `test_combined.py`).

#### Documentation
- **`Architecture.md`** -- Updated engine.py and speculative.py sections to reflect decomposed method structure.
- **`WALKTHROUGH.md`** -- Expanded project structure to list all source files, test files, commands, and documentation. Updated key files table with accurate descriptions.

## [0.4.0] - 2026-03-16
### Added
- **Prefix Caching**: Physical KV cache storage with block-based hashing. Reduces TTFT for recurring prompts to near-zero.
- **Hardware-Native Quantization**: Added support for AWQ and GPTQ backends (`-q awq`, `-q gptq`).
- **Async Streaming**: Decoupled token decoding from the inference loop, eliminating $O(N^2)$ string overhead.
- KV Cache block reference counting for memory safety during prefix reuse.

## [0.3.0] — 2026-03-16

### Performance Optimization and Continuous Batching

#### Added
- **Continuous Batching** (`scheduler.py`, `engine.py`) — Completely refactored the request scheduler to use a centralized `InferenceLoop`. Multiple requests are now admitted into a single batch and processed concurrently, dramatically increasing throughput.
- **~~`torch.compile` Support~~ (Removed in v0.5.0)** (`engine.py`, `config.py`, `cli.py`) — *Originally integrated PyTorch 2.0+ graph compilation. Removed due to Windows incompatibility with Triton/MSVC backends.*
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
