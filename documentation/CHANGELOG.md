---
title: "CHANGELOG"
category: "wLLm"
tags: []
status: "Active"
created: "2026-04-01"
---
# Changelog

All notable changes to WinLLM are documented here.

> Note: entries up to 1.0.3 describe the codebase as it existed at the time of each release and reference the old flat module layout (`engine.py`, `scheduler.py`, ...), which was restructured in the unreleased changes below.

---

## [Unreleased] - 2026-07-16
### Clean startup output and smoother chat streaming

Every model load printed two spurious diagnostics: transformers 5.x deprecated
the `torch_dtype=` kwarg, and torch warned that `expandable_segments` (which
`configure_cuda_backends` forced into `PYTORCH_CUDA_ALLOC_CONF`) is not
supported on Windows and ignored it.

#### Fixed
- **`models/loader.py`, `backends/directml.py`**: pass `dtype=` to
  `from_pretrained` instead of the deprecated `torch_dtype=`.
- **`hardware/cuda.py`**: only set `expandable_segments:True` off-Windows;
  on win32 it was a no-op that made torch emit a UserWarning at load.

#### Changed
- **`cli/formatting.py`**: chat live-render refresh raised 10 → 24 fps so
  streamed tokens appear individually at typical decode speeds instead of
  4-5 at a time.

---

## [Unreleased] - 2026-07-15
### Size-aware auto quantization

The default quantization was unconditionally bnb NF4, which on models that
fit in VRAM at full precision costs ~35% decode throughput and doubles load
time (LFM2.5-1.2B on an 8 GB RTX 4070: 21→37 tok/s, 6.6→3.3 s load).

#### Changed
- **`QuantizationType.AUTO`** is the new `ModelConfig` default. The loader
  resolves it at load time (`models/quantization.py::resolve_auto_quantization`):
  full precision when the estimated fp16/bf16 weight size plus KV/activation
  headroom fits in free VRAM, NF4 otherwise (or when the size is unknown).
  Weight size comes from the local model dir, the HF cache, or the Hub's
  safetensors metadata, in that order. No CUDA resolves to `none`
  (bitsandbytes requires it).
- **Hardware defaults** (`hardware/defaults.py`) no longer pick quantization
  from total VRAM alone (`4bit if <16 GB`) — they leave `auto` for the
  size-aware resolution; `WINLLM_QUANTIZATION` still overrides.
- Config-default sentinels in `hardware/tuning.py` and `models/profiles.py`
  compare against `AUTO` instead of overloading `NF4` as "unset", so an
  explicit `-q 4bit` is now distinguishable from the default.

---

## [Unreleased] - 2026-07-14
### Rich CLI Output

CLI presentation moved from hand-rolled ANSI to the `rich` library (new
runtime dependency). Shared `Console` in `winllm/cli/console.py`; spec in
`documentation/2026-07-14-cli-rich-output-design.md`.

#### Added
- **Live markdown chat streaming** (`cli/formatting.py`) — the assistant's reply renders as markdown while it streams (syntax-highlighted code blocks, emphasis, lists); `<think>` reasoning streams as dim text under a dim rule. `ThinkingStreamFormatter` replaced by a presentation-free `ThinkTagParser` plus a `RichChatRenderer` built on `rich.live.Live` (capped at 10 refreshes/s).
- **Model-load spinner** — `chat` and `benchmark` show an animated status while `load_model()` runs.
- **Tables** — `detect` (hardware + recommended defaults), `list` (cached models with size/modified and a totals caption), `benchmark` (per-prompt results filling in live, summary panel).
- **GitHub Dark theme** — central rich `Theme` (`GITHUB_DARK` in `cli/console.py`, GitHub Primer dark palette) drives labels, rules, tables, panels, spinner, and log-level badges via semantic `wllm.*` style names; chat code blocks highlight with pygments' `github-dark`.

#### Changed
- **Logging** (`cli/main.py`) — `RichHandler` on the shared console; `transformers`/`urllib3`/`filelock`/`accelerate` capped at WARNING unless `-v` (which also restores timestamps and module paths). Logs now share the stdout console with command output (previously stderr) so spinners, live regions, and log lines coexist.
- **`detect` slimmed** — now shows the GPU table plus a `Platform … · Total VRAM …` line; the Profile display and Recommended-defaults table are gone from human output (both remain in `--json`).

## [Unreleased] - 2026-07-07
### Fused CUDA Sampling Kernel & Grammar Bitmask Kernel

Levers #3 and #4 from PERFORMANCE_NOTES.md, delivered as one optional
native extension (`native/sampling/`, pybind11 + nvcc + MSVC; build with
`uv pip install ./native/sampling`). wLLM runs identically without it.

#### Added
- **Fused CUDA sampling kernel** (`native/sampling/`, `sampling/native.py`) — one kernel launch replaces the ~15-launch per-token torch pipeline (repetition penalty → temperature → top-k → top-p → categorical draw); on Windows/WDDM the launches, not the math, were the cost. Exact top-k/top-p thresholds via MSB radix select on float bit patterns; penalty values round through the logits dtype so **temp-0 output stays token-identical to the torch path** (verified full-generation). `sample_token` routes eligible calls automatically: CUDA logits, uniform batch params, and no request generator — seeded temp>0 requests keep the torch path so seeded reproducibility is untouched (pinned by test). Microbench: 0.97 → 0.34 ms per call at chat defaults, 1.02 → 0.22 ms for a verify batch of 8. End-to-end chat (SmolLM2-135M, graphs + suffix decoding, interleaved A/B): **+13% at temp 0.7** (87.9 → 99.4 tok/s); temp 0 unchanged within noise, bit-identical output. The extension is deliberately **torch-free** (raw device pointers + stream handle in, cudart linked statically) because the CUDA 13.2 toolkit vs cu12.8 torch rules out `torch.utils.cpp_extension`.
- **Grammar bitmask kernel** (same module, `sampling/grammar.py`) — applies xgrammar's packed bitmask to logits in one launch instead of the unpack-to-bool + `masked_fill_` chain; grammar rows also qualify for the fused sampler. Structured output: **+11%** (54.5 → 60.5 tok/s, `json_object`, temp 0.7). Pure-torch unpack remains the CPU/CI/`WINLLM_PURE_PYTHON=1` fallback.
- **Native sampling parity tests** (`tests/test_native_sampling.py`) — kept-set exactness and frequency agreement against the torch pipeline across dtypes/params, bit-exact greedy+penalty, bitmask-vs-unpack equality, eligibility routing (seeded/mixed-batch/CPU fall back), and a failure-latch test (a kernel error disables the native path instead of taking decode down). Skipped wholesale without the built module + CUDA.

#### Fixed
- **PERFORMANCE_NOTES correction** — the decode dtype on this setup is **bfloat16**, not fp16 as the 2026-07-06 headline claimed; discovered debugging a one-token temp-0 divergence that only bf16's 0.25 ulp could produce (37.0/1.1 → 33.75).

## [Unreleased] - 2026-07-06 (second pass)
### Decode-Path CPU Overhead: WDDM Sync Elimination & Native Suffix Cache

See `documentation/PERFORMANCE_NOTES.md` for the full research log
(profiles, methodology, sources). Chat path, SmolLM2-135M fp16 with
`--cuda-graphs --suffix-decoding`: 86 → 93 tok/s at temp 0
(token-identical), 86 → 112 tok/s at temp 0.7.

#### Changed (sampling hot path)
- **Sync-free sampling ops** (`sampling/ops.py`, `sampling/sampler.py`) — profiling showed 55% of decode wall time was `torch.tensor(..., device="cuda")` calls and GPU boolean skip-checks inside the per-token sampling pipeline (each is a WDDM stream sync; the model forward itself was 0.8 ms). All skip decisions now happen in Python on the parameter lists; uniform parameters (always, for single-request chat) take scalar paths that transfer nothing; repetition penalty is a gather/scatter over the generated ids instead of a vocab-sized mask; the `isnan().any()` guard became an unconditional `nan_to_num` (the check cost more than the kernel).
- **Batched verify sampling** (`inference/generation.py`) — `_speculative_commit` now samples all verify rows in one call + one `.tolist()` when `repetition_penalty == 1.0`, instead of one `sample_token(...).item()` sync per draft token. Rows past the first rejection are discarded, which cannot change committed output; the sequential path remains for penalty requests.

- **Adaptive verify throttle** (`inference/generation.py::_VerifyThrottle`) — graph-verified speculation now backs off exponentially (skip 1, 2, 4, ... capped 32) while verifies come back fully rejected, and resets on any acceptance. Templated output whose repeated pattern continues with new values (0% acceptance) previously paid a wasted verify replay per token; now it pays ~zero speculation overhead. `_speculative_commit` returns `(committed, accepted)` to feed the throttle. Note: pinned-staging for the graph input buffers was tried and **reverted** — 5× slower under WDDM (negative result documented in PERFORMANCE_NOTES.md and a code comment).

#### Added
- **Native (C++) suffix cache** (`native/suffix/`, pybind11 + MSVC) — optional drop-in for `SuffixCache`, ~5× faster on match-heavy streams with *flat* per-step cost (6 µs), making higher `max_occurrences`/`max_back_ext` free. Build: `uv pip install ./native/suffix`. Pure-Python fallback is automatic (CI needs no compiler); `WINLLM_PURE_PYTHON=1` forces it. A 400-step randomized parity test keeps both implementations exactly in agreement, and both expose a new `has_request()` probe.
- **Chat CLI formatting** (`cli/commands/chat.py`) — colored role labels, and a dim per-response footer with tokens · time · tok/s plus per-response speculation acceptance (drafted/accepted delta from the engine counters), TTY-gated like the thinking formatter.
- **`documentation/PERFORMANCE_NOTES.md`** — the profiling method, WDDM sync economics with external sources, per-fix measurements, and the prioritized list of remaining levers (pipelined decode, pinned staging, fused sampling kernel, grammar-mask kernel, global suffix tree).

#### Fixed (review findings)
- **Seeded requests kept off the batched verify path** (`inference/generation.py`) — regression in the batched-verify change above, caught before release: batching draws multinomial samples for discarded rows too, so a *seeded* temp>0 request consumed extra RNG draws per verify step and its output diverged from a speculation-off run, breaking the documented losslessness invariant. Seeded sampling requests now use the sequential path (one draw per committed token, exactly like plain decode); unseeded and greedy requests — including the default chat path — still batch. Regression test pins the RNG-stream alignment; verified token-identical on GPU (SmolLM2, seed 1234, temp 0.7, spec on vs off).
- **Chat CLI silently swallowed failed generations** (`cli/commands/chat.py`) — a request that failed validation (prompt too long, insufficient KV cache) streamed nothing but still printed a healthy-looking `0 tok · 0.0s` stats footer and appended an *empty* assistant turn to the conversation, corrupting the chat template for every later message. Failures now print the error and drop the failed user turn from history.
- **Double clone on the grammar sampling path** (`sampling/sampler.py`) — grammar-constrained requests cloned the logits for masking and then the pipeline cloned them again every token; `sample_token` now tracks ownership and clones once.

## [Unreleased] - 2026-07-06
### Model-Free Speculative Decoding, Safer Validation Probes & Streaming Correctness

#### Added
- **Suffix decoding — model-free speculative decoding** (`inference/suffix_cache.py`, `inference/suffix_speculative.py`, `--suffix-decoding`) — implements the core of SuffixDecoding (Oliaro et al., NeurIPS 2025, arXiv:2411.04975): drafts future tokens by finding the longest suffix of the committed text that already occurred in the prompt/output and proposing what followed it, with speculation length adaptive in the match length (`α·p`). No draft model, no extra VRAM, works with any architecture; committed tokens are always the target's own samples, so output is **token-identical to plain decoding**. Measured 1.14–1.24× on structured output and up to 1.84× on input-grounded generation (RTX 4070 Laptop). Plugs into the existing `runtime.speculative_engine` slot used by the scheduler's decode path; steps with no usable match fall back to a plain decode at zero extra cost. Simplifications vs. the paper: linear drafts (not speculation trees) and per-request history.
- **Cache-rollback soundness probe** (`inference/suffix_speculative.py::validate`) — speculative rejection requires dropping KV entries for rejected draft positions, and hybrid caches with recurrent state don't rewind: `Lfm2HybridConvCache.crop()` exists but is *lossy* (probe logit diff ~8–12 after 3 junk tokens), silently corrupting output after every rejection. At load, the engine now runs a context → junk → crop → compare probe and only enables suffix decoding when rollback is bit-faithful (`DynamicCache` passes exactly). Unsound architectures degrade to plain decode with a clear warning.
- **`.gitattributes`** — normalizes text files to LF (CRLF for `*.bat`/`*.ps1`), ending the LF→CRLF churn on every touched file under `core.autocrlf=true`.
- **Graph-verified speculation** (`inference/cudagraph.py`, `inference/generation.py`) — the CUDA-graph decoder now captures a *second* graph for a fixed-width verify forward (`VERIFY_BUCKET + 1` tokens), and the blocking path scores suffix drafts with one graph replay per step when both `--cuda-graphs` and `--suffix-decoding` are set. Rejection rollback is a position rewind on the `StaticCache` — lossless by construction (junk positions are overwritten before any causal query can attend them), no `crop()` involved. Validated at load like the decode graph (teacher-forced per-position logits agreement; failure disables only the verify path). Measured on SmolLM2-135M fp16 (chat path): 87.6 → 100.3 tok/s on structured output, token-identical. This also gives the CLI chat path speculation for the first time (previously scheduler-only).
- **Speculation acceptance metrics** (`server/metrics.py`, `inference/speculative.py`) — `/metrics` now exposes `winllm_spec_drafted_tokens_total`, `winllm_spec_accepted_tokens_total`, and `winllm_spec_acceptance_rate` for whichever speculative engine is active (the draft-model engine gained the same counters the suffix engine had).

#### Changed (CUDA graph validation)
- **Teacher-forced logits validation** (`inference/cudagraph.py`) — the load-time self-check no longer compares exact greedy tokens (too weak at 4 steps — it passed while real generation diverged at token 6 — and too strict if lengthened, since fp16/quantized kernels legitimately flip near-tie argmaxes under `StaticCache`). The probe now feeds identical eager-chosen tokens to both paths and requires mutual top-k agreement of the logits at every step (8 steps), which catches capture corruption without rejecting benign numeric noise.
- **Hybrid architectures rejected at construction** (`inference/cudagraph.py`) — `StaticCache` constructs for conv/SSM hybrids (e.g. LFM2) but explodes mid-forward; `layer_types` containing non-attention entries now disables graphs up front with a clear message (transformers 5.x removed `_supports_static_cache`).
- **StaticCache VRAM now visible to the KV estimator** (`inference/engine.py`) — graph-decoder (and speculative) init moved *before* `KVCacheManager` creation so the estimator's free-VRAM read accounts for the full-context `StaticCache` materialized by the validation probe.

#### Fixed (review findings — streaming & API contract)
- **Speculative steps dropped tokens from SSE streams** (`inference/streaming.py`, `core/types.py`) — `StreamEmitter.emit` sent only `output_token_ids[-1]`; a speculative step committing N tokens streamed 1. Emit now drains every unsent token via a per-request `_emit_cursor`.
- **One bad `response_format` failed the whole batch** (`inference/engine.py`) — an unparseable JSON schema raised out of `_ensure_grammar_states` and the scheduler's catch-all failed *every* active request. Grammar setup failures now fail only the offending request, which is also excluded from the step so it can't emit unconstrained tokens.
- **Speculation overshot `max_tokens`** (`inference/suffix_speculative.py`, `inference/speculative.py`) — both engines now cap drafts at the remaining budget (`max_tokens − committed − 1`); an all-accepted step lands exactly on the cap instead of up to 17 (suffix) / 5 (draft-model) tokens over.
- **Streamed responses contained stop strings** (`server/streaming.py`) — the scheduler trims `output_text` at the stop match, but SSE clients had already received the stop string plus overshoot. A `StopStringGate` now withholds `len(longest stop) − 1` trailing characters so no stop string (even one spanning chunks) ever reaches the client; held-back clean text is flushed before the terminal chunk.
- **Pathological suffix-match cost** (`inference/suffix_cache.py`) — `propose()` scans only the 32 most recent occurrences of the anchor bigram (recency already won ties); previously a common bigram in a long generation meant O(occurrences × 64) Python work per decoded token.

#### Changed (deduplication)
- **One eager decode step** (`inference/eager.py`) — the single-token forward (`token → model(past) → last logits`) existed in four copies (DecodeRunner, BlockingGenerator, suffix fallback, validation probes); all now call `eager_decode_step`, so the probes exercise literally the production invocation. The suffix fallback also gained the shared `DecodeInputBuffer` (no more per-step CUDA allocation).
- **`trim_cache` moved to `inference/buffers.py`** — generic KV trimming (Cache `.crop()` or legacy tuple slice) now lives with the cache plumbing; both speculative engines import it from there (`_trim_kv` wrapper deleted). This also fixed the draft-model engine's trim, which only handled legacy tuple caches, not transformers 5.x `Cache` objects.
- **Validation probe construction** deduplicated into `InferenceEngine._probe_tokens()`.

#### Tests
- Expanded to **384 tests**. New files: `test_suffix_decoding.py` (suffix cache proposals/budget/LRU, engine accept/reject/EOS, losslessness vs plain decode, crop-soundness validation, budget cap, emitter drain), `test_stop_gate.py` (stop spanning chunks, false-alarm release, earliest-of-multiple stops, single-char stops); `test_cudagraph.py` extended (hybrid guard, logits-agreement comparator); `test_speculative.py` extended (draft budget caps); `test_engine.py` emitter tests updated to the drain contract (the old assertions encoded the token-drop bug).

#### Documentation
- `README.md` (suffix decoding), `COMMANDS.md` (`--suffix-decoding`), `Architecture.md` (suffix decoding, updated CUDA-graph validation), `WALKTHROUGH.md` (model-free speculation section, quantization speed note), `SOLID_Architecture.md` (new modules), and this entry.

---

## [Unreleased] - 2026-07-02
### Structured Output, CUDA Graphs, Observability & API Correctness

#### Added
- **Structured output (JSON mode)** (`sampling/grammar.py`, `sampling/sampler.py`, `server/app.py`) — OpenAI-style `response_format` (`json_object` / `json_schema`) on `/v1/chat/completions`, backed by xgrammar grammar-constrained decoding. Works with any zero-day model: the token bitmask is unpacked with plain torch ops, so it runs on CUDA without Triton (which xgrammar's own GPU kernel needs and Windows lacks). Optional dependency: `pip install winllm[structured]`. Grammar-constrained requests automatically bypass speculative decoding (draft proposals aren't masked).
- **CUDA graph decode** (`inference/cudagraph.py`, `--cuda-graphs`) — captures the single-request decode forward in a `torch.cuda.CUDAGraph` over a `StaticCache`, eliminating per-token kernel-launch overhead without torch.compile/Triton. Measured ~7× decode throughput on SmolLM2-135M (142 vs 20 tok/s on an RTX 4070 Laptop). A load-time self-check replays a short greedy probe against eager output and **disables the graph path on any mismatch or capture failure**, so graph-unsafe architectures silently fall back to the normal loop. Opt-in via `--cuda-graphs` / `ModelConfig.enable_cuda_graphs`.
- **Prometheus `/metrics` endpoint + latency percentiles** (`server/metrics.py`, `scheduling/stats.py`) — request/token counters, queue and KV-cache gauges, and p50/p90/p99 summaries for TTFT (queue-inclusive), end-to-end latency, and per-request decode throughput over a sliding window. Hand-rolled text exposition format (no new dependency). Percentiles also appear in `/health` under `scheduler.stats`.
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — runs the full test suite CPU-only on `windows-latest` and `ubuntu-latest` with CPU torch wheels and only the deps the tests import (the GPU-only extras are lazily imported at model-load time).

#### Fixed (API correctness)
- **`stop` sequences were silently ignored on the API** (`scheduling/scheduler.py`) — the server routes through the continuous-batching scheduler, whose finish check only knew EOS and `max_tokens`; stop strings were only honored in the blocking CLI path. The scheduler now decodes output incrementally (one decode per token, cursor-tracked) and finishes/trims on stop-string hits identically to the blocking path.
- **Failed requests got stuck in the active batch** (`scheduling/scheduler.py`) — a `generate_step` exception marked requests FAILED and woke their waiters but never removed them from the batch: the loop re-ran them forever and their KV blocks never freed. Failure is now a first-class finish reason collected by the same cleanup path (batch removal + KV free + waiter signal).
- **SSE streams never received a terminal chunk** (`scheduling/scheduler.py`) — the scheduler only signaled `_stream_callback` on finish, never the token-ID callback the API server registers, so streaming responses only ended via the stream timeout. Both callback flavors are now signaled.
- **`finish_reason` was hardcoded to `"stop"`** (`core/types.py`, `server/app.py`, `server/streaming.py`) — requests now carry a real `finish_reason` (`stop` / `length` / `cancelled` / `error`) set by both generation paths and surfaced in non-streaming and SSE responses, so clients can detect `max_tokens` truncation.

#### Changed (Throughput)
- **Incremental batched-cache repacks** (`inference/buffers.py`, `inference/decode.py`) — on batch membership change, surviving requests' KV rows now move via one fused gather+scatter per layer tensor (`PersistentBatchCache.refresh`) instead of a per-request Python copy loop; only newly admitted requests pay a per-row prefill-cache copy, and left-padding shrinks to the surviving max length. The per-step attention mask is now built as a single vectorized comparison instead of a per-row loop (previously B kernel launches *per generated token*).

#### Fixed (Robustness / Hygiene)
- **KV estimator computed a 1-block budget when CUDA is available but no device is visible** (`kvcache/estimator.py`) — e.g. `CUDA_VISIBLE_DEVICES=""` with a driver present made every request fail admission ("Request too large for KV cache"); zero visible devices now uses the CPU fallback budget.
- **Version drift** (`pyproject.toml`) — package version is now sourced dynamically from `winllm.__version__` (was 1.0.0 vs 1.0.1).
- **Removed the engine's test-only delegation shim** (`inference/engine.py`) — tests now target the collaborators (`_blocking_generator`, `_decode_runner`, `_emitter`, `_runtime`) directly.
- **Repo cleanup** — deleted superseded root scripts (`fix_torch.py`, `research_usps.py`, `setup_and_test.*`, `test_detect.bat`, `req_torch.txt`, `check_imports.py`) and the 250 KB `insights.txt`.

#### Tests
- Expanded to **339 tests**. New files: `test_grammar.py` (bitmask unpacking, grammar state lifecycle, schema compilation, response_format validation), `test_cudagraph.py` (capture, probe validation, eager fallback on mismatch/failure), `test_metrics.py` (Prometheus exposition rendering), `test_scheduler_loop.py` (stop-string finish, failure cleanup, terminal stream chunks, `finish_reason` propagation); `test_engine_batched.py` expanded for the fused repack path and vectorized attention mask.

#### Documentation
- `COMMANDS.md` (`--cuda-graphs` flag, `/metrics` + `response_format` on serve), `Architecture.md` (CUDA graphs, grammar decoding, metrics, finish handling), `WALKTHROUGH.md` (structured-output and monitoring sections, CUDA graph optimization, updated test map), `SOLID_Architecture.md` (new modules in the package map), `README.md`, and this entry.

---

## [Previous Unreleased] - 2026-06-16
### Correctness & Efficiency Fixes

#### Added (UX)
- **Segregated reasoning in the chat CLI** (`cli/formatting.py`, `cli/commands/chat.py`) — reasoning models (DeepSeek-R1, QwQ, ...) emit their chain-of-thought inside `<think>...</think>`, which previously printed inline and blurred together with the answer. The new `ThinkingStreamFormatter` detects the tags *as tokens stream in* (correctly, even when a tag is split across token boundaries) and renders the reasoning dimmed under a `┌─ thinking ─┐` header, then the answer normally below. Colorizes only on a real TTY; models that emit no tags stream through unchanged.

#### Changed (Throughput)
- **Persistent batched KV cache** (`inference/buffers.py`, `inference/decode.py`, `inference/engine.py`, `scheduling/scheduler.py`) — batched decode previously repacked every request's full KV history into a shared buffer **and** cloned it all back out *every* step (~2× the whole batch's KV copied per generated token, scaling with context length). The new `PersistentBatchCache` keeps the batched cache and feeds it straight back while batch membership is unchanged, repacking only when a request joins/leaves; per-request caches are exposed as zero-copy *views* so `req._past_key_values` stays correct with no clone. `BatchKVBufferPool` (and its per-step buffer zeroing) is removed. Finished requests now have their KV references cleared so completed-store entries can't pin the batched tensors.

#### Fixed (Correctness)
- **Speculative decoding corrupted the KV cache after a rejection** (`inference/speculative.py`) — `_trim_target_kv` trimmed to `len(prompt) + len(output)` instead of `… - 1`, keeping one extra slot that held the *rejected draft token's* key/value rather than the accepted correction's. Every rejection (i.e. most steps) then decoded on a poisoned cache. Trimming now restores the true invariant (`prompt + output - 1`) for the target cache.
- **Speculative draft model never saw the prompt** (`inference/speculative.py`) — `_draft_past_key_values` was reset to `None` every step, so the draft proposed from a single token of context, crushing the acceptance rate. The draft cache is now prefilled with the prior context (`_ensure_draft_context`) and maintained incrementally across steps (extend-by-one on full acceptance, then trimmed to the same invariant in `_sync_caches`).
- **EOS detection broke for multi-EOS tokenizers** (`core/types.py`, `scheduling/scheduler.py`, `inference/generation.py`, `inference/speculative.py`) — equality against a scalar `eos_token_id` silently failed when a tokenizer exposes a *list* (e.g. Llama-3), so generation only stopped at `max_tokens`. New `normalize_eos_ids()` produces a set used by all stop-checks; also fixes a latent `{list}` unhashable crash in `get_stop_conditions`.
- **Prefix cache only ever stored the first block** (`scheduling/scheduler.py`, `kvcache/manager.py`, `kvcache/prefix.py`) — promotion saved just block 0, so a shared prompt skipped at most 16 tokens regardless of length. Now every complete prompt block is promoted as a cumulative chain (each block stored once, O(n) memory); `PrefixCache.match` concatenates the matched chain.

#### Fixed (Efficiency / Robustness)
- **CLI streaming was O(n²)** (`inference/generation.py`) — the blocking generator re-decoded the full sequence every token. Now decodes incrementally per token, matching the API/`StreamEmitter` path.
- **Top-k sampling sorted the whole vocab** (`sampling/ops.py`) — replaced `torch.sort` with `torch.topk(max_k)` (O(vocab) vs O(vocab·log vocab)).
- **`AdmissionController` was reconstructed every loop iteration** (`scheduling/scheduler.py`) — now built once (lazily, after the model loads).
- **`total_cached_tokens` double-counted shared prefix blocks** (`kvcache/blocks.py`) — now sums distinct physical blocks.
- **VRAM estimate ignored PyTorch's reserved pool** (`kvcache/estimator.py`) — uses `max(allocated, reserved)` for a conservative free-VRAM figure.
- **Prefix cache was unbounded and pinned blocks forever** (`kvcache/prefix.py`, `kvcache/manager.py`, `kvcache/blocks.py`) — on a long-running server it slowly starved the live KV budget. Added a budget (`KVCacheConfig.prefix_cache_block_fraction`, default 25% of total blocks) with **leaf-only LRU eviction**: only a prefix with no longer prefix depending on it is removed (least-recently-matched first), and `BlockAllocator.unpin_block` reclaims the freed block. Never orphans a chain or evicts a block another chain still needs.
- **Speculative draft loaded inconsistently** (`models/loader.py`) — the draft model bypassed quantization, `attn_implementation`, dtype, and the backend registry (loaded full-precision/eager via a bare `from_pretrained`). It now loads through the same backend pipeline and shares the target's load kwargs (`_build_load_kwargs`), never sharded by tensor parallelism.
- **`CompletedRequestStore` sorted all entries to evict** (`scheduling/completed.py`) — switched to an insertion-ordered `OrderedDict`; TTL and count eviction now pop the oldest from the front in O(evicted) instead of O(n log n).

#### Changed (SOLID cleanup)
- **KV layout knowledge removed from the scheduler** — new pure helper `kvcache.slice_prompt_blocks()` owns the `[batch, heads, seq, head_dim]` carving; the scheduler just orchestrates `slice_prompt_blocks(...) → promote_prefix_chain(...)`.
- **Single prefix-promotion path** — `KVCacheManager.promote_prefix_chain()` is now the one bookkeeping path (pin-once, store-chain); `promote_to_prefix()` is a thin backward-compatible wrapper over it.
- **Shared model load-kwargs builder** — target and draft loads share `ModelLoader._build_load_kwargs`, so the draft can't drift from the target's quantization/dtype/attention settings.

#### Tests
- Expanded to **302 tests**. New coverage: end-to-end speculative output-equivalence with target-greedy decoding (draft-agrees, draft-disagrees, cache-invariant), target-KV trim after rejection, multi-block prefix promote/match + single-pin accounting, the `slice_prompt_blocks` carving helper, prefix-cache eviction (budget cap, leaf-only, LRU refresh, block unpin), persistent batched cache (repack-once across stable steps, repack-on-membership-change, view correctness), and the chat-CLI `ThinkingStreamFormatter` (passthrough, segregation, tag-split-across-deltas, partial-tail safety, color toggle). `test_resets_draft_kv_cache` → `test_maintains_draft_kv_cache` (the old test asserted the draft-context bug).

---

## [Unreleased] - 2026-06-11
### SOLID Architecture Restructure

#### Changed (Breaking — import paths)
- **Full package restructure** — The flat `winllm/*.py` modules were decomposed into single-responsibility packages: `core/`, `config/`, `hardware/`, `backends/`, `models/`, `kvcache/`, `sampling/`, `inference/`, `scheduling/`, `server/`, and `cli/`. Behavior is unchanged; import paths are not. See `documentation/SOLID_Architecture.md` for the full map and design rationale.
  - `winllm.engine` → `winllm.inference` (engine facade + `PrefillRunner`, `DecodeRunner`, `BlockingGenerator`, `StreamEmitter`, buffer pools)
  - `winllm.scheduler` → `winllm.scheduling` (+ `AdmissionController`, `CompletedRequestStore`, `SchedulingPolicy`/FCFS)
  - `winllm.kv_cache` → `winllm.kvcache` (+ `BlockAllocator`, `KVMemoryEstimator`, `PrefixCache` behind the `KVCacheManager` facade)
  - `winllm.sampler` → `winllm.sampling` (pure ops + composable `LogitsProcessor` pipeline)
  - `winllm.backend` → `winllm.backends` (`BackendFactory` replaced by class-based backends + `BackendRegistry`)
  - `winllm.device` → `winllm.hardware` (detection, defaults, memory, CUDA setup, config tuning)
  - `winllm.model_loader` / `winllm.registry` / `winllm.utils` → `winllm.models` (loader, quantization, introspection, profiles, chat templates)
  - `winllm.api_server` → `winllm.server` (schemas, SSE streaming, app factory)
  - `winllm.types` → `winllm.core.types`; new `winllm.core.interfaces` defines the `InferenceBackend`, `LogitsProcessor`, `SchedulingPolicy`, and `StoppingCriterion` extension points
  - `winllm.cli` + `winllm.commands` → `winllm.cli` package (`main.py` + `commands/`); the `winllm.cli:main` console entry point is unchanged
- **Config purity** — `apply_hardware_defaults()` methods were removed from the config dataclasses; hardware tuning now lives in `winllm.hardware.tuning` (`apply_model_defaults`, `apply_scheduler_defaults`, `apply_kv_cache_defaults`, `apply_hardware_defaults`).
- **Public API convenience** — `winllm/__init__.py` now lazily exports the main entry points (`InferenceEngine`, `Scheduler`, `create_app`, configs, `GenerationRequest`) without importing torch at `import winllm` time.

#### Added
- `documentation/SOLID_Architecture.md` — package map plus how each SOLID principle maps to a concrete seam (backend registry, logits-processor pipeline, scheduling policies).

#### Tests
- Suite updated to the new layout and expanded to **282 tests** (backend registry dispatch and ONNX kwarg routing now have direct, assertable coverage).

---

## [1.0.3] - 2026-04-09
### Day Zero Blackwell (RTX 50-series) Support

#### Added
- **Native Blackwell Support** (sm_120) — Enabled support for the NVIDIA RTX 5060, 5070, 5080, and 5090 series.
- **CUDA 12.8 Upgrade** (`pyproject.toml`) — Upgraded core PyTorch dependency to CUDA 12.8 package index, providing the necessary `sm_120` kernels missing in older versions.
- **Blackwell-Specific Optimizations** (`device.py`) — Optimized default batch sizes and context lengths for the increased throughput and memory bandwidth of the Blackwell architecture.
- **Architecture Mismatch Diagnostics** (`diagnose_gpu.py`) — Added smart detection for Blackwell GPUs with recommendations for the correct CUDA environment.

#### Changed
- **`bitsandbytes` Upgrade** (`pyproject.toml`) — Bumped to `>=0.45.0` to ensure compatible 4-bit/8-bit kernels for Blackwell.

---

## [1.0.2] - 2026-04-06
### Zero-Dependency Bootstrapping

#### Added
- **Python & UV Bootstrapping** (`install.bat`) — The installation script now detects systems with no Python or `uv` installed. It automatically fetches and installs the `uv` toolchain via PowerShell and uses it to provision a managed Python 3.12 environment, enabling true one-click setup on clean Windows machines.
- **Dynamic Session PATH Refresh** (`install.bat`) — Bootstrapped tools are automatically injected into the current command session's `PATH` for zero-restart execution.
- **`uv python install 3.12` Integration** (`install.bat`) — Guarantees compatible Python 3.12 availability regardless of global system configuration.

#### Documentation
- **`README.md`** — Updated "Rapid Deployment" section to highlight zero-dependency installation capabilities.

## [1.0.1] - 2026-04-04
### Stability Bug Fixes & Test Hardening

#### Critical Fixes
- **Prefix caching completely broken** (`scheduler.py`) -- `_try_promote_prefix_cache` used Python `hash()` while `_admit_requests` used SHA-256 via `_get_prefix_hashes()`. Promoted entries were never found during lookup. Fixed to use `_get_prefix_hashes()` consistently.
- **`_get_prefix_hashes` crashes on real models** (`scheduler.py`) -- `bytes()` serialization only handles token IDs 0-255, but real vocabularies use IDs up to 150k+. Replaced with `struct.pack()` for proper integer serialization.
- **Speculative decode KV extend always 0** (`engine.py`) -- `len(output_token_ids) - generation_tokens` is always 0 (they are the same value). Fixed by capturing output length before the speculative step and computing the delta.
- **`SequenceBlocks.total_tokens` returns 0 on direct construction** (`kv_cache.py`) -- `_total_tokens` cached counter was only set by `KVCacheManager.allocate_sequence`, not by direct `SequenceBlocks(blocks=...)` construction. Added `__post_init__` to compute from blocks.

#### Performance Fixes
- **`extend_sequence` iterates all blocks** (`kv_cache.py`) -- Only the last block can ever have free slots. Changed to check only the last block.
- **O(n*m) scheduler cleanup** (`scheduler.py`) -- `list.remove(req)` in a loop. Replaced with set-based filtering via list comprehension.
- **Sampler corrupts caller's logits** (`sampler.py`) -- `apply_temperature` uses `logits.div_()` in-place, which corrupts the caller's tensor when it shares memory with model outputs. Added `logits.clone()` at pipeline entry in `sample_token`.
- **`_decode_batch` processes cancelled requests** (`engine.py`) -- Cancelled requests in a batch consumed GPU compute. Added filtering before the forward pass.

#### Cleanup
- Removed dead `_prefix_len` backward-compat alias from `GenerationRequest` (`types.py`).
- Renamed misleading `max_prompt_len` to `max_seq_len` in `_decode_batch` (`engine.py`).
- Removed misleading "no logits.clone needed" comment (`sampler.py`).

#### Test Fixes (8 pre-existing failures resolved)
- Fixed `test_stream_callback_path` -- tested obsolete cursor-based streaming logic; updated to match current single-token decode behavior.
- Fixed 3 scheduler hash tests -- asserted `hash()` values but code uses SHA-256; updated to test structural properties.
- Fixed 3 speculative KV mock tests -- mock KV structure was `((tensor,),)` (1-tuple) but `_trim_target_kv` expects `((key, value),)` (2-tuple).
- Fixed `test_with_blocks` -- covered by `SequenceBlocks.__post_init__` fix.

#### Test Expansion (231 -> 275 tests, +44)
- **`test_model_loader.py`** [NEW] (17 tests) -- Quantization config building (NF4, INT8, GPTQ, AWQ, dtype), KV param extraction (Llama, GPT, computed head_dim), device map resolution, lifecycle management.
- **`test_engine.py`** (+9 tests) -- Cancelled request handling, request completion states, stream finished signals, token properties, prefill completion.
- **`test_kv_cache.py`** (+12 tests) -- `extend_sequence` (last-block fill, overflow, unknown seq, OOM, multiple), prefix cache (no match, promote/match, idempotent, reset).
- **`test_scheduler.py`** (+5 tests) -- Prefix hash consistency (same prefix, cumulative, large blocks, high cardinality).
- **`test_sampler.py`** (+2 tests) -- Logits non-mutation verification, greedy fast-path validation.

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
