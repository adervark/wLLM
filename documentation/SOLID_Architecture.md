# wLLM Architecture — SOLID by Construction

This document describes the package layout introduced in the SOLID restructure
and explains which principle each boundary serves. If you are adding a feature,
start here: the structure tells you where new code belongs.

---

## Package Map

```
winllm/
├── core/            Domain layer — types + abstractions everything depends on
│   ├── types.py         GenerationRequest, RequestStatus
│   └── interfaces.py    InferenceBackend, LogitsProcessor, SchedulingPolicy, StoppingCriterion
│
├── config/          Pure-data configuration (one module per subsystem)
│   ├── model.py         ModelConfig, QuantizationType, DType
│   ├── sampling.py      SamplingParams
│   ├── scheduling.py    SchedulerConfig
│   ├── kv_cache.py      KVCacheConfig
│   └── server.py        ServerConfig
│
├── hardware/        What hardware exists and what it implies
│   ├── info.py          DeviceInfo / GPUInfo detection + SystemProfile
│   ├── defaults.py      HardwareDefaults + build_defaults + env overrides
│   ├── tuning.py        apply_*_defaults(config, defaults) — config tuning
│   ├── memory.py        GPU memory introspection (MemoryUtils)
│   └── cuda.py          Process-global CUDA backend setup (TF32, allocator)
│
├── backends/        How weights get loaded/executed (pluggable)
│   ├── registry.py      BackendRegistry — name → backend class
│   ├── pytorch.py       PyTorchBackend
│   ├── onnx.py          OnnxRuntimeBackend (+ pure build_ort_kwargs routing)
│   ├── directml.py      DirectMLBackend
│   ├── tokenizers.py    Shared tokenizer loading (with -ONNX fallback)
│   └── compat.py        'Day zero' architecture aliasing
│
├── models/          Model knowledge and load orchestration
│   ├── loader.py        ModelLoader — sequences a load via a backend
│   ├── quantization.py  HF quantization config builders
│   ├── introspection.py KV-dimension extraction from model configs
│   ├── profiles.py      Known model-family heuristics (llama, mistral, ...)
│   └── chat_template.py Chat messages → prompt string
│
├── kvcache/         Logical KV cache accounting
│   ├── blocks.py        KVBlock, SequenceBlocks, BlockAllocator (ref-counted)
│   ├── estimator.py     KVMemoryEstimator — blocks that fit in VRAM
│   ├── prefix.py        compute_prefix_hashes, slice_prompt_blocks, PrefixCache storage
│   └── manager.py       KVCacheManager — facade composing the three above
│
├── sampling/        Token sampling
│   ├── ops.py           Pure logit transforms (penalty, temperature, top-k/p)
│   ├── processors.py    LogitsProcessor classes + DEFAULT_PIPELINE
│   ├── grammar.py       GrammarBackend / GrammarState — xgrammar-constrained decoding (structured output)
│   ├── native.py        FusedSampler — optional fused CUDA kernel (native/sampling/), torch fallback
│   └── sampler.py       sample_token entry point (fast paths + fused kernel + pipeline + grammar mask)
│
├── inference/       Running the model
│   ├── runtime.py       ModelRuntime — shared loaded-model state
│   ├── buffers.py       DecodeInputBuffer, PersistentBatchCache (cross-step batched KV)
│   ├── prefill.py       PrefillRunner (single / chunked / batched)
│   ├── decode.py        DecodeRunner (single / batched / speculative dispatch)
│   ├── generation.py    BlockingGenerator — full single-request loop
│   ├── eager.py         eager_decode_step — the one shared single-token forward
│   ├── speculative.py   SpeculativeEngine (draft model + verify)
│   ├── suffix_cache.py  SuffixCache — model-free draft source (SuffixDecoding; optional native drop-in)
│   ├── suffix_speculative.py  SuffixSpeculativeEngine — suffix drafts + verify + rollback probe
│   ├── cudagraph.py     CUDAGraphDecoder — graph-captured decode + eager-fallback self-check
│   ├── streaming.py     StreamEmitter — token → callback delivery
│   └── engine.py        InferenceEngine — facade wiring it all together
│
├── scheduling/      Continuous batching
│   ├── scheduler.py     Scheduler — the background inference loop
│   ├── policies.py      SchedulingPolicy implementations (FCFS) + registry
│   ├── admission.py     AdmissionController — memory-aware + prefix matching
│   ├── completed.py     CompletedRequestStore — TTL/count-bounded retention
│   └── stats.py         SchedulerStats
│
├── server/          OpenAI-compatible HTTP layer
│   ├── app.py           create_app — FastAPI factory + routes
│   ├── schemas.py       Pydantic wire contract
│   ├── metrics.py       Prometheus text-format rendering for /metrics
│   └── streaming.py     SSE token streaming
│
└── cli/             Command-line interface
    ├── main.py          Argument parsing + dispatch (winllm.cli:main)
    ├── formatting.py    ThinkTagParser (splits streamed deltas into thinking/answer segments) and RichChatRenderer (renders them as live markdown via rich)
    └── commands/        One module per command (serve, chat, benchmark, ...)
```

Dependencies point inward: `server`/`cli` → `scheduling` → `inference` →
(`models`, `kvcache`, `sampling`) → (`backends`, `hardware`) → `config`/`core`.
`core` depends on nothing but `config`.

Outside the package, the repo-level `native/` directory holds the optional
compiled accelerators (`native/suffix`: C++ suffix cache; `native/sampling`:
fused CUDA sampling + grammar bitmask kernels). They are separate
pip-installable projects, not part of the `winllm` dependency graph — the
package only ever *imports them optionally* (`inference/suffix_cache.py`,
`sampling/native.py`) and falls back to the pure implementations when they
aren't built, so CI and zero-compiler installs are unaffected.

---

## How Each SOLID Principle Shows Up

### S — Single Responsibility
Every module above has one reason to change. The old `engine.py` (827 lines)
mixed lifecycle, prefill, decode, buffer management, streaming, and a blocking
generation loop; those are now seven small collaborators behind the
`InferenceEngine` facade. Likewise the old `scheduler.py` mixed queueing,
admission, eviction, hashing, and statistics — now four collaborators behind
`Scheduler`.

### O — Open/Closed
The extension points are registries and pipelines, so new behavior is added by
*adding* code, not editing dispatch logic:

- **New inference backend** → subclass `InferenceBackend`, call
  `default_registry.register(MyBackend)` (`backends/registry.py`).
  Nothing in `ModelLoader` changes. (This is exactly where a future
  `LlamaCppBackend` for GGUF would plug in.)
- **New sampling technique** → write a `LogitsProcessor` class and insert it
  into a pipeline (`sampling/processors.py`).
- **New scheduling order** → implement `SchedulingPolicy`, add it to
  `POLICIES` (`scheduling/policies.py`); selected via
  `SchedulerConfig.scheduling_policy`.

### L — Liskov Substitution
All backends return the same `(model, tokenizer)` contract and are freely
interchangeable wherever `InferenceBackend` is expected; all policies honor the
`select`/`requeue` contract. No caller type-checks for a concrete class.

### I — Interface Segregation
Components receive only the state they need. `PrefillRunner`/`DecodeRunner`
get a `ModelRuntime` (model, tokenizer, device, KV manager) — they never see
scheduling, serving, or lifecycle concerns. `AdmissionController` sees only
the `KVCacheManager`. `StreamEmitter` sees only the tokenizer side of runtime.

### D — Dependency Inversion
High-level orchestrators depend on the abstractions in `core/interfaces.py`,
not concretions: `ModelLoader` asks the registry for *an* `InferenceBackend`;
`Scheduler` admits through *a* `SchedulingPolicy`. Configuration objects are
pure data and never import services — hardware tuning was moved out of the
config classes into `hardware/tuning.py` so `config/` has zero outward
dependencies.

---

## Key Flows (unchanged behavior, new homes)

**Serving a request:**
`server/app.py` route → `scheduling/scheduler.py` queue →
`scheduling/admission.py` (prefix match + KV budget) →
`inference/engine.py::generate_step` → `inference/prefill.py` or
`inference/decode.py` → `sampling/sampler.py` → tokens stream back through
`inference/streaming.py` → `server/streaming.py` SSE.

**Loading a model:**
`inference/engine.py::load_model` → `hardware/cuda.py` global setup →
`models/loader.py` (quantization config + device map) →
`backends/registry.py` → concrete backend → KV capacity estimated by
`kvcache/estimator.py`.

**CLI chat (no scheduler):**
`cli/commands/chat.py` → `inference/engine.py::generate` →
`inference/generation.py::BlockingGenerator` (validate → prefill → decode
loop → finalize).

---

## Conventions

- Heavy imports (`torch`, `transformers`, `optimum`) are deferred to call
  sites where practical; `import winllm` stays cheap (lazy `__getattr__`
  exports in `winllm/__init__.py`).
- `KVCacheManager` exposes `_sequences`, `_block_pool`, etc. as read-only
  properties purely for white-box tests/diagnostics; production code must use
  the public methods.
- The `InferenceEngine` no longer carries `_`-prefixed delegation shims for
  its collaborators; tests target the collaborators directly
  (`_blocking_generator`, `_decode_runner`, `_emitter`, `_runtime`).
