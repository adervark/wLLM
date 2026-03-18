# wLLM Architecture & Design

This document provides a comprehensive, visual guide to the software architecture, design patterns, and internal workflows of the WinLLM inference engine. It is designed to help contributors quickly understand how the system is organized, how data flows through the engine, and how memory is managed.

> [!NOTE] Table of Contents
> - [1. High-Level System Architecture](#1-high-level-system-architecture)
> - [2. The Request Lifecycle Flow](#2-the-request-lifecycle-flow)
> - [3. Dynamic Memory & Hardware Management](#3-dynamic-memory--hardware-management)
> - [4. Class & Data Flow Diagram](#4-class--data-flow-diagram)
> - [5. Component Deep Dive](#5-component-deep-dive)

---

## 1. High-Level System Architecture

At its core, WinLLM is divided into three main layers: the API Layer (FastAPI), the Core Engine (Request Scheduling & Memory Management), and the Inference Engine (PyTorch Generation Loop).

```mermaid
flowchart TB
    %% Definitions
    Client([API Client / HTTP])
    
    subgraph APILayer ["API Layer Async Thread"]
        Server["api_server.py (FastAPI App)"]
        Router["Endpoints (/v1/chat/completions)"]
        Lifespan["Lifespan Context Manager"]
    end
    
    subgraph CoreEngine ["Core Management Thread (Async)"]
        Schedule["scheduler.py (Request Queue)"]
        KVManager["kv_cache.py (Dynamic VRAM)"]
        Config["config.py (Unified Defaults)"]
    end
    
    subgraph InferenceLayer ["Inference Loop Thread (Background)"]
        Loop["Inference Loop (Continuous Batching)"]
        SpecEngine["speculative.py (Speculative Decoding)"]
        Engine["engine.py (Batched Generation)"]
        Loader["model_loader.py (Draft Support)"]
        Sample["sampler.py (Logits to Tokens)"]
    end

    subgraph AutoConfig ["Hardware & Model Auto-Tuning"]
        Hardware["device.py (Hardware Detection)"]
        Registry["registry.py (Model Profile)"]
    end
    
    %% Relationships
    Client <-->|REST / SSE Streams| Router
    Server --> Lifespan
    Lifespan -.->|Trigger load/unload| Loader
    
    Router -->|Creates GenerationRequest| Schedule
    Schedule <-->|Checks admission| KVManager
    Schedule -->|Centralized Loop| Loop
    
    Loop -->|Calls generate_step| Engine
    Loop <-->|Verification Loop| SpecEngine
    
    Engine <-->|Queries Weights| Loader
    Engine <-->|Computes next token| Sample
    Engine -.->|Claims/Frees Blocks| KVManager
    Engine -.->|Updates Request State| Schedule

    Hardware -->|Builds Defaults| Config
    Config -->|Applies overrides| Registry
```

---

## 2. The Request Lifecycle Flow

When a user submits a prompt, it travels exactly through this pipeline:

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API Server (api_server.py)
    participant S as Scheduler (scheduler.py)
    participant L as Inference Loop (Background Thread)
    participant E as Engine (engine.py)
    participant K as KV Manager (kv_cache.py)

    C->>A: POST /v1/chat/completions (prompt)
    A->>S: submit(GenerationRequest)
    Note over S: Add to _waiting queue
    
    loop Every 100ms or on New Request
        S->>K: can_allocate(new_req)?
        alt Enough VRAM
            S->>K: allocate_sequence()
            S->>L: Admit to _active_reqs
        end
        
        Note over L: ITERATION STEP
        L->>E: generate_step(batch)
        Note over E: 1. Prefill for new reqs
        Note over E: 2. Decode for existing reqs
        E-->>L: batch (updated tokens)
        
        alt Speculative Enabled
            L->>L: Speculative Verification Loop
        end
        
        Note over L: Update Request Status & Stream
        L-->>A: [Callback] Stream latest tokens
        A-->>C: SSE chunks
        
        alt Finished / Cancelled
            L->>K: free_sequence(req_id)
            L->>S: Move to _completed
        end
    end
```

---

## 3. Dynamic Memory & Hardware Management

WinLLM handles memory entirely mathematically at runtime, rather than relying on hardcoded rules.

### Hardware Detection Pipeline

```mermaid
flowchart LR
    A["device.py - Hardware Detection"] --> B["Aggregate VRAM and GPU Count"]
    B --> C{Total VRAM?}
    C -- "Under 16 GB" --> D["Quantization = 4bit"]
    C -- "16 GB or more" --> E["Quantization = none"]
    
    B --> F["Max Batch Size Calculation"]
    B --> G["Context Length Tiering"]
    
    B --> H{Compute Capability?}
    H -- "8.0 or higher" --> I["Backend = flash_attention_2"]
    H -- "Lower than 8.0" --> J["Backend = sdpa"]
    
    D & E & F & G & I & J --> K((HardwareDefaults))
    K --> L["Process Environment Overrides"]
    L --> M["Apply to Model Config"]
```

### The Paged Attention (KV Cache) Simulator

Because pure PyTorch doesn't natively support memory paging like vLLM does, `kv_cache.py` simulates block-level allocation. When the scheduler receives a request, the `KVCacheManager`:

1. Uses actual model parameters (`num_layers`, `num_kv_heads`, `head_dim`) to compute precise token byte costs.
2. Checks remaining available system VRAM via `_get_total_available_vram()`.
3. Pre-allocates a percentage (default 90%) into logical blocks of 16 tokens.
4. Tells the scheduler if there is enough block space to fit the incoming prompt + generation.

---

## 4. Class & Data Flow Diagram

This diagram illustrates how core classes interact, and how data structures (like configs and requests) are passed throughout the system.

```mermaid
classDiagram
    %% Core Data Structures
    class ModelConfig {
        +str model_name_or_path
        +str draft_model_name_or_path
        +bool compile
        +QuantizationType quantization
        +apply_hardware_defaults(defaults)
    }
    
    class SamplingParams {
        +int max_tokens
        +float temperature
        +float top_p
        +int top_k
        +float repetition_penalty
    }
    
    class GenerationRequest {
        +str request_id
        +list[int] output_token_ids
        +RequestStatus status
        +tuple _past_key_values
        +int _prefix_len
    }

    class HardwareDefaults {
        +int max_batch_size
        +int max_model_len
        +str attention_backend
        +float kv_cache_fraction
    }
    
    %% Core Managers
    class InferenceEngine {
        +ModelConfig model_config
        +KVCacheManager kv_cache_manager
        +SpeculativeEngine speculative_engine
        +generate_step(requests)
    }
    
    class Scheduler {
        +deque _waiting
        +list _active_reqs
        +Thread _loop_thread
        +submit(request)
        -_run_inference_loop()
    }
    
    class KVCacheManager {
        +allocate_sequence(seq_id, tokens)
        +extend_sequence(seq_id, tokens)
        +free_sequence(seq_id)
    }
    
    class ModelLoader {
        +ModelConfig config
        +load() Model, Tokenizer
        +get_kv_cache_params() dict
        -_resolve_device_map()
    }

    class SpeculativeEngine {
        +PreTrainedModel target_model
        +PreTrainedModel draft_model
        +step(request)
    }
    
    %% Relationships and Data Flow
    ModelConfig <-- InferenceEngine : Contains
    ModelConfig <-- ModelLoader : Uses
    HardwareDefaults ..> ModelConfig : Applies overrides
    
    SamplingParams <-- GenerationRequest : Contains
    GenerationRequest <-- Scheduler : Batches
    GenerationRequest <-- InferenceEngine : Processes & Modifies
    GenerationRequest <-- SpeculativeEngine : Modifies
    
    InferenceEngine *-- KVCacheManager : Initializes & Calls
    InferenceEngine *-- ModelLoader : Initializes & Calls
    InferenceEngine *-- SpeculativeEngine : Initializes
    Scheduler o-- InferenceEngine : Calls generate_step()
    
    %% API entry point
    class APIServer {
        +chat_completions(req)
        +completions(req)
        -_stream_response()
    }
    
    APIServer --> GenerationRequest : Creates
    APIServer --> Scheduler : Submits via submit()
```

---

## 5. Component Deep Dive

### [`api_server.py`](../winllm/api_server.py) | The Gateway
- Emulates standard OpenAI REST API.
- Implements FastAPI's modern `@asynccontextmanager` `lifespan` hook. The model is loaded onto the GPU during startup, and gracefully unloaded during shutdown (Ctrl+C).
- Handles streaming by acting as an asynchronous bridge to the synchronous PyTorch loops. Uses `asyncio.Queue` and `loop.call_soon_threadsafe()`.
- Catches GPU timeouts and injects JSON-formatted error chunks securely into the SSE stream.

### [`scheduler.py`](../winllm/scheduler.py) | The Task Orchestrator
- **Continuous Batching**: No longer uses a semaphore for simple concurrency. Instead, it maintains a background `_loop_thread` that constantly attempts to admit new requests into an active batch based on KV cache availability.
- **Async Interface**: Provides `submit()` and `submit_streaming()` as async interfaces, while the actual heavy lifting happens in the background thread via the `InferenceLoop`.

### [`engine.py`](../winllm/engine.py) | The Batched Inference Engine
- **`generate_step()`**: The primary entry point for inference. It takes a *list* of requests and performs one iteration of prefill or decode for all of them.
- **Integrated `torch.compile`**: Supports compiling the forward pass into optimized kernels, significantly improving throughput by reducing Python overhead in the decode iterations.

### [`speculative.py`](../winllm/speculative.py) | Accelerated Generation
- **Draft Model Logic**: Implements speculative decoding where a smaller model proposes tokens that the larger target model verifies in a single forward pass.
- **Acceptance Loop**: Dynamically adjusts the target model's KV cache and output tokens based on how many "draft" tokens were correct.

### [`kv_cache.py`](../winllm/kv_cache.py) | Logical Memory Tracker
- **Iteration-Level Allocation**: Tracks block usage across the entire batch.
- **Sequence Management**: Provides `allocate_sequence`, `extend_sequence`, and `free_sequence` methods invoked by the scheduler and engine during the generation lifecycle.

### [`cli.py`](../winllm/cli.py) & [`config.py`](../winllm/config.py) | Unified Configurations
- Centralizes Dataclasses (`ModelConfig`, `SchedulerConfig`, `KVCacheConfig`, `SamplingParams`).
- CLI params naturally cascade into Config objects. The `--auto-config` flag triggers the dynamic hardware discovery sequence, overwriting baseline constraints with optimized formulas.

### [`registry.py`](../winllm/registry.py) | Model Introspection
- Examines the HuggingFace repo name (e.g. `meta-llama/Llama-3.1-8B-Instruct`).
- Determines the architectural family (Llama, Gemma, Mistral, Qwen).
- Injects ideal hyper-parameters (e.g. `max_context_window=32768`, `rope_scaling=True`) before the tensors are ever initialized in VRAM.
