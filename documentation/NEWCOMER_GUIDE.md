---
title: "NEWCOMER_GUIDE"
category: "wLLm"
tags: []
status: "Active"
created: "2026-04-01"
---
# WinLLM Maintainer Guide: From Zero to Core Contributor

Welcome to the definitive maintainer's guide for WinLLM (wLLM). If you are new to coding, PyTorch, or large language models, this document is your blueprint. 

wLLM is an OpenAI-compatible API server and inference engine. It is designed to be a Windows-native equivalent specifically engineered to be readable and hackable in pure Python.

This guide will teach you exactly how the system is structured, the mathematical concepts that operate under the hood, and how you can add massive new features to the engine securely.

---

## 1. Theoretical Concepts You Must Know

Before looking at the PyTorch files, you need to understand the three primary architectural concepts that make wLLM incredibly fast:

### Tokenization & Logits
Language models do not read text. They read numbers (Token IDs). When you send `"Hello"`, the Tokenizer converts it into `[1532]`. The Neural Network processes `[1532]` and spits out a massive array of numbers called `logits`. There is one logit for every possible word in the English language (e.g., 128,000 numbers).
The **Sampler** takes these numbers, converts them into percentages (probabilities), applies effects like Temperature, and rolls a weighted dice to pick the next Token ID.

### The KV Cache (Key-Value Cache)
If you ask an AI to write an essay, generating the 100th word normally requires the AI to re-read the first 99 words all over again. This requires massive computational bounds. 
To bypass this, wLLM uses a **KV Cache**. During the first pass, we save the mathematical "state" (the Key and Value tensors) of the 99 words into a giant pool of GPU Memory. For the 100th word, we only calculate the math for the 99th word and connect it to the saved cache.

### Continuous Batching
Processing one user at a time is slow. If three users ask a question, wLLM merges them into a single giant matrix (Batch). 
Because users finish speaking at different times, wLLM uses **Continuous Batching**. At every single generated word (iteration), the engine can dynamically eject finished users from the batch and instantly inject new users from the waiting queue without stopping the GPU!

---

## 2. Comprehensive Directory Structure

The codebase is organized into single-responsibility packages (see [SOLID_Architecture.md](SOLID_Architecture.md) for the full design rationale). Here is a breakdown of every package and its specific responsibility in the ecosystem.

### The Interfaces (Connecting to the Outside World)
These packages never do math. They just handle raw text and networking.
- `winllm/cli/` - The CLI entry point. `cli/main.py` parses arguments like `--model` or `--quantization`, and `cli/commands/` holds one module per sub-command (`chat`, `serve`, `benchmark`, `list`, `detect`, `remove`).
- `winllm/server/` - A highly concurrent `FastAPI` web server. `server/schemas.py` defines the OpenAI wire contract, `server/app.py` maps routes like `/v1/chat/completions` into internal `GenerationRequest` Python objects, and `server/streaming.py` streams words back to the client via asynchronous Server-Sent Events (SSE).
- `winllm/models/chat_template.py` - Helpers for formatting conversational "Chat Templates" (converting JSON messages like `[{"role": "user", "content": "hi"}]` into the exact string formatting expected by Llama or Mistral).

### The Core Engine (The Hot Loops)
This is where the actual token generation occurs.
- `winllm/inference/` - **The Heart.** The `InferenceEngine` facade lives in `inference/engine.py`; the actual work is split across `inference/prefill.py` (processing prompts), `inference/decode.py` (generating one token per step, including the batched fast path with pre-allocated `torch.zeros()` buffers from `inference/buffers.py`), `inference/generation.py` (the blocking single-request loop used by the CLI), and `inference/streaming.py` (token callbacks). All of them share a `ModelRuntime` (`inference/runtime.py`) holding the model, tokenizer, and device. The accelerators live alongside: `inference/cudagraph.py` (CUDA-graph decode), `inference/speculative.py` (draft-model speculation), `inference/suffix_cache.py` + `inference/suffix_speculative.py` (model-free SuffixDecoding), with the shared per-token forward in `inference/eager.py`.
- `winllm/scheduling/` - **The Traffic Cop.** `scheduling/scheduler.py` contains the infinite background loop maintaining a `self._waiting` deque and a `self._active_reqs` list. Every few milliseconds, the `AdmissionController` (`scheduling/admission.py`) asks the KV Manager how much memory is left. If space allows, it promotes waiting users to active, builds an execution Batch list, and hands it directly to the engine. Finished requests land in the `CompletedRequestStore` (`scheduling/completed.py`).
- `winllm/sampling/` - **The Mathematician.** The `sample_token()` pipeline (`sampling/sampler.py`) completely avoids GPU memory fragmentation by utilizing strictly in-place PyTorch modifications (`logits.div_()`, `.scatter_()`). The pure transforms live in `sampling/ops.py`, and each one is wrapped as a composable `LogitsProcessor` in `sampling/processors.py`. It calculates Repetition Penalties, Temperature variants, and Nucleus extraction formats to guarantee human-like text outputs. When the optional compiled module from `native/sampling/` is installed, `sampling/native.py` runs that whole pipeline as a single CUDA kernel launch (the per-launch overhead, not the math, is what costs time on Windows) — and quietly steps aside on CPU tensors, seeded requests, or when the module isn't built.
- `winllm/core/types.py` - **The Data Schemas.** Contains the `GenerationRequest` data-class. This object acts as the passport for a request, holding its prompt, active token sequence, stop parameters, and tracking metrics. The abstract interfaces (`InferenceBackend`, `LogitsProcessor`, `SchedulingPolicy`) live next door in `winllm/core/interfaces.py`.

### Memory & Hardware Management
These packages ensure your graphics card doesn't explode.
- `winllm/kvcache/` - **The RAM Manager.** A Paged-Attention style tracking pool. At startup, the `KVMemoryEstimator` (`kvcache/estimator.py`) divides all of your available GPU VRAM into hundreds of small "Blocks". When a user is talking, the ref-counted `BlockAllocator` (`kvcache/blocks.py`) leases blocks to the user. When the user disconnects, it immediately releases the blocks back into the central pool via O(1) integer tracking counters. Prompt-prefix caching lives in `kvcache/prefix.py`, and `kvcache/manager.py` is the facade everything talks to.
- `winllm/hardware/` - Queries your Windows hardware capabilities via `torch.cuda` (`hardware/info.py`). It automatically calculates mathematically optimal defaults for batch sizing, VRAM fractions, tensor-parallel sizes, and attention backends like SDPA or FlashAttention 2 (`hardware/defaults.py`), then applies them to your configs (`hardware/tuning.py`) so the user never has to configure them manually.
- `winllm/models/` & `winllm/backends/` - Handles downloading models from the internet, injecting 4-bit and 8-bit `bitsandbytes` weights (`models/quantization.py`), and applying GPU device maps (`device_map="auto"`) via `models/loader.py`. The `backends/` package allows routing inference paths completely away from PyTorch into ONNX Runtime or DirectML — each runtime is a class registered in `backends/registry.py`.

---

## 3. Code Tracing: The Lifecycle of an API Request

To truly understand how to edit the codebase, follow this trace of how a JSON payload transforms into a streamed AI response:

1. **The Web Hook:** A client POSTs to `/v1/chat/completions` in `server/app.py`.
2. **The Passport:** The FastAPI server converts the JSON into a `GenerationRequest` object (defined in `core/types.py`). It assigns a callback `req._token_callback` to listen for new words.
3. **The Submission:** The server calls `await scheduler.submit(request)`.
4. **The Queue:** The `Scheduler` (`scheduling/scheduler.py`) adds the request to the `_waiting` queue. The server thread is instantly put to sleep using `await request._completed_event.wait()` so it consumes zero CPU.
5. **The Admission:** In a separate background thread, `_run_inference_loop` asks the `AdmissionController` — which in turn asks `kvcache/manager.py` — if there are enough free memory blocks to hold the user's prompt. If so, it moves the user to `_active_reqs`.
6. **The Prefill:** The scheduler passes the batch to `engine.generate_step()`. Because it's a new user, the `PrefillRunner` (`inference/prefill.py`) tokenizes the massive prompt, pushes it entirely through the neural network in a single burst, and saves the massive mathematically calculated `past_key_values` tensor layer down into the user's allocated KV cache.
7. **The Decode Loop:** On the next iteration, the `DecodeRunner` (`inference/decode.py`) runs `decode_batch()`. It takes the last generated word, packs every user's cache into persistent pre-allocated buffer slices (`inference/buffers.py`), and feeds the single word into the model.
8. **The Probability Matrix:** The model spits out a tensor array sized `[batch_size, vocabulary_length]`. 
9. **The Sampling:** `sampling/sampler.py` is handed the matrix. It mutates the logits in-place to enforce Temperature variants, runs a multinomial softmax roll, and returns the chosen `Token IDs`.
10. **The Broadcast:** The `StreamEmitter` (`inference/streaming.py`) triggers `request._token_callback()`, and `server/streaming.py` instantly shoots an SSE HTTP packet back across the globe to the client holding the word.
11. **The Cleanup:** When the model naturally shoots out an `<|eos|>` token, the scheduler marks the request as `COMPLETED`. The KV manager frees the specific RAM blocks entirely, and `_loop.call_soon_threadsafe(event.set)` forcibly awakens the API server thread to close out the connection safely.

---

## 4. How to Safely Add Major Features

You can maintain and scale this repository entirely in standard Object-Oriented Python.

### Step 1: Follow the Interface Definitions
Whenever you want to add an option (e.g., `max_tokens` limit enforcement), always start at the entry point. Add the parameter to the `SamplingParams` dataclass inside `config/sampling.py`. 

### Step 2: Implement the Hot-Loop Logic
Use "Find in Project" to see where `SamplingParams` is consumed (usually `inference/` or `sampling/`). If you want to stop generating early, you would locate the specific `max_new_tokens` barrier logic inside `BlockingGenerator._run_decode_loop()` in `inference/generation.py`.

If your feature is a whole new *kind* of thing — a new inference backend, a new sampling technique, a new scheduling order — implement the matching interface from `core/interfaces.py` and register it (`backends/registry.py`, `sampling/processors.py`, or `scheduling/policies.py`). You should not need to edit any existing dispatch code.

### Step 3: Run the Test Suite immediately!
The wLLM engine is protected by **282 highly isolated Automated Tests**. 
If you modify a PyTorch slicing tensor boundary, or attempt to modify an asynchronous event queue, there is a very high probability you will silently cause a bug or Memory Leak.

Any time you save a `.py` file, execute this:
```bash
uv run pytest tests/ -v
```
It takes less than 7 seconds to complete. The tests will rigorously verify mathematically identical tensor parity, simulate broken streaming networks, and validate the Pydantic API schemas. If your code produces an error, PyTest will scream exactly which line of your feature is invalid!

### Step 4: Write Your Own Test
If your feature adds completely new logic (e.g., a "Ban Word" array algorithm inside `sampling/`), create a new test inside `tests/test_sampling.py`. Define a fake tensor block, pass it through your new method, and `.assert()` that the output accurately enforces the banned word index!

---

## 5. Quick Wins to Practice On

If you are looking to get your hands dirty quickly, try tracking down and modifying these specific behaviors to build your confidence:

- **Difficulty (Low): Customize the CLI Intro Text**  
  Open `winllm/cli/commands/chat.py`. Locate the `print()` function that writes `"Model loaded! Type 'quit' or 'exit' to stop."` and make it display the loaded model's architecture parameter size!

- **Difficulty (Medium): Engine Timeout Rejection**  
  If the GPU hangs, the Request objects will wait forever. Add a `timeout` float property to `GenerationRequest` inside `core/types.py`. Then head into `BlockingGenerator._run_decode_loop()` (`inference/generation.py`) and use `time.time() - request.started_at` to forcibly inject a `RequestStatus.FAILED` state if the generation surpasses `X` seconds!

- **Difficulty (High): Blacklist Tokens (Logit Processors)**  
  Open `winllm/sampling/processors.py`. Create a new `LogitsProcessor` class called `TokenBanProcessor` that uses the `logits.masked_fill_()` PyTorch operator to statically set the probability of all bad words to `float('-inf')`! Insert it into `DEFAULT_PIPELINE` right before the `multinomial` probabilistic dice-roll — no changes to the sampler itself required.
