---
title: "Bottlenecks"
category: "wLLm"
tags: []
status: "Active"
created: "2026-04-01"
---
making wLLM as fast as vLLM or Ollama requires me to make a purpose built intermediary that would have to handle and sit between the program and CUDA kernels basically the GPU

## The Zero-Day Wedge
wLLM's founding functional advantage over incumbents (Ollama, LM Studio) is **Zero-Day Model Support**.
- Because we use raw HuggingFace weights without conversion (GGUF/Exl2), we can run any new architecture the moment it drops.
- Custom-kernel engines (vLLM etc.) still own the raw-throughput ceiling — that gap requires C++/CUDA/Triton kernels in the *model forward*, which wLLM deliberately avoids because they're per-architecture work that kills the wedge.

## What Narrowed the Gap (2026-07)
Techniques that reclaim real ground *without* giving up the zero-day wedge — each is model-agnostic, validates itself at load time (or falls back automatically), and never touches architecture-specific code:
- **CUDA graph decode** (`--cuda-graphs`): ~7–8× single-request decode by eliminating Python kernel-launch overhead — a raw CUDA driver feature, no compiler required.
- **SuffixDecoding** (`--suffix-decoding`): model-free speculation drafted from the request's own text; 1.1–1.9× on repetitive/structured output with token-identical results.
- **Structured output** (xgrammar): guaranteed-valid JSON from any zero-day model — a capability, not just a speedup.
- **WDDM-aware sampling** (`sampling/ops.py` rewrite, 2026-07-06): on Windows the per-token cost is stream submissions/syncs, not math — eliminating per-step device-tensor creation and GPU boolean checks took the chat path from 86 to ~106 tok/s alone.
- **Optional native kernels** (`native/`, 2026-07-07): the line "wLLM avoids C++" now has a precise boundary — native code is fine *around* the model (sampling, grammar masks, draft caches) where one implementation serves every architecture, just never *inside* it. The fused CUDA sampling kernel (+13% at temp 0.7), grammar bitmask kernel (+11% on JSON mode), and C++ suffix cache are all opt-in builds with automatic pure-torch/Python fallback, so the zero-compiler install still works and CI never needs nvcc.
- Measured on this stack: `torch.compile` (via community triton-windows) *loses* to the CUDA-graph path (~80 vs ~98 tok/s on SmolLM2-135M) and costs a ~2-minute warmup — on Windows, graphs + targeted kernels beat whole-model compilation.