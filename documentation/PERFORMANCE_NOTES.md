---
title: "Performance Notes: Decode-Path CPU Overhead on Windows"
category: "wLLm"
tags: ["performance", "profiling", "wddm", "native"]
status: "Active"
created: "2026-07-06"
---
# Performance Notes: Decode-Path CPU Overhead on Windows

Research log for the 2026-07-06 and 2026-07-07 optimization passes on the
blocking (chat) decode path. Read this before optimizing anything in the
per-token loop — the profile results here are counter-intuitive and the
rules at the end exist because violating them costs ~1 ms *per violation
per token* on Windows.

## Headline results

SmolLM2-135M-Instruct fp16, RTX 4070 Laptop (8 GB), Windows 11 (WDDM),
torch 2.11.0+cu128, chat path with `--cuda-graphs --suffix-decoding`,
"list Item 1..30" prompt, 256-token cap, mean of 4 runs:

| configuration | temp 0.0 | temp 0.7 |
|---|---|---|
| before (commit 59c6629) | 86.3 tok/s | 86.2 tok/s |
| + sampling rewrite | 95.9 tok/s | 104.6 tok/s |
| + batched verify sampling | ~93 tok/s (noise) | **112.4–113.3 tok/s** |

Temp-0 output is deterministic (208 tokens, token-identical before/after).
All 385 tests pass at every step.

## Method

`cProfile` around one `engine.generate()` (after a warmup generation for
graph capture), sorted by `tottime` and `cumulative`, plus a clean
un-profiled wall-clock pass. Scripts were scratch files; the recipe is:
build the engine exactly as `cmd_chat` does, run once to warm, time 3–4
generations, then profile one.

## Finding: the GPU was idle-waiting on Python — but not where expected

The profile of one ~90-token generation:

- **`torch.tensor`: 123 calls, 169 ms — 55% of all decode wall time.**
- CUDA graph replay (the actual model): 18 calls, 15 ms.
- The suffix cache (pure Python) didn't register at all (µs range).

Every `sample_token()` call built ~6 small tensors from Python lists
(penalties, generated-id set, temperatures, top-k values, top-p values,
greedy mask) — each `torch.tensor(..., device="cuda")` is a synchronous
host→device copy that must wait for everything queued on the stream, and
each `if torch.all(...)` skip-check is a device→host sync. On Windows the
GPU is driven through the WDDM display driver model, where each such
submission/sync carries far higher fixed cost than on Linux/TCC — a
well-documented gap (see references). Net effect: ~9 ms of sampling
overhead per token against a 0.8 ms model forward.

## Fix 1: sampling rewrite (`winllm/sampling/ops.py`, `sampler.py`)

Rules now encoded in `ops.py`, valid for every op in the per-token path:

1. **Decide skips in Python.** The parameter lists are already Python;
   `all(p == 1.0 for p in penalties)` is free, `if torch.all(tensor)` is a
   sync.
2. **Uniform parameters use scalar paths.** Single-request chat always has
   one temperature/top-k/top-p — `logits.div_(0.7)` and
   `shifted_probs >= 0.7` broadcast Python floats; no tensor, no transfer.
3. **Repetition penalty is gather/scatter on the generated ids** (one
   small transfer) instead of a vocab-sized bool mask plus two
   `torch.where` passes.
4. **`nan_to_num` runs unconditionally** — the `isnan().any()` guard was a
   per-step sync costing more than the kernel it tried to skip.
5. **Mixed-batch fallbacks still exist** (per-row tensors, built on CPU
   then moved) — they're only taken when parameters actually differ.

After the rewrite: `torch.tensor` fell to 61 calls / 2 ms, and
`sample_token` from ~12 ms to ~1.5 ms per call (profiled cumulative).

## Fix 2: batched verify sampling (`inference/generation.py`)

`_speculative_commit` sampled each verify row sequentially —
`sample_token(...).item()` per draft = one device sync per draft token.
When `repetition_penalty == 1.0` (grammar requests never reach
speculation), row distributions are independent of acceptance, so all
rows are now sampled in one batched call and materialized with a single
`.tolist()`. Rows past the first rejection are discarded, which cannot
change committed output — the same argument vLLM's batch verification
uses. The sequential path remains for penalty ≠ 1.0.

Caveat found in review: the discard argument fails for **seeded** temp>0
requests — batching consumes multinomial draws for the discarded rows
too, desyncing the generator stream from a speculation-off run and
breaking seeded reproducibility (the losslessness invariant, observed as
generator-state divergence after the first verify with a rejection).
Seeded sampling requests therefore take the sequential path (one draw
per committed token, exactly like plain decode); unseeded and greedy
requests — including the default chat path — still batch, so the
measured wins above are unaffected. Pinned by
`test_seeded_sampling_consumes_rng_like_plain_decode`.

## Fix 3: native C++ suffix cache (`native/suffix/`)

pybind11 port of `SuffixCache` (bigram index + backward extension), built
with MSVC via `uv pip install ./native/suffix`. Wiring:

- `winllm/inference/suffix_cache.py` rebinds `SuffixCache` to the native
  class when `winllm_suffix` is importable; `PySuffixCache` is always the
  pure-Python fallback (CI never needs a compiler).
- `WINLLM_PURE_PYTHON=1` forces the fallback.
- `tests/test_suffix_decoding.py::TestNativeParity` drives both
  implementations through 400 randomized sync/propose steps (small vocab,
  tight caps, LRU eviction in play) and requires identical proposals.
  The behavioral tests also run against whichever implementation is
  active.
- `sync()` converts only the unseen tail of the token list across the
  boundary, keeping it O(new tokens) per step.

Micro-benchmark (4000-token stream, sync+propose per step):

| stream | Python | C++ |
|---|---|---|
| random (no matches) | 7.2 µs/step | 5.9 µs/step |
| repetitive (match-heavy) | 31.6 µs/step | 6.1 µs/step |

**Honest assessment:** at ~6 µs/step this was never the chat-path
bottleneck (the profile said so before the port). Its value is that C++
time is *flat* regardless of match density, so `max_occurrences` /
`max_back_ext` can be raised for better draft quality at zero CPU cost —
and it's the template for future native modules.

## Fix 4: adaptive verify throttle (`inference/generation.py`)

On templated output ("Item 1 ... Item 2 ...") the repeated surface pattern
always continues with a *new* value, so suffix drafts are deterministically
wrong: 0% acceptance, and every token still paid a wasted verify replay.
`_VerifyThrottle` applies exponential backoff after each fully-rejected
verify (skip 1, 2, 4, ... capped at 32 opportunities) and snaps back to
full speculation on a single acceptance. Pathological text now pays ~zero
speculation overhead; repetitive text is unaffected. Measured: 89.1 → 97.0
tok/s at temp 0 on the adversarial prompt (same session, old vs new).

## Negative result: pinned staging buffers made it 5× SLOWER

The "obvious" next step after the sampling rewrite was to remove the two
remaining per-step H2D writes in `CUDAGraphDecoder.decode()` (input token,
position) by staging them in pinned host memory and issuing
`copy_(non_blocking=True)`. Measured result: **93 → 19 tok/s** — a 5×
collapse, output still token-identical. Async memcpys into a CUDA graph's
private memory pool evidently serialize catastrophically under WDDM,
while the plain `buf[0, 0] = token_id` assignments compile to scalar fill
kernels that stay cheap. The revert carries a NOTE comment in
`cudagraph.py` so nobody re-attempts it without re-measuring. Lesson:
on WDDM, *measure every* "known good" CUDA idiom — the driver model
inverts several of them.

## What the benchmark prompt teaches about suffix decoding

At temp 0 the "Item 1..30" prompt shows **0% acceptance (0/232)**: the
repeated surface pattern (`Item N: value N\n`) always continues with a
*different* number, so the first draft token is deterministically wrong.
Suffix drafting needs verbatim repetition (JSON keys, quoted spans,
agentic traces — the paper's target workloads), not templated repetition.
Speculation stayed lossless throughout (token-identical temp-0 output);
a rejected draft costs only the verify replay it already spent.

## 2026-07-07 pass: fused CUDA sampling kernel (`native/sampling/`)

Levers #3 and #4 from the 2026-07-06 list, done together in one extension.

### Toolchain: why the module is torch-free

The installed toolkit is CUDA 13.2 while torch is built against cu12.8;
`torch.utils.cpp_extension` refuses a major-version mismatch outright. The
extension therefore includes **no torch headers at all**: Python passes raw
`tensor.data_ptr()` addresses and `torch.cuda.current_stream().cuda_stream`
into a pybind11 binding, the binding is compiled by MSVC, the kernels by
nvcc (`setup.py` runs nvcc itself and hands the objects to the normal
setuptools link; `-ccbin` points at the cl.exe distutils locates, since
MSVC isn't on PATH), and cudart is linked statically so the .pyd needs
only the driver. Same-stream discipline makes the raw-pointer lifetime
story identical to normal torch ops. Build:
`uv pip install ./native/sampling` (~10 s).
**Gotcha:** when iterating on the .cu, add `--no-cache` — uv re-served a
stale cached wheel after a source edit once during this work, which cost a
debugging detour.

### Kernel design (`winllm_sampling.cu`)

One launch per `sample_token` call replaces the ~15-launch torch pipeline
(penalty → temperature → top-k → top-p → softmax → multinomial); under
WDDM the launches, not the math, were the cost. One block per row, all
math fp32, caller-provided `[batch, 2, vocab]` fp32 scratch, caller logits
never mutated:

- repetition penalty scattered over the (wrapper-deduped) generated ids;
- exact top-k threshold by 4-pass MSB **radix select on the float bit
  patterns** of the probabilities (positive floats order like their
  uint32 views), tie-keeping like `ops.py`'s `logits < threshold` mask;
- nucleus boundary by the same radix walk on per-bucket **mass**
  histograms (exact up to fp32 summation order; value ties at the
  boundary are all kept, where torch keeps a sort-order-dependent subset);
- categorical draw via Philox (seed = process-random, subsequence = row,
  offset = call counter) over thread-contiguous partial sums, so the
  drawn interval always contains its target exactly;
- temperature 0 = penalized argmax, first index on ties.

Eligibility (decided in `sampler.py`): CUDA logits, uniform sampling
params across the batch, and **no request generator** — the kernel's RNG
is not the request's seeded torch generator, so seeded temp>0 requests
stay on the torch path to preserve seeded reproducibility (greedy never
draws, so seeded greedy still fuses). Failures latch the module off and
fall back to torch.

### Correctness detour: the model decodes in bf16, not fp16

Temp-0 output initially diverged from the torch path by one token — a
near-tie where torch's penalized logit was 33.75 from a raw 37.0 at
penalty 1.1 (37/1.1 = 33.636: impossible in fp16, whose ulp there is
0.03125). The logits are **bfloat16** (ulp 0.25 at that magnitude — the
"fp16" in the 2026-07-06 headline was wrong), and torch computes the
penalty in the tensor dtype. The kernel now rounds penalized values back
through the source dtype; temp-0 output is token-identical to the torch
path again (verified full-generation, fused vs torch).

### Measured

`sample_token` microbench (SmolLM2 vocab 49152, bf16, 200 generated ids,
mean of 300 calls after warmup):

| case | torch | fused |
|---|---|---|
| greedy + penalty 1.1 | 0.238 ms | 0.068 ms |
| temp 0.7, chat defaults | 0.965 ms | 0.343 ms |
| temp 0.7, penalty 1.0 | 0.887 ms | 0.227 ms |
| verify batch of 8, no penalty | 1.015 ms | 0.224 ms |

End-to-end chat path (interleaved A/B in one process — separate-process
runs drift several tok/s with laptop GPU clocks; fixed 128-token output):
temp 0.7 **87.9 → 99.4 tok/s** (+13%); temp 0 neutral (95.7 vs 93.9,
within noise) with bit-identical tokens — the greedy path was never
sampling-bound.

### Grammar bitmask kernel (lever #4, same module)

`wls_apply_bitmask` applies xgrammar's packed int32 bitmask to a logits
row in one launch, replacing the unpack-to-bool + `masked_fill_` chain in
`sampling/grammar.py` (the pure-torch unpack remains the CPU/CI
fallback). Grammar-constrained rows also qualify for the fused sampler
(the mask is just -inf logits by then), so structured output gets both
kernels: measured 54.5 → 60.5 tok/s (+11%) on `json_object` output,
SmolLM2 chat path, temp 0.7.

## Remaining levers (priority order)

1. **Pipelined decode:** copy the sampled token device-to-device into the
   graph input buffer and enqueue the next replay *before* the `.item()`
   for stop checks — overlaps the forward with CPU bookkeeping; rollback
   for the overshoot step is the same lossless position rewind the verify
   path uses.
2. **Pinned staging buffers** for the transfers that remain (penalty ids,
   graph inputs).
3. ~~Fused CUDA sampling kernel~~ — done 2026-07-07 (`native/sampling/`),
   see above. Note it was built standalone (pybind11 + nvcc), *not* via
   `torch.utils.cpp_extension`, because of the CUDA 13.2 / cu12.8 torch
   mismatch.
4. ~~Grammar bitmask kernel~~ — done 2026-07-07, same module.
5. **Cross-request suffix tree** (the paper's global tree) for
   multi-request workloads with shared structure.

## References

- SuffixDecoding (Oliaro et al., NeurIPS 2025 Spotlight):
  [arXiv:2411.04975](https://arxiv.org/abs/2411.04975),
  [project page](https://suffix-decoding.github.io/)
- WDDM vs TCC/MCDM submission & transfer overhead:
  [NVIDIA forums — WDDM vs TCC](https://forums.developer.nvidia.com/t/will-microsoft-windows-mcdm-improve-the-wddm-vs-tcc-situation/310058),
  [NVIDIA/cuda-python#1207](https://github.com/NVIDIA/cuda-python/issues/1207),
  [microsoft/graphics-driver-samples#103](https://github.com/microsoft/graphics-driver-samples/issues/103)
- [pybind11](https://pybind11.readthedocs.io/) (`Pybind11Extension`, MSVC
  via setuptools)
