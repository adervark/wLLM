"""wLLM decode-rate benchmark for LFM2.5-1.2B-Thinking (vs llama.cpp).

Decode-only tok/s = (tokens - 1) / (finished - first_token), directly
comparable to llama.cpp's "eval time" rate. Hybrid arch: no CUDA graphs,
no suffix decoding (rollback probe rejects it); fused sampling kernel
engages automatically on the CUDA logits.
"""
import statistics
import torch
from winllm.config import ModelConfig, KVCacheConfig, SamplingParams, QuantizationType
from winllm.core.types import GenerationRequest
from winllm.inference import InferenceEngine
from winllm.sampling.native import fused_sampler

engine = InferenceEngine(
    ModelConfig(
        model_name_or_path="LiquidAI/LFM2.5-1.2B-Thinking",
        quantization=QuantizationType.NONE,
    ),
    KVCacheConfig(),
)
engine.load_model()
print("fused sampling kernel enabled:", fused_sampler.enabled)

from winllm.models.chat_template import format_chat_prompt
prompt = format_chat_prompt(engine.tokenizer, [
    {"role": "user", "content": "Write a long story about a robot exploring an abandoned city."}])

def run(tag):
    import time
    params = SamplingParams(temperature=0.7, max_tokens=256)
    req = GenerationRequest(request_id=tag, prompt=prompt, sampling_params=params)
    first = None

    def cb(text, finished):
        nonlocal first
        if first is None and not finished:
            first = time.perf_counter()

    req._stream_callback = cb
    t0 = time.perf_counter()
    r = engine.generate(req)
    t1 = time.perf_counter()
    n = len(r.output_token_ids)
    return n, (n - 1) / (t1 - first), len(r.prompt_token_ids) / (first - t0)

run("warmup")
rates, pp = [], []
for i in range(5):
    n, tg, p = run(f"run{i}")
    rates.append(tg)
    pp.append(p)
    print(f"  run {i}: {n} tok, decode {tg:.1f} tok/s, prefill {p:.0f} tok/s")
print(f"wLLM LFM2.5-1.2B bf16: decode mean {statistics.mean(rates):.1f} tok/s "
      f"(median {statistics.median(rates):.1f}), prefill mean {statistics.mean(pp):.0f} tok/s")
print(f"dtype of weights: {next(engine._runtime.model.parameters()).dtype}")
