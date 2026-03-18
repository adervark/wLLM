"""Run a simple throughput benchmark."""

from __future__ import annotations

from .common import build_model_config, apply_auto_config


def cmd_benchmark(args):
    """Run a simple throughput benchmark."""
    from ..config import SamplingParams, SchedulerConfig, KVCacheConfig
    from ..engine import InferenceEngine
    from ..types import GenerationRequest
    from ..device import get_aggregate_gpu_memory

    model_config = build_model_config(args)
    kv_cache_config = KVCacheConfig()
    scheduler_config = SchedulerConfig()

    if args.auto_config:
        apply_auto_config(model_config, scheduler_config, kv_cache_config)

    engine = InferenceEngine(model_config, kv_cache_config)
    print(f"Loading model: {args.model} ({model_config.quantization.value})...")
    engine.load_model()

    mem = get_aggregate_gpu_memory()
    print(f"\nGPU memory: {mem}")
    print(f"Running benchmark ({args.num_prompts} prompts, {args.max_tokens} max tokens each)...\n")

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to sort a list using quicksort.",
        "What are the main differences between TCP and UDP?",
        "Describe the process of photosynthesis step by step.",
        "Write a haiku about artificial intelligence.",
    ]

    sampling = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    total_tokens = 0
    total_time = 0.0

    for i in range(min(args.num_prompts, len(prompts))):
        request = GenerationRequest(
            prompt=prompts[i],
            sampling_params=sampling,
        )
        result = engine.generate(request)
        total_tokens += result.generation_tokens
        total_time += result.elapsed

        print(f"  Prompt {i+1}: {result.generation_tokens} tokens in {result.elapsed:.2f}s ({result.tokens_per_second:.1f} tok/s)")

    print(f"\n{'='*50}")
    print(f"  Total: {total_tokens} tokens in {total_time:.2f}s")
    print(f"  Average: {total_tokens/total_time:.1f} tokens/sec")
    print(f"  GPU memory: {get_aggregate_gpu_memory()}")

    engine.unload_model()
