"""Run a simple throughput benchmark."""

from __future__ import annotations

from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .common import build_model_config, apply_auto_config
from ..console import console


def cmd_benchmark(args):
    """Run a simple throughput benchmark."""
    from ...config import SamplingParams, SchedulerConfig, KVCacheConfig
    from ...core.types import GenerationRequest
    from ...hardware import get_aggregate_gpu_memory
    from ...inference import InferenceEngine

    model_config = build_model_config(args)
    kv_cache_config = KVCacheConfig()
    scheduler_config = SchedulerConfig()

    if args.auto_config:
        apply_auto_config(model_config, scheduler_config, kv_cache_config)

    engine = InferenceEngine(model_config, kv_cache_config)
    with console.status(
        f"Loading [bold]{args.model}[/bold] ({model_config.quantization.value})..."
    ):
        engine.load_model()

    mem = get_aggregate_gpu_memory()
    console.print(f"\nGPU memory: {mem}")
    console.print(
        f"Running benchmark ({args.num_prompts} prompts, "
        f"{args.max_tokens} max tokens each)...\n"
    )

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

    table = Table(title="Benchmark results", title_justify="left", border_style="wllm.border")
    table.add_column("Prompt", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("tok/s", justify="right")

    # Rows appear in the terminal as each prompt finishes.
    with Live(table, console=console, refresh_per_second=4):
        for i in range(min(args.num_prompts, len(prompts))):
            request = GenerationRequest(
                prompt=prompts[i],
                sampling_params=sampling,
            )
            result = engine.generate(request)
            total_tokens += result.generation_tokens
            total_time += result.elapsed

            table.add_row(
                str(i + 1),
                str(result.generation_tokens),
                f"{result.elapsed:.2f}s",
                f"{result.tokens_per_second:.1f}",
            )

    summary = (
        f"Total: {total_tokens} tokens in {total_time:.2f}s\n"
        f"Average: {total_tokens / total_time:.1f} tokens/sec\n"
        f"GPU memory: {get_aggregate_gpu_memory()}"
    )
    console.print(Panel(summary, title="Summary", expand=False, border_style="wllm.border"))

    engine.unload_model()
