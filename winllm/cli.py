"""CLI entry point for WinLLM."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import QuantizationType

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Shared quantization string → enum mapping
QUANT_MAP = {
    "auto": None,
    "none": QuantizationType.NONE,
    "4bit": QuantizationType.NF4,
    "8bit": QuantizationType.INT8,
}


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_common_model_args(parser):
    """Add model-related arguments shared by serve, chat, and benchmark."""
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or path")
    parser.add_argument("--quantization", "-q", choices=["auto", "none", "4bit", "8bit"], default="auto")
    parser.add_argument("--max-model-len", type=int, default=None, help="Auto-detected if not specified")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")


def _build_model_config(args):
    """Build a ModelConfig from parsed CLI args (shared by all commands)."""
    from .config import ModelConfig

    quantization = QUANT_MAP.get(args.quantization)

    kwargs = {"model_name_or_path": args.model}
    if quantization is not None:
        kwargs["quantization"] = quantization
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if hasattr(args, "gpu_memory_utilization") and args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    kwargs["device_map_strategy"] = args.device_map_strategy
    kwargs["cpu_offload"] = args.cpu_offload
    kwargs["device"] = args.device

    return ModelConfig(**kwargs)


def _apply_auto_config(model_config, scheduler_config, kv_cache_config):
    """Detect hardware and apply optimal defaults."""
    from .device import DeviceInfo
    hw = DeviceInfo.detect()

    print(f"\n  Detected: {hw.device_count} GPU(s), {hw.total_vram_gb} GB total VRAM")
    print(f"     Profile:  {hw.profile.value}")
    for gpu in hw.devices:
        print(f"     GPU {gpu.index}: {gpu.name} ({gpu.total_vram_gb} GB)")

    model_config.apply_hardware_defaults(hw.defaults)
    scheduler_config.apply_hardware_defaults(hw.defaults)
    kv_cache_config.apply_hardware_defaults(hw.defaults)

    return hw


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    from .config import ServerConfig, SchedulerConfig, KVCacheConfig
    from .api_server import create_app

    model_config = _build_model_config(args)

    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        model_alias=args.model_alias,
    )

    scheduler_config = SchedulerConfig(
        max_batch_size=args.max_batch_size,
    )

    kv_cache_config = KVCacheConfig()

    # Auto-config: detect hardware and apply optimal settings
    if args.auto_config:
        _apply_auto_config(model_config, scheduler_config, kv_cache_config)

    tp_str = f", tp={model_config.tensor_parallel_size}" if model_config.tensor_parallel_size > 1 else ""
    offload_str = ", cpu_offload=on" if model_config.cpu_offload else ""

    print(f"""
WinLLM v0.1.0
--------------------------------------------------------------
  Model:    {args.model}
  Quant:    {model_config.quantization.value}
  Device:   {model_config.device_map_strategy}{tp_str}{offload_str}
  Server:   http://{args.host}:{args.port}
--------------------------------------------------------------
""")

    app = create_app(model_config, server_config, scheduler_config, kv_cache_config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_chat(args):
    """Interactive chat in the terminal."""
    from .config import SamplingParams, SchedulerConfig, KVCacheConfig
    from .engine import InferenceEngine, GenerationRequest
    from .utils import format_chat_prompt

    model_config = _build_model_config(args)
    kv_cache_config = KVCacheConfig()

    if args.auto_config:
        scheduler_config = SchedulerConfig()
        _apply_auto_config(model_config, scheduler_config, kv_cache_config)

    engine = InferenceEngine(model_config, kv_cache_config)

    print(f"Loading model: {args.model} ({model_config.quantization.value})...")
    engine.load_model()
    print("Model loaded! Type 'quit' or 'exit' to stop.\n")

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            prompt = format_chat_prompt(engine.tokenizer, messages)

            print("\nAssistant: ", end="", flush=True)
            collected = []

            def on_token(text: str, finished: bool):
                if not finished:
                    print(text, end="", flush=True)
                    collected.append(text)

            request = GenerationRequest(
                prompt=prompt,
                sampling_params=sampling,
                _stream_callback=on_token,
            )

            result = engine.generate(request)
            print()

            messages.append({"role": "assistant", "content": result.output_text})

    finally:
        engine.unload_model()


def cmd_benchmark(args):
    """Run a simple throughput benchmark."""
    from .config import SamplingParams, SchedulerConfig, KVCacheConfig
    from .engine import InferenceEngine, GenerationRequest
    from .device import get_aggregate_gpu_memory

    model_config = _build_model_config(args)
    kv_cache_config = KVCacheConfig()
    scheduler_config = SchedulerConfig()

    if args.auto_config:
        _apply_auto_config(model_config, scheduler_config, kv_cache_config)

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


def cmd_list(args):
    """List downloaded models from the HuggingFace cache."""
    import os
    from pathlib import Path
    from datetime import datetime

    # Resolve cache directory
    cache_dir = Path(os.environ.get(
        "HF_HOME",
        os.environ.get("HUGGINGFACE_HUB_CACHE",
                       Path.home() / ".cache" / "huggingface" / "hub"),
    ))

    # HF_HOME points to the root; the actual model repos are under hub/
    if (cache_dir / "hub").is_dir():
        cache_dir = cache_dir / "hub"

    if not cache_dir.is_dir():
        print(f"Cache directory not found: {cache_dir}")
        print("No models downloaded yet.")
        return

    # Scan for model directories (they start with "models--")
    model_dirs = sorted([
        d for d in cache_dir.iterdir()
        if d.is_dir() and d.name.startswith("models--")
    ])

    if not model_dirs:
        print("No models found in cache.")
        return

    # Collect model info
    models = []
    for d in model_dirs:
        # Convert "models--meta-llama--Llama-2-7b-chat-hf" → "meta-llama/Llama-2-7b-chat-hf"
        name = d.name.removeprefix("models--").replace("--", "/")

        # Calculate total size
        total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())

        # Get last modified time
        try:
            mtime = max(f.stat().st_mtime for f in d.rglob("*") if f.is_file())
            modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            modified = "unknown"

        models.append((name, total_bytes, modified))

    # Format sizes
    def _fmt_size(b: int) -> str:
        if b >= 1024 ** 3:
            return f"{b / (1024 ** 3):.1f} GB"
        elif b >= 1024 ** 2:
            return f"{b / (1024 ** 2):.1f} MB"
        else:
            return f"{b / 1024:.0f} KB"

    # Print table
    name_width = max(len(m[0]) for m in models)
    name_width = max(name_width, 5)  # min width for "MODEL" header

    header_model = "MODEL".ljust(name_width)
    print(f"\n  {header_model}   {'SIZE':>10}   {'MODIFIED':>16}")
    print(f"  {'─' * name_width}   {'─' * 10}   {'─' * 16}")

    total_size = 0
    for name, size_bytes, modified in models:
        total_size += size_bytes
        print(f"  {name.ljust(name_width)}   {_fmt_size(size_bytes):>10}   {modified:>16}")

    print(f"\n  {len(models)} model(s), {_fmt_size(total_size)} total")
    print(f"  Cache: {cache_dir}\n")


def cmd_detect(args):
    """Detect and display hardware info."""
    from .device import DeviceInfo
    import json

    hw = DeviceInfo.detect()

    print(f"\nHardware Detection")
    print(f"--------------------------------------------------------------")

    if hw.device_count == 0:
        print(f"  No GPUs detected — CPU-only mode")
    else:
        for gpu in hw.devices:
            print(f"  GPU {gpu.index}: {gpu.name} ({gpu.total_vram_gb} GB)")

    print(f"\n  Profile:     {hw.profile.value}")
    print(f"  Platform:    {hw.platform}")
    print(f"  Total VRAM:  {hw.total_vram_gb}")
    
    print(f"\nRecommended defaults:")
    d = hw.defaults
    print(f"  Quantization:       {d.default_quantization}")
    print(f"  Max batch size:     {d.max_batch_size}")
    print(f"  Max context length: {d.max_model_len}")
    print(f"  Tensor parallel:    {d.tensor_parallel_size}")
    print(f"  Device map:         {d.device_map_strategy}")
    print(f"--------------------------------------------------------------")

    if args.json:
        print(f"\nJSON:\n{json.dumps(hw.summary(), indent=2)}")


def _add_scaling_args(parser):
    """Add common scaling arguments to a subparser."""
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--device-map-strategy", choices=["auto", "balanced", "balanced_low_0", "sequential"],
                        default="auto", help="How to distribute model across GPUs")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload excess layers to CPU RAM")
    parser.add_argument("--device", default="auto",
                        help="Device to use: auto, cuda, cuda:0, cpu")
    parser.add_argument("--auto-config", action="store_true",
                        help="Auto-detect hardware and set optimal config")


def main():
    parser = argparse.ArgumentParser(
        prog="winllm",
        description="WinLLM — Windows-native LLM inference engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    _add_common_model_args(serve_parser)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", "-p", type=int, default=8000)
    serve_parser.add_argument("--max-batch-size", type=int, default=4)
    serve_parser.add_argument("--model-alias", default=None, help="Override model name in API responses")
    serve_parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    _add_scaling_args(serve_parser)

    # --- chat ---
    chat_parser = subparsers.add_parser("chat", help="Interactive chat in terminal")
    _add_common_model_args(chat_parser)
    chat_parser.add_argument("--max-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--system-prompt", "-s", default=None, help="System prompt")
    _add_scaling_args(chat_parser)

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run throughput benchmark")
    _add_common_model_args(bench_parser)
    bench_parser.add_argument("--max-tokens", type=int, default=256)
    bench_parser.add_argument("--num-prompts", type=int, default=5)
    _add_scaling_args(bench_parser)

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List downloaded models from HuggingFace cache")
    list_parser.add_argument("--verbose", "-v", action="store_true")

    # --- detect ---
    detect_parser = subparsers.add_parser("detect", help="Detect and display hardware info")
    detect_parser.add_argument("--json", action="store_true", help="Also print JSON output")
    detect_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(getattr(args, "verbose", False))

    cmd_map = {"serve": cmd_serve, "chat": cmd_chat, "benchmark": cmd_benchmark, "list": cmd_list, "detect": cmd_detect}
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
