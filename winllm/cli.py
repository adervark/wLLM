"""CLI entry point for WinLLM."""

from __future__ import annotations

import argparse
import logging
import sys

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _apply_auto_config(model_config, scheduler_config, kv_cache_config):
    """Detect hardware and apply optimal defaults."""
    from .device import DeviceInfo
    hw = DeviceInfo.detect()

    print(f"\n  🔍 Detected: {hw.device_count} GPU(s), {hw.total_vram_gb} GB total VRAM")
    print(f"     Profile:  {hw.profile.value}")
    for gpu in hw.devices:
        print(f"     GPU {gpu.index}: {gpu.name} ({gpu.total_vram_gb} GB)")

    model_config.apply_hardware_defaults(hw.defaults)
    scheduler_config.apply_hardware_defaults(hw.defaults)
    kv_cache_config.apply_hardware_defaults(hw.defaults)

    print(f"     Auto-config → quant={model_config.quantization.value}, "
          f"batch={scheduler_config.max_batch_size}, "
          f"ctx={model_config.max_model_len}, "
          f"tp={model_config.tensor_parallel_size}")
    return hw


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    from .config import ModelConfig, QuantizationType, ServerConfig, SchedulerConfig, KVCacheConfig
    from .api_server import create_app

    quant_map = {"auto": None, "none": QuantizationType.NONE, "4bit": QuantizationType.NF4, "8bit": QuantizationType.INT8}
    quantization = quant_map.get(args.quantization)

    model_config_kwargs = {"model_name_or_path": args.model}
    if quantization is not None:
        model_config_kwargs["quantization"] = quantization
    if args.max_model_len is not None:
        model_config_kwargs["max_model_len"] = args.max_model_len
    if args.trust_remote_code:
        model_config_kwargs["trust_remote_code"] = True
    if args.gpu_memory_utilization is not None:
        model_config_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        
    model_config = ModelConfig(
        **model_config_kwargs,
        tensor_parallel_size=args.tensor_parallel_size,
        device_map_strategy=args.device_map_strategy,
        cpu_offload=args.cpu_offload,
        device=args.device,
    )

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
╔══════════════════════════════════════════════════════════════╗
║                      🚀 WinLLM v0.1.0                      ║
╠══════════════════════════════════════════════════════════════╣
║  Model:    {args.model:<48s}║
║  Quant:    {model_config.quantization.value:<48s}║
║  Device:   {model_config.device_map_strategy}{tp_str}{offload_str:<{48 - len(tp_str) - len(offload_str)}}║
║  Server:   http://{args.host}:{args.port:<39}║
╚══════════════════════════════════════════════════════════════╝
""")

    app = create_app(model_config, server_config, scheduler_config, kv_cache_config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_chat(args):
    """Interactive chat in the terminal."""
    from .config import ModelConfig, QuantizationType, SamplingParams, SchedulerConfig, KVCacheConfig
    from .engine import InferenceEngine, GenerationRequest

    quant_map = {"none": QuantizationType.NONE, "4bit": QuantizationType.NF4, "8bit": QuantizationType.INT8}
    quantization = quant_map.get(args.quantization, QuantizationType.NF4)

    model_config_kwargs = {
        "model_name_or_path": args.model,
        "quantization": quantization,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cpu_offload": args.cpu_offload,
        "device": args.device,
    }
    if args.max_model_len is not None:
        model_config_kwargs["max_model_len"] = args.max_model_len

    model_config = ModelConfig(**model_config_kwargs)

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
                user_input = input("\n🧑 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            try:
                prompt = engine.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                parts = []
                for msg in messages:
                    if msg["role"] == "system":
                        parts.append(f"System: {msg['content']}")
                    elif msg["role"] == "user":
                        parts.append(f"User: {msg['content']}")
                    elif msg["role"] == "assistant":
                        parts.append(f"Assistant: {msg['content']}")
                parts.append("Assistant:")
                prompt = "\n".join(parts)

            print("\n🤖 Assistant: ", end="", flush=True)
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

            print(
                f"  [{result.generation_tokens} tokens, "
                f"{result.tokens_per_second:.1f} tok/s, "
                f"{result.elapsed:.2f}s]"
            )

            messages.append({"role": "assistant", "content": result.output_text})

    finally:
        engine.unload_model()


def cmd_benchmark(args):
    """Run a simple throughput benchmark."""
    from .config import ModelConfig, QuantizationType, SamplingParams, SchedulerConfig, KVCacheConfig
    from .engine import InferenceEngine, GenerationRequest
    from .model_loader import get_aggregate_gpu_memory

    quant_map = {"auto": None, "none": QuantizationType.NONE, "4bit": QuantizationType.NF4, "8bit": QuantizationType.INT8}
    quantization = quant_map.get(args.quantization)

    model_config_kwargs = {"model_name_or_path": args.model}
    if quantization is not None:
        model_config_kwargs["quantization"] = quantization
    if args.max_model_len is not None:
        model_config_kwargs["max_model_len"] = args.max_model_len
    if args.trust_remote_code:
        model_config_kwargs["trust_remote_code"] = True
        
    model_config = ModelConfig(
        **model_config_kwargs,
        tensor_parallel_size=args.tensor_parallel_size,
        cpu_offload=args.cpu_offload,
        device=args.device,
    )

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


def cmd_detect(args):
    """Detect and display hardware info."""
    from .device import DeviceInfo
    import json

    hw = DeviceInfo.detect()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   🔍 Hardware Detection                     ║
╠══════════════════════════════════════════════════════════════╣""")

    if hw.device_count == 0:
        print(f"║  No GPUs detected — CPU-only mode                          ║")
    else:
        for gpu in hw.devices:
            name_line = f"  GPU {gpu.index}: {gpu.name} ({gpu.total_vram_gb} GB)"
            print(f"║{name_line:<62}║")

    print(f"║{'':62s}║")
    print(f"║  Profile:     {hw.profile.value:<47s}║")
    print(f"║  Platform:    {hw.platform:<47s}║")
    print(f"║  Total VRAM:  {hw.total_vram_gb:<47.1f}║")
    print(f"║{'':62s}║")
    print(f"║  Recommended defaults:{'':40s}║")
    d = hw.defaults
    print(f"║    Quantization:       {d.default_quantization:<38s}║")
    print(f"║    Max batch size:     {d.max_batch_size:<38d}║")
    print(f"║    Max context length: {d.max_model_len:<38d}║")
    print(f"║    Tensor parallel:    {d.tensor_parallel_size:<38d}║")
    print(f"║    Device map:         {d.device_map_strategy:<38s}║")

    print(f"╚══════════════════════════════════════════════════════════════╝")

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
    serve_parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or path")
    serve_parser.add_argument("--quantization", "-q", choices=["auto", "none", "4bit", "8bit"], default="auto")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", "-p", type=int, default=8000)
    serve_parser.add_argument("--max-model-len", type=int, default=None, help="Auto-detected if not specified")
    serve_parser.add_argument("--max-batch-size", type=int, default=4)
    serve_parser.add_argument("--model-alias", default=None, help="Override model name in API responses")
    serve_parser.add_argument("--trust-remote-code", action="store_true")
    serve_parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    serve_parser.add_argument("--verbose", "-v", action="store_true")
    _add_scaling_args(serve_parser)

    # --- chat ---
    chat_parser = subparsers.add_parser("chat", help="Interactive chat in terminal")
    chat_parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or path")
    chat_parser.add_argument("--quantization", "-q", choices=["auto", "none", "4bit", "8bit"], default="auto")
    chat_parser.add_argument("--max-model-len", type=int, default=None)
    chat_parser.add_argument("--max-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--system-prompt", "-s", default=None, help="System prompt")
    chat_parser.add_argument("--trust-remote-code", action="store_true")
    chat_parser.add_argument("--verbose", "-v", action="store_true")
    _add_scaling_args(chat_parser)

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run throughput benchmark")
    bench_parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or path")
    bench_parser.add_argument("--quantization", "-q", choices=["auto", "none", "4bit", "8bit"], default="auto")
    bench_parser.add_argument("--max-model-len", type=int, default=None)
    bench_parser.add_argument("--max-tokens", type=int, default=256)
    bench_parser.add_argument("--num-prompts", type=int, default=5)
    bench_parser.add_argument("--trust-remote-code", action="store_true")
    bench_parser.add_argument("--verbose", "-v", action="store_true")
    _add_scaling_args(bench_parser)

    # --- detect ---
    detect_parser = subparsers.add_parser("detect", help="Detect and display hardware info")
    detect_parser.add_argument("--json", action="store_true", help="Also print JSON output")
    detect_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(getattr(args, "verbose", False))

    cmd_map = {"serve": cmd_serve, "chat": cmd_chat, "benchmark": cmd_benchmark, "detect": cmd_detect}
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
