"""Interactive chat in the terminal."""

from __future__ import annotations

from .common import build_model_config, apply_auto_config


def cmd_chat(args):
    """Interactive chat in the terminal."""
    from ..config import SamplingParams, SchedulerConfig, KVCacheConfig
    from ..engine import InferenceEngine
    from ..types import GenerationRequest
    from ..utils import format_chat_prompt

    model_config = build_model_config(args)
    kv_cache_config = KVCacheConfig()

    if args.auto_config:
        scheduler_config = SchedulerConfig()
        apply_auto_config(model_config, scheduler_config, kv_cache_config)

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
