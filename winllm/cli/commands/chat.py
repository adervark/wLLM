"""Interactive chat in the terminal."""

from __future__ import annotations

from .common import build_model_config, apply_auto_config
from ..console import console
from ..formatting import RichChatRenderer


def cmd_chat(args):
    """Interactive chat in the terminal."""
    from ...config import SamplingParams, SchedulerConfig, KVCacheConfig
    from ...core.types import GenerationRequest
    from ...inference import InferenceEngine
    from ...models.chat_template import format_chat_prompt

    model_config = build_model_config(args)
    kv_cache_config = KVCacheConfig()

    if args.auto_config:
        scheduler_config = SchedulerConfig()
        apply_auto_config(model_config, scheduler_config, kv_cache_config)

    engine = InferenceEngine(model_config, kv_cache_config)

    with console.status(
        f"Loading [bold]{args.model}[/bold] ({model_config.quantization.value})..."
    ):
        engine.load_model()
    console.print("Model loaded. Type [bold]quit[/bold] or [bold]exit[/bold] to stop.\n")

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
                # Label printed separately so styling never sits inside the
                # input() prompt, where line editing can miscount columns.
                console.print("\n[wllm.user]You:[/] ", end="")
                user_input = input().strip()
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

            # Label on its own line: the renderer's live region draws
            # block-level markdown, which can't share the label's line.
            console.print("\n[wllm.assistant]Assistant:[/]")
            renderer = RichChatRenderer()

            def on_token(text: str, finished: bool):
                if finished:
                    renderer.close()
                else:
                    renderer.feed(text)

            request = GenerationRequest(
                prompt=prompt,
                sampling_params=sampling,
                _stream_callback=on_token,
            )

            # Speculation counters are lifetime totals on the engine; snapshot
            # around generate() to report this response's share.
            spec = engine.speculative_engine
            drafted_before = spec.drafted_tokens if spec else 0
            accepted_before = spec.accepted_tokens if spec else 0

            result = engine.generate(request)
            renderer.close()  # idempotent; ensures the live region is released

            if result.error:
                # Nothing streamed and there is no assistant turn to keep;
                # drop the user turn too so the failed exchange doesn't
                # poison the context of every later prompt.
                messages.pop()
                console.print(f"\n[wllm.error]Error:[/] {result.error}")
                continue

            stats = (
                f"{result.generation_tokens} tok · "
                f"{result.elapsed:.1f}s · "
                f"{result.tokens_per_second:.1f} tok/s"
            )
            if spec:
                drafted = spec.drafted_tokens - drafted_before
                accepted = spec.accepted_tokens - accepted_before
                if drafted:
                    stats += f" · spec {accepted}/{drafted} ({accepted / drafted:.0%})"
            console.print(f"[wllm.muted]── {stats}[/wllm.muted]")

            messages.append({"role": "assistant", "content": result.output_text})

    finally:
        engine.unload_model()
