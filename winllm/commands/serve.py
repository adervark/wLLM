"""Start the OpenAI-compatible API server."""

from __future__ import annotations

from .common import build_model_config, apply_auto_config


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    from ..config import ServerConfig, SchedulerConfig, KVCacheConfig
    from ..api_server import create_app
    from .. import __version__

    model_config = build_model_config(args)

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
        apply_auto_config(model_config, scheduler_config, kv_cache_config)

    tp_str = f", tp={model_config.tensor_parallel_size}" if model_config.tensor_parallel_size > 1 else ""
    offload_str = ", cpu_offload=on" if model_config.cpu_offload else ""

    print(f"""
WinLLM v{__version__}
--------------------------------------------------------------
  Model:    {args.model}
  Quant:    {model_config.quantization.value}
  Device:   {model_config.device_map_strategy}{tp_str}{offload_str}
  Server:   http://{args.host}:{args.port}
--------------------------------------------------------------
""")

    app = create_app(model_config, server_config, scheduler_config, kv_cache_config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
