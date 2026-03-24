"""CLI entry point for WinLLM."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import QuantizationType
from .commands import cmd_serve, cmd_chat, cmd_benchmark, cmd_list, cmd_detect, cmd_remove
from . import __version__

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


def _add_common_model_args(parser):
    """Add model-related arguments shared by serve, chat, and benchmark."""
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or path")
    parser.add_argument("--quantization", "-q", choices=["auto", "none", "4bit", "8bit"], default="auto")
    parser.add_argument("--max-model-len", type=int, default=None, help="Auto-detected if not specified")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")


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
        prog="wLLM",
        description="wLLM — Windows-native LLM inference engine",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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

    # --- remove ---
    remove_parser = subparsers.add_parser("remove", help="Remove a downloaded model from HuggingFace cache")
    remove_parser.add_argument("model", help="Model ID to remove, e.g., 'mistralai/Mistral-7B-v0.1'")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(getattr(args, "verbose", False))

    cmd_map = {
        "serve": cmd_serve,
        "chat": cmd_chat,
        "benchmark": cmd_benchmark,
        "list": cmd_list,
        "detect": cmd_detect,
        "remove": cmd_remove
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
