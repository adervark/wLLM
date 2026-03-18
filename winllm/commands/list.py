"""List downloaded models from the HuggingFace cache."""

from __future__ import annotations


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
    print(f"  {'-' * name_width}   {'-' * 10}   {'-' * 16}")

    total_size = 0
    for name, size_bytes, modified in models:
        total_size += size_bytes
        print(f"  {name.ljust(name_width)}   {_fmt_size(size_bytes):>10}   {modified:>16}")

    print(f"\n  {len(models)} model(s), {_fmt_size(total_size)} total")
    print(f"  Cache: {cache_dir}\n")
