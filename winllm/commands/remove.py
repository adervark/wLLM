"""Remove a downloaded model from the HuggingFace cache."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def cmd_remove(args):
    """Remove a downloaded model from the HuggingFace cache."""
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
        return

    model_id = args.model
    if not model_id:
        print("Please specify a model ID to remove, e.g., 'wllm remove mistralai/Mistral-7B-v0.1'")
        return

    # Convert "meta-llama/Llama-2-7b-chat-hf" -> "models--meta-llama--Llama-2-7b-chat-hf"
    folder_name = "models--" + model_id.replace("/", "--")
    target_dir = cache_dir / folder_name

    if not target_dir.exists():
        print(f"Model '{model_id}' is not in your cache.")
        return

    print(f"Found model cache for '{model_id}' at:")
    print(f"  {target_dir}")
    
    confirm = input(f"\nAre you sure you want to delete this model and free up its space? [y/N]: ")
    if confirm.strip().lower() in ['y', 'yes']:
        print(f"Deleting {folder_name}...")
        try:
            shutil.rmtree(target_dir)
            print("Successfully deleted!")
        except Exception as e:
            print(f"Failed to delete directory: {e}")
    else:
        print("Deletion cancelled.")
