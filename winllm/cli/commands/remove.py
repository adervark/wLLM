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

    if getattr(args, "all", False):
        model_dirs = sorted([
            d for d in cache_dir.iterdir()
            if d.is_dir() and d.name.startswith("models--")
        ])
        
        if not model_dirs:
            print("No models found in cache.")
            return

        print(f"Found {len(model_dirs)} model(s) in cache:")
        for d in model_dirs:
            name = d.name.removeprefix("models--").replace("--", "/")
            print(f"  {name}")
            
        confirm = input(f"\nAre you sure you want to delete ALL these models and free up their space? [y/N]: ")
        if confirm.strip().lower() in ['y', 'yes']:
            for target_dir in model_dirs:
                folder_name = target_dir.name
                print(f"Deleting {folder_name}...")
                try:
                    shutil.rmtree(target_dir)
                except Exception as e:
                    print(f"Failed to delete directory {folder_name}: {e}")
            print("Successfully deleted all selected models!")
        else:
            print("Deletion cancelled.")
        return

    model_id = args.model
    if not model_id:
        print("Please specify a model ID to remove (e.g., 'wllm remove mistralai/Mistral-7B-v0.1') or use '--all'.")
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
