"""KV cache configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    """Configuration for KV cache management."""
    block_size: int = 16
    max_blocks_per_seq: int = 256
    gpu_memory_fraction: float = 0.4
    # Cap the prefix cache at this fraction of the total KV block budget so
    # pinned prefixes can't monotonically starve the live KV cache. Evicted
    # via leaf-only LRU (see PrefixCache).
    prefix_cache_block_fraction: float = 0.25
    # --- Model-aware estimation (set at runtime) ---
    num_layers: int = 0           # Populated from loaded model
    num_kv_heads: int = 0         # Populated from loaded model
    head_dim: int = 0             # Populated from loaded model
    dtype_bytes: int = 2          # 2 for float16, 4 for float32
