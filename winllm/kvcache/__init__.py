"""KV cache management, split by responsibility (SRP):

  - blocks.py:    block bookkeeping data structures + allocator
  - estimator.py: VRAM-based capacity estimation
  - prefix.py:    prefix hashing + cached-prefix storage
  - manager.py:   the facade the engine and scheduler talk to
"""

from .blocks import BlockAllocator, KVBlock, SequenceBlocks
from .estimator import KVMemoryEstimator
from .prefix import PrefixCache, compute_prefix_hashes, slice_prompt_blocks
from .manager import KVCacheManager

__all__ = [
    "KVBlock",
    "SequenceBlocks",
    "BlockAllocator",
    "KVMemoryEstimator",
    "PrefixCache",
    "compute_prefix_hashes",
    "slice_prompt_blocks",
    "KVCacheManager",
]
