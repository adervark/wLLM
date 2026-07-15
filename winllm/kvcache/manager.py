"""KV cache manager facade for memory-aware scheduling.

This is a logical manager — it tracks allocations and memory budgets
so the scheduler can make admission decisions. The actual KV cache
tensors are managed internally by HuggingFace's model.

Composes three single-responsibility collaborators:
  - KVMemoryEstimator: how many blocks fit in VRAM
  - BlockAllocator:    ref-counted block bookkeeping
  - PrefixCache:       promoted prompt prefixes
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import KVCacheConfig
from .blocks import BlockAllocator, KVBlock
from .estimator import KVMemoryEstimator
from .prefix import PrefixCache

logger = logging.getLogger(__name__)


class KVCacheManager:
    """Manages KV cache block allocation and GPU memory budget."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.block_size = config.block_size

        self.estimator = KVMemoryEstimator(config)
        self.allocator = BlockAllocator(config.block_size)
        self.prefix_cache = PrefixCache()

        self.max_total_blocks = self.estimator.estimate_max_blocks()

        logger.info(
            "KVCacheManager initialized: block_size=%d, max_blocks=%d (~%d tokens capacity)",
            self.block_size,
            self.max_total_blocks,
            self.max_total_blocks * self.block_size,
        )

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def update_model_params(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
    ):
        """Update KV estimation with actual model dimensions and recalculate."""
        self.config.num_layers = num_layers
        self.config.num_kv_heads = num_kv_heads
        self.config.head_dim = head_dim
        self.config.dtype_bytes = dtype_bytes
        self.max_total_blocks = self.estimator.estimate_max_blocks()
        logger.info(
            "KV cache re-estimated with model params: max_blocks=%d",
            self.max_total_blocks,
        )

    @property
    def num_free_blocks(self) -> int:
        return self.max_total_blocks - self.allocator.allocated_blocks

    @property
    def utilization(self) -> float:
        if self.max_total_blocks == 0:
            return 0.0
        return self.allocator.allocated_blocks / self.max_total_blocks

    def can_allocate(self, num_tokens: int) -> bool:
        return self.allocator.blocks_needed(num_tokens) <= self.num_free_blocks

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def allocate_sequence(
        self, seq_id: str, num_tokens: int, prefix_blocks: Optional[list[KVBlock]] = None
    ) -> bool:
        return self.allocator.allocate(
            seq_id, num_tokens, self.num_free_blocks, prefix_blocks=prefix_blocks
        )

    def extend_sequence(self, seq_id: str, additional_tokens: int) -> bool:
        if seq_id not in self.allocator.sequences:
            return self.allocate_sequence(seq_id, additional_tokens)
        return self.allocator.extend(seq_id, additional_tokens, self.num_free_blocks)

    def free_sequence(self, seq_id: str):
        self.allocator.free(seq_id)
        logger.debug(
            "Released sequence %s, utilization now %.1f%%",
            seq_id, self.utilization * 100,
        )

    # ------------------------------------------------------------------
    # Prefix caching
    # ------------------------------------------------------------------

    def match_prefix(self, prefix_hashes: list[int]) -> tuple[list[KVBlock], Optional[tuple]]:
        """Find the longest matching prefix sequence of blocks and its tensors."""
        return self.prefix_cache.match(prefix_hashes)

    def promote_prefix_chain(
        self, prefix_hashes: list[int], seq_id: str, per_block_kv: list[tuple]
    ):
        """Promote complete prompt blocks as a cumulative-prefix chain.

        ``prefix_hashes[i]`` is the cumulative hash of blocks ``0..i`` and
        ``per_block_kv[i]`` holds *only* block ``i``'s KV tensors (already carved
        by ``slice_prompt_blocks``). Each physical block is pinned exactly once
        (by the first hash that introduces it), so shared prefixes don't inflate
        the allocated-block accounting. This is the single promotion path; tensor
        carving lives in the caller, accounting lives here.
        """
        if seq_id not in self.allocator.sequences:
            return

        seq_blocks = self.allocator.sequences[seq_id].blocks
        n = min(len(prefix_hashes), len(per_block_kv), len(seq_blocks))
        for i in range(n):
            prefix_hash = prefix_hashes[i]
            if prefix_hash in self.prefix_cache:
                continue
            # Pin only the new block this hash introduces; lower blocks were
            # pinned by their own (shorter) cumulative hash.
            self.allocator.pin_block(seq_blocks[i])
            parent_hash = prefix_hashes[i - 1] if i > 0 else None
            self.prefix_cache.store(
                prefix_hash, seq_blocks[: i + 1], per_block_kv[i], parent_hash
            )

        self._evict_prefix_cache()
        logger.debug("Promoted prefix chain of up to %d blocks for seq %s", n, seq_id)

    @property
    def max_prefix_cache_blocks(self) -> int:
        """Block budget for the prefix cache (a fraction of the KV budget)."""
        return max(0, int(self.max_total_blocks * self.config.prefix_cache_block_fraction))

    def _evict_prefix_cache(self):
        """Shrink the prefix cache to its budget, unpinning evicted blocks."""
        budget = self.max_prefix_cache_blocks
        while self.prefix_cache.size > budget:
            block = self.prefix_cache.pop_lru_leaf()
            if block is None:
                break  # every entry still has a dependent child
            self.allocator.unpin_block(block)

    def promote_to_prefix(self, prefix_hash: int, seq_id: str, num_tokens: int, past_key_values: tuple):
        """Promote a single prompt block (thin wrapper over the chain path).

        Retained for callers/tests that promote one block at a time; ``num_tokens``
        is accepted for backward compatibility but only the first block is stored.
        """
        self.promote_prefix_chain([prefix_hash], seq_id, [past_key_values])

    # ------------------------------------------------------------------
    # Introspection / reset
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "total_blocks": self.max_total_blocks,
            "allocated_blocks": self.allocator.allocated_blocks,
            "free_blocks": self.num_free_blocks,
            "utilization": round(self.utilization, 3),
            "active_sequences": len(self.allocator.sequences),
            "total_cached_tokens": self.allocator.total_cached_tokens(),
        }

    def reset(self):
        self.allocator.reset()
        self.prefix_cache.reset()
        logger.info("KV cache reset")

    # ------------------------------------------------------------------
    # White-box accessors (used by tests/diagnostics)
    # ------------------------------------------------------------------

    @property
    def _sequences(self):
        return self.allocator.sequences

    @property
    def _block_pool(self):
        return self.allocator.block_pool

    @property
    def _allocated_blocks(self) -> int:
        return self.allocator.allocated_blocks

    @property
    def _next_block_id(self) -> int:
        return self.allocator.next_block_id

    @property
    def _prefix_cache_blocks(self):
        return self.prefix_cache.blocks_by_hash

    @property
    def _prefix_cache_tensors(self):
        return self.prefix_cache.tensors_by_hash
