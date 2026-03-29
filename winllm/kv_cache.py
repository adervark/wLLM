"""KV Cache block manager for memory-aware scheduling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import KVCacheConfig

logger = logging.getLogger(__name__)


@dataclass
class KVBlock:
    """A single KV cache block tracking token slots."""
    block_id: int
    num_tokens: int = 0
    max_tokens: int = 16
    ref_count: int = 0 # Track how many sequences/prefixes use this block

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.max_tokens

    @property
    def free_slots(self) -> int:
        return self.max_tokens - self.num_tokens


@dataclass
class SequenceBlocks:
    """Tracks all blocks allocated for a single sequence."""
    seq_id: str
    blocks: list[KVBlock] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(b.num_tokens for b in self.blocks)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


class KVCacheManager:
    """Manages KV cache block allocation and GPU memory budget.

    This is a logical manager — it tracks allocations and memory budgets
    so the scheduler can make admission decisions. The actual KV cache
    tensors are managed internally by HuggingFace's model.

    Supports model-aware estimation when KV dimensions are provided,
    and aggregates VRAM across multiple GPUs.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.block_size = config.block_size
        self._next_block_id = 0

        # Calculate available blocks based on GPU memory
        # seq_id -> SequenceBlocks
        self._sequences: dict[str, SequenceBlocks] = {}

        # prefix_hash -> list[KVBlock]
        self._prefix_cache_blocks: dict[int, list[KVBlock]] = {}
        
        # prefix_hash -> tuple[tuple[torch.Tensor]] (physical past_key_values)
        self._prefix_cache_tensors: dict[int, tuple] = {}

        # block_id -> KVBlock (for global ref management)
        self._block_pool: dict[int, KVBlock] = {}

        self.max_total_blocks = self._estimate_max_blocks()
        self._allocated_blocks: int = 0 # This will be updated by _update_allocated_count

        logger.info(
            "KVCacheManager initialized: block_size=%d, max_blocks=%d (~%d tokens capacity)",
            self.block_size,
            self.max_total_blocks,
            self.max_total_blocks * self.block_size,
        )

    def _estimate_per_token_kv_bytes(self) -> int:
        """Estimate bytes consumed per token in KV cache.

        Uses model dimensions if available, otherwise falls back to
        a conservative estimate for a ~7B parameter model.
        """
        if (
            self.config.num_layers > 0
            and self.config.num_kv_heads > 0
            and self.config.head_dim > 0
        ):
            # Precise calculation from model architecture
            # KV per token = 2 (key+value) * num_layers * num_kv_heads * head_dim * dtype_bytes
            per_token = (
                2
                * self.config.num_layers
                * self.config.num_kv_heads
                * self.config.head_dim
                * self.config.dtype_bytes
            )
            logger.info(
                "KV cache per-token estimate (model-aware): %d bytes "
                "(layers=%d, kv_heads=%d, head_dim=%d, dtype=%dB)",
                per_token,
                self.config.num_layers,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.dtype_bytes,
            )
            return per_token
        else:
            # Fallback estimate for ~7B model: ~512 KB per token
            logger.info("KV cache per-token estimate (fallback): 524288 bytes")
            return 512 * 1024

    def _get_total_available_vram(self) -> float:
        """Get total available VRAM in bytes across all GPUs."""
        if not torch.cuda.is_available():
            return 0.0

        total_available = 0.0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            total_available += props.total_memory - allocated

        return total_available

    def _estimate_max_blocks(self) -> int:
        """Estimate how many KV blocks we can afford based on GPU memory."""
        if not torch.cuda.is_available():
            return self.config.max_blocks_per_seq * 4  # Fallback for CPU

        available = self._get_total_available_vram() * self.config.gpu_memory_fraction
        per_token_bytes = self._estimate_per_token_kv_bytes()
        estimated_bytes_per_block = self.block_size * per_token_bytes

        max_blocks = max(1, int(available / estimated_bytes_per_block))

        # Scale cap based on total VRAM rather than fixed 2048
        total_vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(torch.cuda.device_count())
        )
        # ~500 blocks per 10GB as a reasonable cap
        dynamic_cap = max(2048, int(total_vram_gb * 50))

        result = min(max_blocks, dynamic_cap)
        logger.info(
            "Max KV blocks: %d (%.1f GB available VRAM, %d bytes/block, cap=%d)",
            result, available / (1024 ** 3), estimated_bytes_per_block, dynamic_cap,
        )
        return result

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
        self.max_total_blocks = self._estimate_max_blocks()
        logger.info(
            "KV cache re-estimated with model params: max_blocks=%d",
            self.max_total_blocks,
        )

    @property
    def num_free_blocks(self) -> int:
        return self.max_total_blocks - self._allocated_blocks

    @property
    def utilization(self) -> float:
        if self.max_total_blocks == 0:
            return 0.0
        return self._allocated_blocks / self.max_total_blocks

    def can_allocate(self, num_tokens: int) -> bool:
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return needed_blocks <= self.num_free_blocks

    def allocate_sequence(self, seq_id: str, num_tokens: int, prefix_blocks: Optional[list[KVBlock]] = None) -> bool:
        if seq_id in self._sequences:
            logger.warning("Sequence %s already allocated", seq_id)
            return True

        # If we have prefix blocks, we only need to allocate the rest
        prefix_tokens = sum(b.num_tokens for b in prefix_blocks) if prefix_blocks else 0
        remaining_tokens = max(0, num_tokens - prefix_tokens)
        
        needed_new = (remaining_tokens + self.block_size - 1) // self.block_size
        if needed_new > self.num_free_blocks:
            logger.warning(
                "Cannot allocate %d blocks for seq %s (free=%d)",
                needed_new, seq_id, self.num_free_blocks,
            )
            return False

        blocks = []
        # Reuse prefix blocks first
        if prefix_blocks:
            for b in prefix_blocks:
                b.ref_count += 1
                blocks.append(b)
                if b.block_id not in self._block_pool:
                     self._block_pool[b.block_id] = b

        # Allocate new blocks
        remaining = remaining_tokens
        for _ in range(needed_new):
            tokens_in_block = min(remaining, self.block_size)
            block = KVBlock(
                block_id=self._next_block_id,
                num_tokens=tokens_in_block,
                max_tokens=self.block_size,
                ref_count=1
            )
            self._next_block_id += 1
            blocks.append(block)
            self._block_pool[block.block_id] = block
            remaining -= tokens_in_block

        self._sequences[seq_id] = SequenceBlocks(seq_id=seq_id, blocks=blocks)
        # Note: allocated blocks now counts blocks with ref_count > 0 in a real pool sense
        # For this logic, we keep it simple
        self._update_allocated_count()

        logger.debug(
            "Allocated %d blocks (rewriting %d prefix) for seq %s",
            len(blocks), len(prefix_blocks) if prefix_blocks else 0, seq_id
        )
        return True

    def _update_allocated_count(self):
        self._allocated_blocks = sum(1 for b in self._block_pool.values() if b.ref_count > 0)

    def match_prefix(self, prefix_hashes: list[int]) -> tuple[list[KVBlock], Optional[tuple]]:
        """Find the longest matching prefix sequence of blocks and its tensors."""
        matched_blocks = []
        matched_tensors = None
        
        # Iterating through hashes (which are cumulative) to find the longest match
        for h in prefix_hashes:
            if h in self._prefix_cache_blocks:
                matched_blocks = self._prefix_cache_blocks[h]
                matched_tensors = self._prefix_cache_tensors.get(h)
            else:
                break
        return matched_blocks, matched_tensors

    def promote_to_prefix(self, prefix_hash: int, seq_id: str, num_tokens: int, past_key_values: tuple):
        """Associate a prefix hash with existing blocks and their physical tensors."""
        if seq_id not in self._sequences:
            return
        
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks = self._sequences[seq_id].blocks[:num_blocks]
        
        # Only promote if we don't have it yet
        if prefix_hash not in self._prefix_cache_blocks:
            for b in blocks:
                b.ref_count += 1
            self._prefix_cache_blocks[prefix_hash] = blocks
            
            # We must store a copy of the tensors to avoid them being modified
            # past_key_values is usually a nested tuple of tensors
            # [layer_0_k, layer_0_v], [layer_1_k, layer_1_v] ...
            # Each tensor has shape [batch, heads, seq_len, dim]
            # Since we only cache the prefix, we might want to slice them if they are too long.
            # For now we assume num_tokens exactly matches the tensors provided.
            self._prefix_cache_tensors[prefix_hash] = past_key_values
            
            logger.debug("Promoted %d tokens to physical prefix cache with hash %d", num_tokens, prefix_hash)

    def extend_sequence(self, seq_id: str, additional_tokens: int) -> bool:
        if seq_id not in self._sequences:
            return self.allocate_sequence(seq_id, additional_tokens)

        seq_blocks = self._sequences[seq_id]

        remaining = additional_tokens
        for block in seq_blocks.blocks:
            if remaining <= 0:
                break
            fill = min(remaining, block.free_slots)
            block.num_tokens += fill
            remaining -= fill

        if remaining > 0:
            needed = (remaining + self.block_size - 1) // self.block_size
            if needed > self.num_free_blocks:
                return False

            for _ in range(needed):
                tokens_in_block = min(remaining, self.block_size)
                block = KVBlock(
                    block_id=self._next_block_id,
                    num_tokens=tokens_in_block,
                    max_tokens=self.block_size,
                    ref_count=1
                )
                self._next_block_id += 1
                seq_blocks.blocks.append(block)
                self._block_pool[block.block_id] = block
                remaining -= tokens_in_block

            self._update_allocated_count()

        return True

    def free_sequence(self, seq_id: str):
        if seq_id not in self._sequences:
            return

        seq_blocks = self._sequences.pop(seq_id)
        for b in seq_blocks.blocks:
            b.ref_count -= 1
            if b.ref_count == 0:
                if b.block_id in self._block_pool:
                    del self._block_pool[b.block_id]
        
        self._update_allocated_count()

        logger.debug(
            "Released sequence %s, utilization now %.1f%%",
            seq_id, self.utilization * 100,
        )

    def get_stats(self) -> dict:
        return {
            "total_blocks": self.max_total_blocks,
            "allocated_blocks": self._allocated_blocks,
            "free_blocks": self.num_free_blocks,
            "utilization": round(self.utilization, 3),
            "active_sequences": len(self._sequences),
            "total_cached_tokens": sum(
                s.total_tokens for s in self._sequences.values()
            ),
        }

    def reset(self):
        self._sequences.clear()
        self._block_pool.clear()
        self._prefix_cache_blocks.clear()
        self._prefix_cache_tensors.clear()
        self._next_block_id = 0
        self._allocated_blocks = 0
        logger.info("KV cache reset")
