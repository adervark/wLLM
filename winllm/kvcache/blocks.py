"""KV cache block data structures and reference-counted allocation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KVBlock:
    """A single KV cache block tracking token slots."""
    block_id: int
    num_tokens: int = 0
    max_tokens: int = 16
    ref_count: int = 0  # Track how many sequences/prefixes use this block

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
    _total_tokens: int = 0  # Cached counter -- updated on allocate/extend

    def __post_init__(self):
        # If blocks were provided but _total_tokens wasn't set, compute it
        if self.blocks and self._total_tokens == 0:
            self._total_tokens = sum(b.num_tokens for b in self.blocks)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


class BlockAllocator:
    """Reference-counted block bookkeeping for active sequences.

    Pure accounting: it never touches GPU memory and knows nothing about
    capacity policy — callers pass in the number of free blocks available.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.next_block_id = 0
        self.allocated_blocks = 0
        # seq_id -> SequenceBlocks
        self.sequences: dict[str, SequenceBlocks] = {}
        # block_id -> KVBlock (for global ref management)
        self.block_pool: dict[int, KVBlock] = {}

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _new_block(self, tokens_in_block: int) -> KVBlock:
        block = KVBlock(
            block_id=self.next_block_id,
            num_tokens=tokens_in_block,
            max_tokens=self.block_size,
            ref_count=1,
        )
        self.next_block_id += 1
        self.block_pool[block.block_id] = block
        self.allocated_blocks += 1
        return block

    def pin_block(self, block: KVBlock) -> None:
        """Take a reference on a block (used by prefix reuse/promotion)."""
        if block.ref_count == 0:
            self.allocated_blocks += 1
        block.ref_count += 1
        if block.block_id not in self.block_pool:
            self.block_pool[block.block_id] = block

    def unpin_block(self, block: Optional[KVBlock]) -> None:
        """Release a reference taken by ``pin_block`` (prefix-cache eviction).

        Reclaims the block once its last reference is dropped, mirroring ``free``.
        """
        if block is None or block.ref_count == 0:
            return
        block.ref_count -= 1
        if block.ref_count == 0:
            self.allocated_blocks -= 1
            self.block_pool.pop(block.block_id, None)

    def allocate(
        self,
        seq_id: str,
        num_tokens: int,
        free_blocks: int,
        prefix_blocks: Optional[list[KVBlock]] = None,
    ) -> bool:
        """Allocate blocks for a new sequence, reusing prefix blocks first."""
        if seq_id in self.sequences:
            logger.warning("Sequence %s already allocated", seq_id)
            return True

        # If we have prefix blocks, we only need to allocate the rest
        prefix_tokens = sum(b.num_tokens for b in prefix_blocks) if prefix_blocks else 0
        remaining_tokens = max(0, num_tokens - prefix_tokens)

        needed_new = self.blocks_needed(remaining_tokens)
        if needed_new > free_blocks:
            logger.warning(
                "Cannot allocate %d blocks for seq %s (free=%d)",
                needed_new, seq_id, free_blocks,
            )
            return False

        blocks = []
        # Reuse prefix blocks first
        if prefix_blocks:
            for b in prefix_blocks:
                self.pin_block(b)
                blocks.append(b)

        # Allocate new blocks
        remaining = remaining_tokens
        for _ in range(needed_new):
            tokens_in_block = min(remaining, self.block_size)
            blocks.append(self._new_block(tokens_in_block))
            remaining -= tokens_in_block

        self.sequences[seq_id] = SequenceBlocks(
            seq_id=seq_id, blocks=blocks, _total_tokens=num_tokens
        )

        logger.debug(
            "Allocated %d blocks (reusing %d prefix) for seq %s",
            len(blocks), len(prefix_blocks) if prefix_blocks else 0, seq_id,
        )
        return True

    def extend(self, seq_id: str, additional_tokens: int, free_blocks: int) -> bool:
        """Grow an existing sequence, filling the last block before adding new ones.

        The budget check happens before any mutation, so a rejected extension
        leaves the sequence's accounting exactly as it was.
        """
        seq_blocks = self.sequences[seq_id]

        # Only the last block can have free slots; all previous blocks are full
        last_block = seq_blocks.blocks[-1] if seq_blocks.blocks else None
        fill = min(additional_tokens, last_block.free_slots) if last_block else 0
        remaining = additional_tokens - fill
        if remaining > 0 and self.blocks_needed(remaining) > free_blocks:
            return False

        if last_block is not None:
            last_block.num_tokens += fill

        if remaining > 0:
            needed = self.blocks_needed(remaining)
            for _ in range(needed):
                tokens_in_block = min(remaining, self.block_size)
                seq_blocks.blocks.append(self._new_block(tokens_in_block))
                remaining -= tokens_in_block

        # Update cached total
        seq_blocks._total_tokens += additional_tokens

        return True

    def free(self, seq_id: str) -> None:
        """Release a sequence's references; blocks with no refs are reclaimed."""
        if seq_id not in self.sequences:
            return

        seq_blocks = self.sequences.pop(seq_id)
        for b in seq_blocks.blocks:
            b.ref_count -= 1
            if b.ref_count == 0:
                self.allocated_blocks -= 1
                if b.block_id in self.block_pool:
                    del self.block_pool[b.block_id]

    def total_cached_tokens(self) -> int:
        # Sum over distinct physical blocks so prefix blocks shared by multiple
        # sequences are counted once, not once per referencing sequence.
        return sum(b.num_tokens for b in self.block_pool.values())

    def reset(self) -> None:
        self.sequences.clear()
        self.block_pool.clear()
        self.next_block_id = 0
        self.allocated_blocks = 0
