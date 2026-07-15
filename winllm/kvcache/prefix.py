"""Prompt-prefix hashing and cached-prefix storage."""

from __future__ import annotations

import hashlib
import logging
import struct
from collections import OrderedDict
from typing import Optional

import torch

from .blocks import KVBlock

logger = logging.getLogger(__name__)


def compute_prefix_hashes(tokens: list[int], block_size: int) -> list[int]:
    """Generate cumulative hashes for every complete block prefix.

    Uses incremental SHA-256 so each block adds O(block_size) work
    instead of the previous O(n^2) approach of hashing ever-growing tuples.
    """
    hashes = []
    h = hashlib.sha256()
    num_blocks = len(tokens) // block_size
    for i in range(num_blocks):
        start = i * block_size
        # Use struct.pack for proper serialization of arbitrary-sized token IDs.
        # bytes() only works for values 0-255, which is insufficient for real vocabularies.
        block_bytes = struct.pack(f'<{block_size}i', *tokens[start:start + block_size])
        h.update(block_bytes)
        hashes.append(int.from_bytes(h.digest()[:8], 'little'))
    return hashes


def slice_prompt_blocks(
    past_key_values: tuple, block_size: int, num_blocks: int
) -> list[tuple]:
    """Carve a prompt's KV cache into ``num_blocks`` per-block past_key_values.

    Centralizes the KV tensor layout knowledge (``[batch, heads, seq, head_dim]``)
    so callers like the scheduler can promote prefix blocks without reaching into
    tensor internals. Each returned entry covers exactly one ``block_size`` slice
    and is cloned so it stays valid after the source cache is freed.
    """
    per_block: list[tuple] = []
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        per_block.append(tuple(
            (k[:, :, start:end, :].clone(), v[:, :, start:end, :].clone())
            for (k, v) in past_key_values
        ))
    return per_block


class PrefixCache:
    """Stores promoted prompt prefixes: their blocks and physical KV tensors.

    Entries form a forest of cumulative-hash chains (``h_i``'s parent is the
    next-shorter prefix ``h_{i-1}``). Eviction is **leaf-only LRU**: only a
    hash with no longer prefix depending on it can be removed, and the least
    recently matched such leaf goes first. This keeps the cache bounded without
    ever orphaning a longer chain or evicting a block another chain still needs.
    """

    def __init__(self):
        # prefix_hash -> list[KVBlock]  (cumulative blocks for this prefix)
        self.blocks_by_hash: dict[int, list[KVBlock]] = {}
        # prefix_hash -> tuple[tuple[torch.Tensor]]  (this block's KV only)
        self.tensors_by_hash: dict[int, tuple] = {}
        # prefix_hash -> parent (next-shorter) prefix hash, or None for a root
        self._parent: dict[int, Optional[int]] = {}
        # prefix_hash -> number of present children (longer prefixes)
        self._children: dict[int, int] = {}
        # access order; front = least recently used
        self._lru: "OrderedDict[int, None]" = OrderedDict()

    def __contains__(self, prefix_hash: int) -> bool:
        return prefix_hash in self.blocks_by_hash

    @property
    def size(self) -> int:
        return len(self.blocks_by_hash)

    def match(self, prefix_hashes: list[int]) -> tuple[list[KVBlock], Optional[tuple]]:
        """Find the longest matching prefix sequence of blocks and its tensors.

        Hashes are cumulative, so the chain of consecutive present hashes from
        the front is the longest matching prefix. ``blocks_by_hash`` stores the
        cumulative block list for the longest hash, while ``tensors_by_hash``
        stores each block's KV individually — so the matched tensors are the
        per-block KV of the chain concatenated along the sequence dimension.
        Matched hashes are marked most-recently-used.
        """
        matched_chain: list[int] = []
        for h in prefix_hashes:
            if h in self.blocks_by_hash:
                matched_chain.append(h)
            else:
                break

        if not matched_chain:
            return [], None

        # Refresh LRU: a matched prefix is "recently used".
        for h in matched_chain:
            self._lru.move_to_end(h)

        matched_blocks = self.blocks_by_hash[matched_chain[-1]]

        tensor_chain = [self.tensors_by_hash.get(h) for h in matched_chain]
        if any(t is None for t in tensor_chain):
            return matched_blocks, None
        if len(tensor_chain) == 1:
            return matched_blocks, tensor_chain[0]
        return matched_blocks, self._concat_kv(tensor_chain)

    @staticmethod
    def _concat_kv(tensor_chain: list[tuple]) -> tuple:
        """Concatenate a chain of per-block past_key_values along the seq dim."""
        num_layers = len(tensor_chain[0])
        merged = []
        for layer in range(num_layers):
            keys = [t[layer][0] for t in tensor_chain]
            values = [t[layer][1] for t in tensor_chain]
            merged.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
        return tuple(merged)

    def store(
        self,
        prefix_hash: int,
        blocks: list[KVBlock],
        past_key_values: tuple,
        parent_hash: Optional[int] = None,
    ) -> None:
        """Associate a prefix hash with blocks and their physical tensors.

        The caller is responsible for pinning the blocks (ref-counting) —
        this class only stores and tracks the chain topology for eviction.
        """
        self.blocks_by_hash[prefix_hash] = blocks
        self.tensors_by_hash[prefix_hash] = past_key_values
        self._parent[prefix_hash] = parent_hash
        self._children.setdefault(prefix_hash, 0)
        if parent_hash is not None and parent_hash in self._children:
            self._children[parent_hash] += 1
        self._lru[prefix_hash] = None
        self._lru.move_to_end(prefix_hash)

    def pop_lru_leaf(self) -> Optional[KVBlock]:
        """Evict the least-recently-used leaf; return the block it introduced.

        A leaf is a prefix with no present children. Returns the (unique) block
        that leaf added so the caller can unpin it, or ``None`` if every entry
        still has a dependent child (nothing safely evictable).
        """
        for h in self._lru:  # iterates front (LRU) first
            if self._children.get(h, 0) == 0:
                return self._remove(h)
        return None

    def _remove(self, prefix_hash: int) -> Optional[KVBlock]:
        blocks = self.blocks_by_hash.pop(prefix_hash)
        introduced_block = blocks[-1] if blocks else None
        self.tensors_by_hash.pop(prefix_hash, None)
        parent = self._parent.pop(prefix_hash, None)
        self._children.pop(prefix_hash, None)
        self._lru.pop(prefix_hash, None)
        if parent is not None and parent in self._children:
            self._children[parent] -= 1
        return introduced_block

    def reset(self) -> None:
        self.blocks_by_hash.clear()
        self.tensors_by_hash.clear()
        self._parent.clear()
        self._children.clear()
        self._lru.clear()
