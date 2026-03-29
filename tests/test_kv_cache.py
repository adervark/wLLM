"""Unit tests for the KV cache manager."""

import pytest
import torch
from winllm.kv_cache import KVCacheManager, KVBlock, SequenceBlocks
from winllm.config import KVCacheConfig


class TestKVBlock:
    def test_creation(self):
        block = KVBlock(block_id=0, num_tokens=5, max_tokens=16)
        assert block.free_slots == 11
        assert not block.is_full

    def test_full_block(self):
        block = KVBlock(block_id=0, num_tokens=16, max_tokens=16)
        assert block.is_full
        assert block.free_slots == 0

    def test_empty_block(self):
        block = KVBlock(block_id=0, num_tokens=0, max_tokens=16)
        assert block.free_slots == 16
        assert not block.is_full

    def test_ref_count_default(self):
        block = KVBlock(block_id=0)
        assert block.ref_count == 0


class TestSequenceBlocks:
    def test_empty(self):
        sb = SequenceBlocks(seq_id="test")
        assert sb.total_tokens == 0
        assert sb.num_blocks == 0

    def test_with_blocks(self):
        blocks = [
            KVBlock(block_id=0, num_tokens=16, max_tokens=16),
            KVBlock(block_id=1, num_tokens=8, max_tokens=16),
        ]
        sb = SequenceBlocks(seq_id="test", blocks=blocks)
        assert sb.total_tokens == 24
        assert sb.num_blocks == 2


class TestKVCacheManager:
    @pytest.fixture
    def manager(self):
        config = KVCacheConfig(block_size=16, gpu_memory_fraction=0.4)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 100  # Override for testing
        return mgr

    def test_initial_state(self, manager):
        assert manager.num_free_blocks == 100
        assert manager.utilization == 0.0

    def test_allocate_sequence(self, manager):
        assert manager.allocate_sequence("seq_1", 32)  # 2 blocks
        assert manager.num_free_blocks == 98
        assert manager.utilization == pytest.approx(0.02)

    def test_allocate_over_budget(self, manager):
        # Try to allocate more blocks than available
        huge = manager.max_total_blocks * manager.block_size + 1
        assert not manager.allocate_sequence("big_seq", huge)

    def test_free_sequence(self, manager):
        manager.allocate_sequence("seq_1", 48)  # 3 blocks
        manager.free_sequence("seq_1")
        assert manager.num_free_blocks == 100
        assert manager.utilization == 0.0

    def test_extend_sequence(self, manager):
        manager.allocate_sequence("seq_1", 16)  # 1 block, filled
        assert manager.num_free_blocks == 99
        # Extend by 8 tokens — should allocate a new block
        assert manager.extend_sequence("seq_1", 8)
        assert manager.num_free_blocks == 98

    def test_can_allocate(self, manager):
        assert manager.can_allocate(1600)  # 100 blocks
        assert not manager.can_allocate(1601)  # 101 blocks needed

    def test_multiple_sequences(self, manager):
        manager.allocate_sequence("a", 16)
        manager.allocate_sequence("b", 32)
        manager.allocate_sequence("c", 48)
        stats = manager.get_stats()
        assert stats["active_sequences"] == 3
        assert stats["allocated_blocks"] == 6  # 1+2+3
        assert stats["total_cached_tokens"] == 96

    def test_reset(self, manager):
        manager.allocate_sequence("a", 64)
        manager.allocate_sequence("b", 64)
        manager.reset()
        assert manager.num_free_blocks == 100
        assert manager.utilization == 0.0

    # ─── Prefix caching tests ──────────────────────────────────────

    def test_match_prefix_no_cache(self, manager):
        """No prefix cache should return empty matches."""
        blocks, tensors = manager.match_prefix([hash((1, 2))])
        assert blocks == []
        assert tensors is None

    def test_promote_and_match_prefix(self, manager):
        """After promoting, the same hash should match."""
        manager.allocate_sequence("seq_1", 16)

        prefix_hash = hash(tuple(range(16)))
        mock_kv = ((torch.zeros(1, 1, 16, 64), torch.zeros(1, 1, 16, 64)),)
        manager.promote_to_prefix(prefix_hash, "seq_1", 16, mock_kv)

        blocks, tensors = manager.match_prefix([prefix_hash])
        assert len(blocks) > 0
        assert tensors is not None

    def test_promote_idempotent(self, manager):
        """Promoting the same hash twice should not double-count ref counts."""
        manager.allocate_sequence("seq_1", 16)
        prefix_hash = hash(tuple(range(16)))
        mock_kv = ((torch.zeros(1),),)

        manager.promote_to_prefix(prefix_hash, "seq_1", 16, mock_kv)
        initial_ref = manager._prefix_cache_blocks[prefix_hash][0].ref_count

        manager.promote_to_prefix(prefix_hash, "seq_1", 16, mock_kv)
        assert manager._prefix_cache_blocks[prefix_hash][0].ref_count == initial_ref

    def test_match_prefix_longest_match(self, manager):
        """Should return the longest chain of matching prefix hashes."""
        manager.allocate_sequence("seq_1", 32)

        hash1 = 1001
        hash2 = 1002
        mock_kv = ((torch.zeros(1),),)

        manager.promote_to_prefix(hash1, "seq_1", 16, mock_kv)
        # hash2 is not promoted

        blocks, _ = manager.match_prefix([hash1, hash2])
        # Should match hash1 but stop at hash2
        assert len(blocks) > 0

    # ─── Reset completeness ────────────────────────────────────────

    def test_reset_clears_block_pool(self, manager):
        manager.allocate_sequence("a", 32)
        manager.reset()
        assert len(manager._block_pool) == 0

    def test_reset_clears_prefix_caches(self, manager):
        manager.allocate_sequence("a", 16)
        prefix_hash = hash(tuple(range(16)))
        mock_kv = ((torch.zeros(1),),)
        manager.promote_to_prefix(prefix_hash, "a", 16, mock_kv)
        manager.reset()
        assert len(manager._prefix_cache_blocks) == 0
        assert len(manager._prefix_cache_tensors) == 0

    def test_reset_resets_block_id_counter(self, manager):
        manager.allocate_sequence("a", 32)
        manager.reset()
        assert manager._next_block_id == 0

    # ─── Edge cases ────────────────────────────────────────────────

    def test_extend_nonexistent_sequence(self, manager):
        """Extending a non-existent sequence should auto-allocate."""
        result = manager.extend_sequence("new_seq", 8)
        assert result is True
        assert "new_seq" in manager._sequences

    def test_double_allocate_same_seq(self, manager):
        """Allocating the same sequence twice should return True (idempotent)."""
        assert manager.allocate_sequence("dup", 16) is True
        assert manager.allocate_sequence("dup", 16) is True

    def test_free_nonexistent_sequence(self, manager):
        """Freeing a non-existent sequence should not crash."""
        manager.free_sequence("nonexistent")  # Should not raise

    def test_allocate_with_prefix_blocks(self, manager):
        """Allocating with prefix blocks should reuse existing blocks."""
        # First create some blocks via a regular allocation
        manager.allocate_sequence("prefix_source", 16)
        prefix_blocks = manager._sequences["prefix_source"].blocks[:]

        # Allocate new sequence reusing prefix blocks
        manager.allocate_sequence("new_seq", 32, prefix_blocks=prefix_blocks)
        assert "new_seq" in manager._sequences

    def test_get_stats_format(self, manager):
        stats = manager.get_stats()
        assert "total_blocks" in stats
        assert "allocated_blocks" in stats
        assert "free_blocks" in stats
        assert "utilization" in stats
        assert "active_sequences" in stats
        assert "total_cached_tokens" in stats

    def test_utilization_at_zero_max_blocks(self):
        """Edge case: max_blocks is 0 should not divide-by-zero."""
        config = KVCacheConfig(block_size=16)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 0
        assert mgr.utilization == 0.0

