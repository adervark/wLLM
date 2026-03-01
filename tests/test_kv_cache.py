"""Unit tests for the KV cache manager."""

import pytest
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
