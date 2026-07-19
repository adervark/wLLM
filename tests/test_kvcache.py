"""Unit tests for the KV cache manager."""

import pytest
import torch
from winllm.kvcache import KVCacheManager, KVBlock, SequenceBlocks, slice_prompt_blocks
from winllm.config import KVCacheConfig


class TestSlicePromptBlocks:
    def test_carves_per_block_with_correct_positions(self):
        # Encode the position index along the seq dim so we can verify slicing.
        positions = torch.arange(48, dtype=torch.float).reshape(1, 1, 48, 1)
        past_key_values = ((positions, positions.clone()),)

        blocks = slice_prompt_blocks(past_key_values, block_size=16, num_blocks=3)

        assert len(blocks) == 3
        for i, (k, v) in enumerate((layer[0] for layer in blocks)):
            assert k.shape[2] == 16
            assert torch.equal(k[0, 0, :, 0], torch.arange(i * 16, i * 16 + 16).float())

    def test_clones_are_independent_of_source(self):
        src = torch.zeros(1, 1, 16, 4)
        blocks = slice_prompt_blocks(((src, src.clone()),), block_size=16, num_blocks=1)
        src[:] = 9.0  # mutating the source must not bleed into the cached slice
        assert torch.count_nonzero(blocks[0][0][0]) == 0


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

    def test_promote_prefix_chain_multi_block(self, manager):
        """A multi-block chain should match cumulatively and concat its KV."""
        manager.allocate_sequence("seq_1", 32)  # 2 blocks of 16

        hashes = [2001, 2002]
        # One (key, value) per layer, each covering a single 16-token block.
        per_block_kv = [
            ((torch.zeros(1, 1, 16, 8), torch.ones(1, 1, 16, 8)),),
            ((torch.full((1, 1, 16, 8), 2.0), torch.full((1, 1, 16, 8), 3.0)),),
        ]
        manager.promote_prefix_chain(hashes, "seq_1", per_block_kv)

        # Matching the full chain yields 2 blocks and KV concatenated to 32 tokens.
        blocks, tensors = manager.match_prefix([2001, 2002])
        assert len(blocks) == 2
        assert tensors[0][0].shape[2] == 32

        # Matching only the first hash yields a single 16-token block.
        blocks1, tensors1 = manager.match_prefix([2001])
        assert len(blocks1) == 1
        assert tensors1[0][0].shape[2] == 16

    def test_promote_prefix_chain_pins_each_block_once(self, manager):
        """Cumulative chain promotion must not double-pin shared lower blocks."""
        manager.allocate_sequence("seq_1", 32)
        per_block_kv = [
            ((torch.zeros(1, 1, 16, 8), torch.zeros(1, 1, 16, 8)),),
            ((torch.zeros(1, 1, 16, 8), torch.zeros(1, 1, 16, 8)),),
        ]
        manager.promote_prefix_chain([3001, 3002], "seq_1", per_block_kv)
        # Each physical block is referenced by the sequence (1) plus exactly one
        # prefix pin (1) -> ref_count == 2, never more.
        for block in manager._sequences["seq_1"].blocks:
            assert block.ref_count == 2

    # ─── Prefix cache eviction (LRU, leaf-only, budget-bounded) ─────

    @staticmethod
    def _kv():
        return ((torch.zeros(1, 1, 16, 4), torch.zeros(1, 1, 16, 4)),)

    def test_prefix_cache_bounded_by_budget(self):
        config = KVCacheConfig(block_size=16, prefix_cache_block_fraction=0.03)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 100  # budget = int(100 * 0.03) = 3
        for n in range(6):
            sid = f"s{n}"
            mgr.allocate_sequence(sid, 16)
            mgr.promote_prefix_chain([9000 + n], sid, [self._kv()])
        assert mgr.prefix_cache.size == 3

    def test_prefix_cache_evicts_unpin_block(self):
        config = KVCacheConfig(block_size=16, prefix_cache_block_fraction=0.02)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 100  # budget = 2
        mgr.allocate_sequence("s0", 16)
        mgr.promote_prefix_chain([500], "s0", [self._kv()])
        pinned_block = mgr._sequences["s0"].blocks[0]
        assert pinned_block.ref_count == 2  # sequence + prefix pin

        # Promote enough distinct prefixes to push [500] out (it's the LRU leaf).
        for n in range(3):
            sid = f"x{n}"
            mgr.allocate_sequence(sid, 16)
            mgr.promote_prefix_chain([600 + n], sid, [self._kv()])

        assert 500 not in mgr.prefix_cache
        assert pinned_block.ref_count == 1  # prefix pin released on eviction

    def test_eviction_is_leaf_only(self):
        config = KVCacheConfig(block_size=16, prefix_cache_block_fraction=0.02)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 100  # budget = 2
        mgr.allocate_sequence("s", 32)  # 2 blocks → chain h100 ← h101
        mgr.promote_prefix_chain([100, 101], "s", [self._kv(), self._kv()])

        # A third (unrelated) entry forces one eviction.
        mgr.allocate_sequence("s2", 16)
        mgr.promote_prefix_chain([200], "s2", [self._kv()])

        # h100 has a child (h101) so it cannot be evicted even though it is the
        # LRU entry; the leaf h101 goes instead.
        assert 100 in mgr.prefix_cache
        assert 101 not in mgr.prefix_cache
        assert 200 in mgr.prefix_cache

    def test_match_refreshes_lru(self):
        config = KVCacheConfig(block_size=16, prefix_cache_block_fraction=0.02)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 100  # budget = 2
        mgr.allocate_sequence("a", 16)
        mgr.promote_prefix_chain([300], "a", [self._kv()])
        mgr.allocate_sequence("b", 16)
        mgr.promote_prefix_chain([301], "b", [self._kv()])

        # Touch 300 so it becomes most-recently-used; 301 is now the LRU leaf.
        mgr.match_prefix([300])

        mgr.allocate_sequence("c", 16)
        mgr.promote_prefix_chain([302], "c", [self._kv()])

        assert 300 in mgr.prefix_cache
        assert 301 not in mgr.prefix_cache
        assert 302 in mgr.prefix_cache

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


# --- extend_sequence ---


class TestExtendSequence:
    @pytest.fixture
    def manager(self):
        config = KVCacheConfig(block_size=16, gpu_memory_fraction=0.4)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 100
        return mgr

    def test_extend_fills_last_block_first(self, manager):
        """Extension should fill the last block's free slots before allocating new blocks."""
        manager.allocate_sequence("s1", 10)  # 1 block with 10/16 tokens
        initial_blocks = manager._allocated_blocks
        manager.extend_sequence("s1", 3)  # 10+3=13, still fits in 1 block
        assert manager._allocated_blocks == initial_blocks  # No new blocks
        seq = manager._sequences["s1"]
        assert seq.total_tokens == 13
        assert seq.num_blocks == 1

    def test_extend_creates_new_block(self, manager):
        """Extension that overflows the last block should create new blocks."""
        manager.allocate_sequence("s1", 15)  # 1 block, 15/16 slots used
        initial_blocks = manager._allocated_blocks
        manager.extend_sequence("s1", 5)  # 15+5=20, needs 2 blocks
        assert manager._allocated_blocks == initial_blocks + 1
        seq = manager._sequences["s1"]
        assert seq.total_tokens == 20

    def test_extend_unknown_sequence_allocates(self, manager):
        """Extending a non-existent sequence should allocate it fresh."""
        result = manager.extend_sequence("new", 10)
        assert result is True
        assert "new" in manager._sequences

    def test_extend_when_out_of_blocks_fails(self, manager):
        """Extension that requires more blocks than available should fail."""
        manager.max_total_blocks = 2
        manager.allocate_sequence("s1", 32)  # Uses 2 blocks
        result = manager.extend_sequence("s1", 32)  # Needs 2 more blocks
        assert result is False

    def test_multiple_extends(self, manager):
        """Multiple sequential extensions should work correctly."""
        manager.allocate_sequence("s1", 5)
        for _ in range(10):
            manager.extend_sequence("s1", 1)
        seq = manager._sequences["s1"]
        assert seq.total_tokens == 15

    def test_failed_extend_leaves_accounting_untouched(self, manager):
        """A rejected extension must not partially mutate the sequence's blocks."""
        manager.max_total_blocks = 2
        manager.allocate_sequence("s1", 20)  # 2 blocks: 16 + 4
        seq = manager._sequences["s1"]
        assert manager.extend_sequence("s1", 40) is False
        assert seq.total_tokens == 20
        assert seq.blocks[-1].num_tokens == 4


# --- Fault regressions: prefix cache vs. memory pressure ---


class TestPrefixCacheReclaim:
    @staticmethod
    def _kv():
        return ((torch.zeros(1, 1, 16, 4), torch.zeros(1, 1, 16, 4)),)

    def _manager(self, max_blocks):
        config = KVCacheConfig(block_size=16, prefix_cache_block_fraction=1.0)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = max_blocks
        return mgr

    def test_extend_reclaims_prefix_cache_blocks(self):
        """When decode needs blocks that only the prefix cache holds, the
        cache must be evicted rather than failing the extension."""
        mgr = self._manager(3)
        mgr.allocate_sequence("donor", 16)
        mgr.promote_prefix_chain([700], "donor", [self._kv()])
        mgr.free_sequence("donor")  # block survives, pinned by the prefix cache
        assert mgr.num_free_blocks == 2

        mgr.allocate_sequence("s1", 32)
        assert mgr.num_free_blocks == 0
        assert mgr.extend_sequence("s1", 16) is True
        assert 700 not in mgr.prefix_cache

    def test_allocate_still_fails_when_nothing_reclaimable(self):
        mgr = self._manager(1)
        mgr.allocate_sequence("s1", 16)
        assert mgr.extend_sequence("s1", 16) is False


class TestCanonicalChainPromotion:
    def test_chain_promotion_reuses_canonical_cached_blocks(self):
        """Extending a chain whose lower hashes were promoted by a *different*
        sequence must reference the already-pinned canonical blocks, never the
        promoting sequence's own (soon-to-be-freed) duplicates."""
        config = KVCacheConfig(block_size=16, prefix_cache_block_fraction=1.0)
        manager = KVCacheManager(config)
        manager.max_total_blocks = 100
        kv = ((torch.zeros(1, 1, 16, 4), torch.zeros(1, 1, 16, 4)),)

        # A promotes the shared first block.
        manager.allocate_sequence("A", 16)
        manager.promote_prefix_chain([4001], "A", [kv])
        canonical = manager._prefix_cache_blocks[4001][0]

        # B ran concurrently with the same first block content (different
        # physical blocks) and promotes a longer chain after A.
        manager.allocate_sequence("B", 32)
        manager.promote_prefix_chain([4001, 4002], "B", [kv, kv])

        manager.free_sequence("A")
        manager.free_sequence("B")

        chain = manager._prefix_cache_blocks[4002]
        assert chain[0] is canonical
        # Every block the cache references must still be live (pinned).
        assert all(b.ref_count >= 1 for b in chain)


# --- Prefix cache operations ---


class TestPrefixCache:
    @pytest.fixture
    def manager(self):
        config = KVCacheConfig(block_size=16, gpu_memory_fraction=0.4)
        mgr = KVCacheManager(config)
        mgr.max_total_blocks = 100
        return mgr

    def test_match_prefix_no_match(self, manager):
        blocks, tensors = manager.match_prefix([12345])
        assert blocks == []
        assert tensors is None

    def test_promote_and_match(self, manager):
        """Promoted prefix should be findable via match_prefix."""
        manager.allocate_sequence("s1", 16)  # 1 full block
        fake_tensors = (("key_tensor", "val_tensor"),)
        manager.promote_to_prefix(42, "s1", 16, fake_tensors)
        blocks, tensors = manager.match_prefix([42])
        assert len(blocks) > 0
        assert tensors == fake_tensors

    def test_promote_idempotent(self, manager):
        """Promoting the same hash twice should not duplicate blocks."""
        manager.allocate_sequence("s1", 16)
        fake_tensors = (("k", "v"),)
        manager.promote_to_prefix(42, "s1", 16, fake_tensors)
        blocks_before = len(manager._prefix_cache_blocks)
        manager.promote_to_prefix(42, "s1", 16, fake_tensors)
        assert len(manager._prefix_cache_blocks) == blocks_before

    def test_reset_clears_prefix_cache(self, manager):
        """Reset should clear all prefix cache data."""
        manager.allocate_sequence("s1", 16)
        manager.promote_to_prefix(42, "s1", 16, (("k", "v"),))
        manager.reset()
        assert len(manager._prefix_cache_blocks) == 0
        assert len(manager._prefix_cache_tensors) == 0
        assert manager._allocated_blocks == 0
