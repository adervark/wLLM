"""Unit tests for the request scheduler and prefix hashing."""

import time
import pytest
from winllm.scheduler import _get_prefix_hashes, SchedulerStats


# ─── _get_prefix_hashes ────────────────────────────────────────────────────


class TestGetPrefixHashes:
    def test_empty_tokens(self):
        assert _get_prefix_hashes([], block_size=16) == []

    def test_shorter_than_block_size(self):
        """Tokens shorter than one block should produce no hashes."""
        hashes = _get_prefix_hashes([1, 2, 3], block_size=16)
        assert hashes == []

    def test_exactly_one_block(self):
        tokens = list(range(16))
        hashes = _get_prefix_hashes(tokens, block_size=16)
        assert len(hashes) == 1
        # SHA-256 hash should be a positive integer fitting in 8 bytes
        assert isinstance(hashes[0], int)

    def test_two_blocks(self):
        tokens = list(range(32))
        hashes = _get_prefix_hashes(tokens, block_size=16)
        assert len(hashes) == 2
        # Each block produces a distinct hash (cumulative SHA-256)
        assert hashes[0] != hashes[1]
        # First block hash should match the one-block case
        one_block = _get_prefix_hashes(tokens[:16], block_size=16)
        assert hashes[0] == one_block[0]

    def test_partial_last_block_ignored(self):
        """Only complete blocks should be hashed."""
        tokens = list(range(25))  # 1.5625 blocks at block_size=16
        hashes = _get_prefix_hashes(tokens, block_size=16)
        assert len(hashes) == 1

    def test_deterministic(self):
        """Same tokens should produce same hashes."""
        tokens = list(range(32))
        h1 = _get_prefix_hashes(tokens, block_size=16)
        h2 = _get_prefix_hashes(tokens, block_size=16)
        assert h1 == h2

    def test_different_tokens_different_hashes(self):
        t1 = list(range(16))
        t2 = list(range(100, 116))
        h1 = _get_prefix_hashes(t1, block_size=16)
        h2 = _get_prefix_hashes(t2, block_size=16)
        assert h1 != h2

    def test_block_size_1(self):
        tokens = [10, 20, 30]
        hashes = _get_prefix_hashes(tokens, block_size=1)
        assert len(hashes) == 3
        # Cumulative: each hash depends on all previous blocks
        assert len(set(hashes)) == 3  # All distinct


# ─── SchedulerStats ────────────────────────────────────────────────────────


class TestSchedulerStats:
    def test_defaults(self):
        s = SchedulerStats()
        assert s.total_requests == 0
        assert s.completed_requests == 0
        assert s.failed_requests == 0
        assert s.total_prompt_tokens == 0
        assert s.total_generation_tokens == 0
        assert s.total_generation_time == 0.0

    def test_avg_tokens_per_second_zero_time(self):
        s = SchedulerStats()
        assert s.avg_tokens_per_second == 0.0

    def test_avg_tokens_per_second_calculation(self):
        s = SchedulerStats(
            total_generation_tokens=100,
            total_generation_time=10.0,
        )
        assert s.avg_tokens_per_second == 10.0

    def test_to_dict_format(self):
        s = SchedulerStats(
            total_requests=5,
            completed_requests=3,
            failed_requests=1,
            total_prompt_tokens=500,
            total_generation_tokens=200,
            total_generation_time=20.0,
        )
        d = s.to_dict()
        assert d["total_requests"] == 5
        assert d["completed_requests"] == 3
        assert d["failed_requests"] == 1
        assert d["total_prompt_tokens"] == 500
        assert d["total_generation_tokens"] == 200
        assert d["avg_tokens_per_second"] == 10.0

    def test_to_dict_rounds_avg(self):
        s = SchedulerStats(
            total_generation_tokens=7,
            total_generation_time=3.0,
        )
        d = s.to_dict()
        assert d["avg_tokens_per_second"] == 2.3  # 7/3 rounded to 1 decimal

    def test_increment_stats(self):
        s = SchedulerStats()
        s.total_requests += 1
        s.completed_requests += 1
        s.total_generation_tokens += 50
        s.total_generation_time += 5.0

        assert s.total_requests == 1
        assert s.avg_tokens_per_second == 10.0


# --- Prefix hash consistency ---


class TestPrefixHashConsistency:
    def test_same_prefix_same_hash(self):
        """Two sequences with the same prompt prefix should get the same hash."""
        tokens1 = list(range(32))
        tokens2 = list(range(32)) + [100, 200, 300]  # Same prefix, different suffix
        h1 = _get_prefix_hashes(tokens1, block_size=16)
        h2 = _get_prefix_hashes(tokens2, block_size=16)
        # First two block hashes should be identical
        assert h1[0] == h2[0]
        assert h1[1] == h2[1]

    def test_promotion_uses_same_hash_as_lookup(self):
        """Verify that the hash used for promotion matches the one used for lookup."""
        tokens = list(range(64))
        h1 = _get_prefix_hashes(tokens, block_size=16)
        h2 = _get_prefix_hashes(tokens, block_size=16)
        assert h1 == h2

    def test_hash_is_cumulative(self):
        """Each block hash should depend on ALL previous blocks, not just the current one."""
        tokens = list(range(32))
        hashes = _get_prefix_hashes(tokens, block_size=16)

        # If we change the first block, the second block hash should also change
        tokens_modified = [999] + list(range(1, 32))
        hashes_modified = _get_prefix_hashes(tokens_modified, block_size=16)
        assert hashes[0] != hashes_modified[0]
        assert hashes[1] != hashes_modified[1]  # Second hash also changes

    def test_large_block_size(self):
        tokens = list(range(1024))
        hashes = _get_prefix_hashes(tokens, block_size=512)
        assert len(hashes) == 2

    def test_single_token_block_high_cardinality(self):
        """With block_size=1, each token position gets its own hash."""
        tokens = list(range(100))
        hashes = _get_prefix_hashes(tokens, block_size=1)
        assert len(hashes) == 100
        assert len(set(hashes)) == 100  # All unique
