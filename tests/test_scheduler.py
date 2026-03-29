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
        assert hashes[0] == hash(tuple(tokens[:16]))

    def test_two_blocks(self):
        tokens = list(range(32))
        hashes = _get_prefix_hashes(tokens, block_size=16)
        assert len(hashes) == 2
        assert hashes[0] == hash(tuple(tokens[:16]))
        assert hashes[1] == hash(tuple(tokens[:32]))

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
        assert hashes[0] == hash(tuple([10]))
        assert hashes[1] == hash(tuple([10, 20]))
        assert hashes[2] == hash(tuple([10, 20, 30]))


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
