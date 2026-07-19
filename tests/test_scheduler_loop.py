"""Integration tests for the Scheduler's continuous-batching loop.

Uses a scripted fake engine so the loop's finish detection (EOS, stop
strings, max_tokens, failures) can be tested deterministically without
loading a real model.
"""

import asyncio

import pytest

from winllm.config import KVCacheConfig, SamplingParams, SchedulerConfig
from winllm.core.types import GenerationRequest, RequestStatus
from winllm.kvcache import KVCacheManager
from winllm.scheduling.scheduler import Scheduler

# Token id -> decoded text used by the fake tokenizer
TOKEN_TEXT = {1: "a", 2: "b", 3: "X", 4: "c", 99: ""}
EOS_ID = 99


class FakeTokenizer:
    eos_token_id = EOS_ID

    def encode(self, text, *args, **kwargs):
        return [1, 2, 3]

    def decode(self, ids, *args, **kwargs):
        return "".join(TOKEN_TEXT.get(i, "?") for i in ids)


class ScriptedEngine:
    """Engine double whose generate_step appends pre-scripted tokens."""

    def __init__(self, script, fail_with=None):
        self.tokenizer = FakeTokenizer()
        self.kv_cache_manager = KVCacheManager(KVCacheConfig())
        self.script = list(script)  # tokens appended one per step to every request
        self.fail_with = fail_with
        self.is_ready = True

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode_tokens(self, ids):
        return self.tokenizer.decode(ids)

    def generate_step(self, requests, **kwargs):
        if self.fail_with is not None:
            raise self.fail_with
        for req in requests:
            req._prefill_cursor = len(req.prompt_token_ids)
            step = len(req.output_token_ids)
            token = self.script[step] if step < len(self.script) else self.script[-1]
            req.output_token_ids.append(token)
            if req._token_callback:
                req._token_callback(token, False)
        return requests


def wait_until(cond, timeout=5.0):
    """Poll until cond() is truthy; the loop thread cleans up asynchronously."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def run_request(engine, params, prompt="hi", token_callback=None):
    """Submit one request through a fresh scheduler and wait for completion."""
    scheduler = Scheduler(engine, SchedulerConfig())

    async def _go():
        req = GenerationRequest(prompt=prompt, sampling_params=params)
        if token_callback:
            req._token_callback = token_callback
        return await asyncio.wait_for(scheduler.submit(req), timeout=10), scheduler

    return asyncio.run(_go())


class TestFinishDetection:
    def test_eos_finishes_with_stop_reason(self):
        engine = ScriptedEngine(script=[1, 2, EOS_ID])
        req, _ = run_request(engine, SamplingParams(max_tokens=50))
        assert req.status == RequestStatus.COMPLETED
        assert req.finish_reason == "stop"
        assert req.output_token_ids[-1] == EOS_ID

    def test_max_tokens_finishes_with_length_reason(self):
        engine = ScriptedEngine(script=[1])  # never emits EOS
        req, _ = run_request(engine, SamplingParams(max_tokens=4))
        assert req.status == RequestStatus.COMPLETED
        assert req.finish_reason == "length"
        assert len(req.output_token_ids) == 4

    def test_stop_string_finishes_and_trims_output(self):
        # Script decodes to "a", "b", "X", ... — stop on "X"
        engine = ScriptedEngine(script=[1, 2, 3, 1, 1])
        req, _ = run_request(engine, SamplingParams(max_tokens=50, stop=["X"]))
        assert req.status == RequestStatus.COMPLETED
        assert req.finish_reason == "stop"
        assert req.output_text == "ab"
        # Stopped at the stop string, not at max_tokens
        assert len(req.output_token_ids) == 3

    def test_stop_string_spanning_tokens(self):
        # "ab" spans tokens 1 and 2 — must match across token boundaries
        engine = ScriptedEngine(script=[4, 1, 2, 4, 4])
        req, _ = run_request(engine, SamplingParams(max_tokens=50, stop=["ab"]))
        assert req.finish_reason == "stop"
        assert req.output_text == "c"


class TestFailureCleanup:
    def test_failed_step_fails_request_and_clears_batch(self):
        engine = ScriptedEngine(script=[1], fail_with=RuntimeError("boom"))
        req, scheduler = run_request(engine, SamplingParams(max_tokens=10))
        assert req.status == RequestStatus.FAILED
        assert req.finish_reason == "error"
        assert "boom" in req.error
        # The failed request must not linger in the active batch
        assert wait_until(lambda: scheduler.num_running == 0)

    def test_failure_frees_kv_allocation(self):
        engine = ScriptedEngine(script=[1], fail_with=RuntimeError("boom"))
        _, scheduler = run_request(engine, SamplingParams(max_tokens=10))
        assert wait_until(lambda: len(engine.kv_cache_manager.allocator.sequences) == 0)


class MultiTokenEngine(ScriptedEngine):
    """Engine double that commits several tokens in a single step
    (like speculative decoding does)."""

    def generate_step(self, requests, **kwargs):
        for req in requests:
            req._prefill_cursor = len(req.prompt_token_ids)
            for token in self.script:
                req.output_token_ids.append(token)
        return requests


class TestStopStringPriority:
    def test_trims_at_earliest_occurrence_not_list_order(self):
        # One step commits "ab". With stop=["b", "a"], "a" occurs first in the
        # text and must win, even though "b" is listed first.
        engine = MultiTokenEngine(script=[1, 2])
        req, _ = run_request(engine, SamplingParams(max_tokens=50, stop=["b", "a"]))
        assert req.finish_reason == "stop"
        assert req.output_text == ""


class TestAdmissionReclaim:
    def test_admission_evicts_prefix_cache_instead_of_failing(self):
        """A request that fits only after prefix-cache eviction must be
        admitted, not permanently rejected as too large."""
        import torch

        engine = ScriptedEngine(script=[1, EOS_ID])
        mgr = engine.kv_cache_manager
        mgr.max_total_blocks = 4
        mgr.config.prefix_cache_block_fraction = 1.0

        kv = ((torch.zeros(1, 1, 16, 4), torch.zeros(1, 1, 16, 4)),)
        mgr.allocate_sequence("donor", 48)
        mgr.promote_prefix_chain([9101, 9102, 9103], "donor", [kv, kv, kv])
        mgr.free_sequence("donor")
        assert mgr.num_free_blocks == 1

        # Needs 3 blocks (3 prompt tokens + 30 max_tokens), only 1 free.
        req, _ = run_request(engine, SamplingParams(max_tokens=30))
        assert req.status == RequestStatus.COMPLETED


class TestCancelledInQueue:
    def test_cancelled_waiting_request_skips_kv_allocation(self):
        engine = ScriptedEngine(script=[1, EOS_ID])
        calls = []
        orig = engine.kv_cache_manager.allocate_sequence

        def tracking_allocate(*args, **kwargs):
            calls.append(args)
            return orig(*args, **kwargs)

        engine.kv_cache_manager.allocate_sequence = tracking_allocate
        scheduler = Scheduler(engine, SchedulerConfig())

        async def _go():
            req = GenerationRequest(prompt="hi", sampling_params=SamplingParams(max_tokens=5))
            req.cancel()
            return await asyncio.wait_for(scheduler.submit(req), timeout=10)

        req = asyncio.run(_go())
        assert req.status == RequestStatus.CANCELLED
        assert not calls, "cancelled request should never allocate KV blocks"


class TestQueueFullAccounting:
    def test_queue_full_rejection_counted_in_stats(self):
        engine = ScriptedEngine(script=[EOS_ID])
        scheduler = Scheduler(engine, SchedulerConfig(max_waiting_requests=0))

        async def _go():
            return await scheduler.submit(GenerationRequest(prompt="hi"))

        req = asyncio.run(_go())
        assert req.status == RequestStatus.FAILED
        assert scheduler.stats.failed_requests == 1
        assert scheduler.stats.total_requests == 1


class TestStreamSignaling:
    def test_token_callback_receives_finished_signal(self):
        engine = ScriptedEngine(script=[1, 2, EOS_ID])
        received = []
        req, _ = run_request(
            engine,
            SamplingParams(max_tokens=50),
            token_callback=lambda tid, fin: received.append((tid, fin)),
        )
        assert req.status == RequestStatus.COMPLETED
        assert received, "no tokens streamed"
        assert received[-1][1] is True, "stream never signaled finished"
        # Every earlier emission is a live token
        assert all(fin is False for _, fin in received[:-1])
