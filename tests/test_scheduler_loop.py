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
