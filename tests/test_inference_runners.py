"""Regression tests for PrefillRunner / DecodeRunner fault handling.

Uses a fake model (zero logits, KV grown by input length) so runner-level
behavior — cancellation, timing stamps, KV-exhaustion handling — can be
tested without loading a real model.
"""

import time
from types import SimpleNamespace

import torch

from winllm.config import KVCacheConfig, SamplingParams
from winllm.core.types import GenerationRequest, RequestStatus
from winllm.inference.buffers import DecodeInputBuffer, PersistentBatchCache
from winllm.inference.decode import DecodeRunner
from winllm.inference.prefill import PrefillRunner
from winllm.inference.runtime import ModelRuntime
from winllm.inference.streaming import StreamEmitter
from winllm.kvcache import KVCacheManager

DEVICE = torch.device("cpu")


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [1, 2, 3, 4]

    def decode(self, ids, **kwargs):
        return "x" * len(ids)


class FakeModel:
    """Returns zero logits and a KV cache grown by the input length."""

    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        self.calls += 1
        bsz, seq = input_ids.shape
        prev = past_key_values[0][0].shape[2] if past_key_values else 0
        k = torch.zeros(bsz, 1, prev + seq, 4)
        return SimpleNamespace(
            logits=torch.zeros(bsz, seq, self.vocab_size),
            past_key_values=((k, k.clone()),),
        )


def make_runtime(max_blocks=100):
    rt = ModelRuntime()
    rt.model = FakeModel()
    rt.tokenizer = FakeTokenizer()
    rt.device = DEVICE
    rt.kv_cache_manager = KVCacheManager(KVCacheConfig(block_size=16))
    rt.kv_cache_manager.max_total_blocks = max_blocks
    return rt


def make_decode_req(mgr, prompt_len, alloc_tokens):
    req = GenerationRequest(
        prompt="p",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=50, temperature=0),
    )
    mgr.allocate_sequence(req.request_id, alloc_tokens)
    k = torch.zeros(1, 1, prompt_len, 4)
    req._past_key_values = ((k, k.clone()),)
    req.output_token_ids = [5]
    req._prefill_cursor = prompt_len
    req.status = RequestStatus.RUNNING
    return req


class TestChunkedPrefillTiming:
    def test_chunked_prefill_preserves_started_at(self):
        rt = make_runtime()
        runner = PrefillRunner(rt, StreamEmitter(rt))
        req = GenerationRequest(
            prompt="p",
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(max_tokens=5, temperature=0),
        )

        runner.prefill_single(req, DEVICE, chunked_prefill=True, max_tokens=2)
        assert not req.is_prefill_complete
        first = req.started_at
        time.sleep(0.02)
        runner.prefill_single(req, DEVICE, chunked_prefill=True, max_tokens=2)
        assert req.started_at == first


class TestPrefillCancellation:
    def test_prefill_skips_cancelled_request(self):
        rt = make_runtime()
        runner = PrefillRunner(rt, StreamEmitter(rt))
        req = GenerationRequest(
            prompt="p",
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(temperature=0),
        )
        req.cancel()
        runner.prefill_single(req, DEVICE, chunked_prefill=False, max_tokens=512)
        assert rt.model.calls == 0


class TestDecodeKVExhaustion:
    def test_decode_single_fails_request_when_kv_exhausted(self):
        rt = make_runtime(max_blocks=1)
        req = make_decode_req(rt.kv_cache_manager, prompt_len=15, alloc_tokens=16)
        runner = DecodeRunner(rt, StreamEmitter(rt), DecodeInputBuffer(), PersistentBatchCache())

        runner.decode_single(req, DEVICE)

        assert req.status == RequestStatus.FAILED
        assert "KV cache" in req.error

    def test_decode_batch_fails_requests_when_kv_exhausted(self):
        rt = make_runtime(max_blocks=2)
        reqs = [
            make_decode_req(rt.kv_cache_manager, prompt_len=16, alloc_tokens=16)
            for _ in range(2)
        ]
        runner = DecodeRunner(rt, StreamEmitter(rt), DecodeInputBuffer(), PersistentBatchCache())

        runner.decode_batch(reqs, DEVICE)

        assert all(r.status == RequestStatus.FAILED for r in reqs)
