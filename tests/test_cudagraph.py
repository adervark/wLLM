"""CPU-safe tests for the CUDA graph decode path (fallback behavior).

The capture/replay happy path requires a CUDA device and a real model;
it is exercised by the load-time validate() self-check. These tests cover
the guard rails that must hold everywhere.
"""

import logging

import torch

from winllm.config import ModelConfig
from winllm.inference.cudagraph import CUDAGraphDecoder
from winllm.inference.runtime import ModelRuntime


class DummyModel:
    class config:
        pass


class HybridModel:
    class config:
        # LFM2-style hybrid layout: conv layers mixed with attention
        layer_types = ["conv", "conv", "full_attention", "conv"]


def _config(**kw):
    return ModelConfig(model_name_or_path="dummy", **kw)


class TestCUDAGraphDecoderGuards:
    def test_cpu_device_disables_decoder(self):
        decoder = CUDAGraphDecoder(DummyModel(), _config(), torch.device("cpu"))
        assert decoder.ok is False

    def test_hybrid_layer_types_disable_decoder(self):
        # Rejected at construction, before any device/StaticCache work
        decoder = CUDAGraphDecoder(HybridModel(), _config(), torch.device("cpu"))
        assert decoder.ok is False
        assert decoder._cache is None

    def test_attention_only_layer_types_pass_arch_guard(self, caplog):
        class SlidingModel:
            class config:
                layer_types = ["full_attention", "sliding_attention"]

        # Still disabled on CPU, but by the device guard, not the arch guard —
        # attention-only layouts must not be rejected as hybrids.
        with caplog.at_level(logging.INFO, logger="winllm.inference.cudagraph"):
            decoder = CUDAGraphDecoder(SlidingModel(), _config(), torch.device("cpu"))
        assert decoder.ok is False
        assert "hybrid" not in caplog.text
        assert "not on a CUDA device" in caplog.text

    def test_static_cache_failure_disables_decoder(self):
        if not torch.cuda.is_available():
            # Covered by the CPU guard above on GPU-less machines
            return
        # DummyModel.config lacks everything StaticCache needs -> init fails
        decoder = CUDAGraphDecoder(DummyModel(), _config(), torch.device("cuda"))
        assert decoder.ok is False

    def test_release_marks_unusable(self):
        decoder = CUDAGraphDecoder(DummyModel(), _config(), torch.device("cpu"))
        decoder.release()
        assert decoder.ok is False
        assert decoder._graph is None
        assert decoder._cache is None


class TestLogitsAgree:
    """The validation comparator: mutual top-k containment of argmaxes."""

    def test_identical_logits_agree(self):
        logits = torch.arange(100, dtype=torch.float32).reshape(1, 100)
        assert CUDAGraphDecoder._logits_agree(logits, logits.clone())

    def test_near_tie_argmax_flip_agrees(self):
        # fp16-noise scenario: top-2 swap places but both stay in each
        # other's top-k — must NOT be treated as divergence.
        ref = torch.zeros(1, 100)
        test = torch.zeros(1, 100)
        ref[0, 10], ref[0, 20] = 5.001, 5.000
        test[0, 10], test[0, 20] = 5.000, 5.001
        assert CUDAGraphDecoder._logits_agree(ref, test)

    def test_disjoint_top_tokens_disagree(self):
        # Corruption scenario: the paths favor entirely different tokens.
        ref = torch.arange(100, 0, -1, dtype=torch.float32).reshape(1, 100)   # argmax 0
        test = torch.arange(100, dtype=torch.float32).reshape(1, 100)         # argmax 99
        assert not CUDAGraphDecoder._logits_agree(ref, test)

    def test_nonfinite_logits_disagree(self):
        ref = torch.zeros(1, 100)
        test = torch.full((1, 100), float("nan"))
        assert not CUDAGraphDecoder._logits_agree(ref, test)

    def test_small_vocab_smaller_than_top_k(self):
        logits = torch.tensor([[1.0, 2.0]])
        assert CUDAGraphDecoder._logits_agree(logits, logits.clone())


class FakeVerifyGraph:
    """Stands in for CUDAGraphDecoder in _speculative_commit tests."""

    VERIFY_BUCKET = 8

    def __init__(self, favorite_token, vocab=100):
        self.favorite = favorite_token
        self.advanced = 0

    def verify(self, tokens):
        logits = torch.full((len(tokens), 100), -10.0)
        logits[:, self.favorite] = 10.0
        return logits

    def advance(self, n):
        self.advanced += n


class TestSpeculativeCommit:
    """BlockingGenerator._speculative_commit: graph-verified suffix drafts."""

    def _make(self, favorite_token):
        from unittest.mock import MagicMock

        from winllm.config import SamplingParams
        from winllm.core.types import GenerationRequest
        from winllm.inference.generation import BlockingGenerator
        from winllm.inference.suffix_cache import SuffixCache

        gen = BlockingGenerator(MagicMock(), _config(), MagicMock(), MagicMock())
        graph = FakeVerifyGraph(favorite_token)
        cache = SuffixCache()
        telemetry = MagicMock(drafted_tokens=0, accepted_tokens=0)
        req = GenerationRequest(
            prompt_token_ids=[1, 3],
            output_token_ids=[5, 5, 5],  # suffix (5,5) recurs -> drafts of 5s
            sampling_params=SamplingParams(temperature=0, max_tokens=100),
        )
        return gen, req, (graph, cache, telemetry)

    def test_accepted_drafts_commit_multiple_tokens(self):
        gen, req, spec = self._make(favorite_token=5)
        n, accepted = gen._speculative_commit(
            req, req.sampling_params, None, req.output_token_ids[-1],
            100, {2}, spec,
        )
        graph = spec[0]
        assert n >= 2  # accepted draft(s) + bonus
        assert accepted == n - 1  # everything but the correction/bonus row
        assert graph.advanced == 1 + (n - 1)  # last committed token not in cache
        assert all(t == 5 for t in req.output_token_ids[3:])

    def test_rejected_draft_commits_correction_and_rewinds(self):
        gen, req, spec = self._make(favorite_token=7)  # target disagrees
        n, accepted = gen._speculative_commit(
            req, req.sampling_params, None, req.output_token_ids[-1],
            100, {2}, spec,
        )
        assert (n, accepted) == (1, 0)
        assert req.output_token_ids[-1] == 7  # the target's correction
        assert spec[0].advanced == 1  # only next_token entered the cache

    def test_budget_caps_drafts(self):
        gen, req, spec = self._make(favorite_token=5)
        # 3 committed, max_tokens=5 -> budget 1 draft; accept+bonus = exactly 5
        n, _ = gen._speculative_commit(
            req, req.sampling_params, None, req.output_token_ids[-1],
            5, {2}, spec,
        )
        assert len(req.output_token_ids) <= 5

    def test_eos_from_target_stops_commit(self):
        gen, req, spec = self._make(favorite_token=2)  # target emits EOS
        n, accepted = gen._speculative_commit(
            req, req.sampling_params, None, req.output_token_ids[-1],
            100, {2}, spec,
        )
        assert (n, accepted) == (1, 0)
        assert req.output_token_ids[-1] == 2

    def test_seeded_sampling_consumes_rng_like_plain_decode(self):
        """Seeded temp>0 requests must draw one (1, vocab) sample per
        committed token — exactly like plain decode — or turning
        speculation on/off changes a reproducible request's output."""
        from winllm.config import SamplingParams
        from winllm.inference.suffix_cache import PySuffixCache
        from winllm.sampling import sample_token

        gen, req, spec = self._make(favorite_token=5)
        params = SamplingParams(
            temperature=1.0, seed=1234, max_tokens=100, repetition_penalty=1.0,
        )
        req.sampling_params = params
        graph = spec[0]

        # Entropy-rich rows so RNG consumption order actually matters
        fixed_logits = torch.randn(
            graph.VERIFY_BUCKET + 1, 100,
            generator=torch.Generator().manual_seed(7),
        )
        graph.verify = lambda tokens: fixed_logits[:len(tokens)]

        # Drafts the commit will see (probe cache, same state as spec's)
        probe = PySuffixCache()
        probe.sync(req.request_id, req.prompt_token_ids + req.output_token_ids)
        drafts = probe.propose(req.request_id)[:graph.VERIFY_BUCKET]
        assert drafts, "test setup must yield at least one draft"

        g_commit = torch.Generator().manual_seed(1234)
        gen._speculative_commit(
            req, params, g_commit, req.output_token_ids[-1], 100, {2}, spec,
        )
        committed = req.output_token_ids[3:]

        # Reference: plain decode's accept loop — one single-row draw per
        # committed token, nothing drawn past the first rejection.
        g_plain = torch.Generator().manual_seed(1234)
        rows = fixed_logits[: len(drafts) + 1]
        expected: list[int] = []
        for i, draft in enumerate(drafts):
            tok = sample_token(rows[i:i + 1, :], params, expected, g_plain).item()
            expected.append(tok)
            if tok in {2} or tok != draft:
                break
        else:
            expected.append(
                sample_token(rows[-1:, :], params, expected, g_plain).item()
            )

        assert committed == expected
        # The RNG streams must stay aligned for every subsequent step too
        assert torch.equal(g_commit.get_state(), g_plain.get_state())


class TestVerifyThrottle:
    """Exponential backoff when every draft keeps getting rejected."""

    def _attempts(self, throttle, opportunities, accepted=0):
        tried = 0
        for _ in range(opportunities):
            if throttle.should_try():
                tried += 1
                throttle.record(accepted)
        return tried

    def test_rejections_back_off_geometrically(self):
        from winllm.inference.generation import _VerifyThrottle

        # 100 opportunities, all-rejected verifies: attempts grow ~log then
        # settle at one per max_backoff.
        tried = self._attempts(_VerifyThrottle(max_backoff=32), 100)
        assert tried <= 9

    def test_acceptance_keeps_full_speculation(self):
        from winllm.inference.generation import _VerifyThrottle

        tried = self._attempts(_VerifyThrottle(), 50, accepted=3)
        assert tried == 50  # never throttled while drafts are accepted

    def test_acceptance_resets_backoff(self):
        from winllm.inference.generation import _VerifyThrottle

        t = _VerifyThrottle()
        for _ in range(20):  # drive backoff to the cap
            if t.should_try():
                t.record(0)
        while not t.should_try():  # drain the pending skip window
            pass
        t.record(2)  # an acceptance...
        assert t.should_try()  # ...restores speculation immediately
        t.record(0)
        assert not t.should_try()  # and the next backoff starts at 1 again
        assert t.should_try()


class TestRuntimeIntegration:
    def test_clear_releases_graph_decoder(self):
        runtime = ModelRuntime()
        decoder = CUDAGraphDecoder(DummyModel(), _config(), torch.device("cpu"))
        runtime.graph_decoder = decoder
        runtime.clear()
        assert runtime.graph_decoder is None

    def test_config_flag_defaults_off(self):
        assert _config().enable_cuda_graphs is False
        assert _config(enable_cuda_graphs=True).enable_cuda_graphs is True
