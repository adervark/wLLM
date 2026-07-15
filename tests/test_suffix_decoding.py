"""Unit tests for SuffixDecoding: the suffix cache and the model-free engine."""

from unittest.mock import MagicMock

import pytest
import torch

from winllm.config import SamplingParams
from winllm.core.types import GenerationRequest
from winllm.inference.buffers import trim_cache
from winllm.inference.suffix_cache import PySuffixCache, SuffixCache
from winllm.inference.suffix_speculative import SuffixSpeculativeEngine

try:
    import winllm_suffix
except ImportError:
    winllm_suffix = None


# ─── SuffixCache ────────────────────────────────────────────────────────────


class TestSuffixCache:
    def test_no_history_no_proposal(self):
        cache = SuffixCache()
        assert cache.propose("r1") == []
        cache.sync("r1", [1, 2])
        assert cache.propose("r1") == []  # too short for an earlier match

    def test_no_repeat_no_proposal(self):
        cache = SuffixCache()
        cache.sync("r1", [1, 2, 3, 4, 5])
        assert cache.propose("r1") == []

    def test_repeated_pattern_proposes_continuation(self):
        cache = SuffixCache(alpha=2.0, max_spec=16)
        # Suffix (2, 3) previously occurred at pos 1, followed by 4, 1, 2, 3
        cache.sync("r1", [1, 2, 3, 4, 1, 2, 3])
        # Backward ext: tokens[0]=1 == tokens[4]=1 -> pattern_len 3, budget 6,
        # but only 4 tokens follow the earlier occurrence.
        assert cache.propose("r1") == [4, 1, 2, 3]

    def test_budget_scales_with_match_length(self):
        cache = SuffixCache(alpha=1.0, max_spec=16)
        cache.sync("r1", [9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3])
        # Earlier (2, 3) at pos 1 with backward ext 1 (9 == 9): pattern 3,
        # alpha=1 -> budget 3.
        assert cache.propose("r1") == [4, 5, 6]

    def test_most_recent_occurrence_wins_ties(self):
        cache = SuffixCache(alpha=2.0, max_spec=4)
        # (1, 2) occurs at pos 0 (followed by 7) and pos 3 (followed by 8);
        # equal pattern length -> the later occurrence wins.
        cache.sync("r1", [1, 2, 7, 1, 2, 8, 0, 1, 2])
        assert cache.propose("r1")[0] == 8

    def test_incremental_sync_only_indexes_tail(self):
        cache = SuffixCache()
        cache.sync("r1", [1, 2, 3])
        cache.sync("r1", [1, 2, 3, 4, 1, 2])
        # (1, 2) at pos 0 followed by [3, 4, ...]
        assert cache.propose("r1")[0] == 3

    def test_lru_eviction(self):
        cache = SuffixCache(max_requests=2)
        cache.sync("a", [1, 2, 3])
        cache.sync("b", [1, 2, 3])
        cache.sync("c", [1, 2, 3])
        assert not cache.has_request("a")
        assert cache.has_request("b") and cache.has_request("c")

    def test_evict(self):
        cache = SuffixCache()
        cache.sync("r1", [1, 2, 3, 1, 2])
        cache.evict("r1")
        assert cache.propose("r1") == []


# ─── Native (C++) parity ────────────────────────────────────────────────────


@pytest.mark.skipif(winllm_suffix is None, reason="native winllm_suffix not built")
class TestNativeParity:
    """The optional C++ SuffixCache must match the Python one exactly.

    Small vocab + tight limits maximize bigram collisions, tie-breaks, LRU
    eviction, and occurrence-cap hits — every branch both implementations
    share must agree on every step.
    """

    KW = dict(alpha=1.5, max_spec=8, max_back_ext=16, max_requests=4,
              max_occurrences=8)

    def test_random_stream_parity(self):
        import random

        rng = random.Random(0xC0FFEE)
        py_cache = PySuffixCache(**self.KW)
        cpp_cache = winllm_suffix.SuffixCache(**self.KW)

        # 6 request ids against max_requests=4 keeps LRU eviction in play
        streams = {f"r{i}": [] for i in range(6)}
        for step in range(400):
            rid = f"r{rng.randrange(6)}"
            streams[rid].extend(
                rng.randrange(12) for _ in range(rng.randrange(1, 4))
            )
            py_cache.sync(rid, streams[rid])
            cpp_cache.sync(rid, streams[rid])
            assert cpp_cache.propose(rid) == py_cache.propose(rid), (
                f"divergence at step {step} for {rid}"
            )

        for rid in streams:
            assert cpp_cache.has_request(rid) == py_cache.has_request(rid)
            cpp_cache.evict(rid)
            py_cache.evict(rid)
            assert cpp_cache.propose(rid) == [] == py_cache.propose(rid)


# ─── trim_cache ─────────────────────────────────────────────────────────────


class TestTrimCache:
    def test_none_passthrough(self):
        assert trim_cache(None, 5) is None

    def test_legacy_tuple_sliced(self):
        kv = ((torch.zeros(1, 1, 10, 4), torch.zeros(1, 1, 10, 4)),)
        out = trim_cache(kv, 6)
        assert out[0][0].shape[2] == 6

    def test_cache_object_cropped(self):
        cache = MagicMock(spec=["crop"])
        out = trim_cache(cache, 7)
        cache.crop.assert_called_once_with(7)
        assert out is cache


# ─── SuffixSpeculativeEngine ────────────────────────────────────────────────


def _make_mock_model(vocab_size=100, default_token=5):
    """Mock target model: ``default_token`` always argmax at every position."""
    model = MagicMock()
    param = torch.zeros(1)
    model.parameters.return_value = iter([param])

    def forward_fn(input_ids, past_key_values=None, use_cache=True):
        batch, seq_len = input_ids.shape
        output = MagicMock()
        logits = torch.full((batch, seq_len, vocab_size), -10.0)
        logits[:, :, default_token] = 10.0
        output.logits = logits
        output.past_key_values = (
            (torch.zeros(batch, 1, seq_len, 4), torch.zeros(batch, 1, seq_len, 4)),
        )
        return output

    model.side_effect = forward_fn
    return model


def _make_request(prompt, output):
    req = GenerationRequest(
        prompt_token_ids=prompt,
        output_token_ids=output,
        sampling_params=SamplingParams(temperature=0, max_tokens=100),
    )
    seq_len = len(prompt) + len(output)
    req._past_key_values = (
        (torch.zeros(1, 1, seq_len, 4), torch.zeros(1, 1, seq_len, 4)),
    )
    return req


def _make_engine(default_token=5):
    tok = MagicMock()
    tok.eos_token_id = 2
    return SuffixSpeculativeEngine(
        target_model=_make_mock_model(default_token=default_token), tokenizer=tok
    )


class TestSuffixSpeculativeEngine:
    def test_plain_step_when_no_match(self):
        engine = _make_engine()
        req = _make_request([1, 3, 4], [10])
        alive = engine.step(req)
        assert alive
        assert req.output_token_ids == [10, 5]  # exactly one committed token
        assert engine.drafted_tokens == 0

    def test_drafts_accepted_when_target_agrees(self):
        engine = _make_engine(default_token=5)
        # History [... 5, 5, 5]: suffix (5, 5) recurs, drafts are all 5s —
        # which is exactly what the mock target predicts, so all accepted
        # plus the bonus token from the final position.
        req = _make_request([1, 3], [5, 5, 5])
        before = len(req.output_token_ids)
        engine.step(req)
        added = len(req.output_token_ids) - before
        assert added >= 2  # multiple tokens from one "forward pass"
        assert engine.accepted_tokens == engine.drafted_tokens > 0
        assert all(t == 5 for t in req.output_token_ids[before:])

    def test_drafts_rejected_when_target_disagrees(self):
        engine = _make_engine(default_token=7)
        # Drafts will be 5s (from history), target insists on 7: first draft
        # rejected, but the target's own token is still committed.
        req = _make_request([1, 3], [5, 5, 5])
        before = len(req.output_token_ids)
        alive = engine.step(req)
        assert alive
        assert len(req.output_token_ids) == before + 1
        assert req.output_token_ids[-1] == 7
        assert engine.accepted_tokens == 0

    def test_eos_ends_generation_and_evicts(self):
        engine = _make_engine(default_token=2)  # target always emits EOS
        req = _make_request([1, 3], [5, 5, 5])
        alive = engine.step(req)
        assert not alive
        assert not engine.suffix_cache.has_request(req.request_id)

    def test_output_identical_to_plain_decode(self):
        # The invariant behind losslessness: committed tokens are always the
        # target's samples, drafts only batch the forward passes.
        spec = _make_engine(default_token=5)
        req_spec = _make_request([1, 3], [5, 5, 5])
        for _ in range(4):
            spec.step(req_spec)

        plain = _make_engine(default_token=5)
        req_plain = _make_request([1, 3], [5, 5, 5])
        while len(req_plain.output_token_ids) < len(req_spec.output_token_ids):
            plain._plain_step(req_plain)

        n = len(req_plain.output_token_ids)
        assert req_spec.output_token_ids[:n] == req_plain.output_token_ids[:n]

    def test_kv_cache_trimmed_to_committed_length(self):
        engine = _make_engine(default_token=7)  # first draft always rejected
        req = _make_request([1, 3], [5, 5, 5])
        engine.step(req)
        valid_len = len(req.prompt_token_ids) + len(req.output_token_ids) - 1
        assert req._past_key_values[0][0].shape[2] <= valid_len


class TestBudgetAndStreaming:
    def test_speculation_never_exceeds_max_tokens(self):
        engine = _make_engine(default_token=5)
        req = _make_request([1, 3], [5, 5, 5])
        req.sampling_params = SamplingParams(temperature=0, max_tokens=5)
        # Emulate the scheduler: step until the finish check would fire
        for _ in range(5):
            if len(req.output_token_ids) >= req.sampling_params.max_tokens:
                break
            engine.step(req)
        assert len(req.output_token_ids) <= 5

    def test_emitter_drains_all_speculative_tokens(self):
        from winllm.inference.streaming import StreamEmitter

        emitter = StreamEmitter(MagicMock())
        req = _make_request([1, 3], [10, 11, 12])
        received = []
        req._token_callback = lambda tok, fin: received.append(tok)
        emitter.emit(req)
        assert received == [10, 11, 12]  # not just the last token
        emitter.emit(req)
        assert received == [10, 11, 12]  # cursor prevents re-emission


class TestValidation:
    def test_deterministic_cache_validates(self):
        # Mock forwards depend only on the input, so cropping "junk" away
        # perfectly restores state: the probe must pass.
        engine = _make_engine()
        assert engine.validate([1, 2, 3, 4, 5]) is True

    def test_stateful_cache_fails_validation(self):
        # Model whose logits shift on every call — junk tokens leave a trace
        # that crop can't undo, like a hybrid conv cache.
        engine = _make_engine()
        calls = {"n": 0}

        def stateful_forward(input_ids, past_key_values=None, use_cache=True):
            calls["n"] += 1
            batch, seq_len = input_ids.shape
            output = MagicMock()
            output.logits = torch.full((batch, seq_len, 100), float(calls["n"]))
            output.past_key_values = (
                (torch.zeros(batch, 1, seq_len, 4), torch.zeros(batch, 1, seq_len, 4)),
            )
            return output

        engine.target_model.side_effect = stateful_forward
        assert engine.validate([1, 2, 3, 4, 5]) is False

    def test_probe_exception_fails_validation(self):
        engine = _make_engine()
        engine.target_model.side_effect = RuntimeError("boom")
        assert engine.validate([1, 2, 3, 4, 5]) is False

    def test_too_short_probe_fails_validation(self):
        engine = _make_engine()
        assert engine.validate([1]) is False
