"""Tests for grammar-constrained decoding (structured output)."""

import math
from unittest.mock import MagicMock

import pytest
import torch

from winllm.config import SamplingParams
from winllm.sampling import sample_token
from winllm.sampling.grammar import GrammarState, unpack_bitmask
from winllm.server.schemas import ChatCompletionRequest

try:
    import xgrammar as xgr
    HAS_XGRAMMAR = True
except ImportError:
    HAS_XGRAMMAR = False


def make_bitmask(vocab_size, allowed):
    """Pack an allowed-token set into xgrammar's int32 bitmask layout."""
    n_words = math.ceil(vocab_size / 32)
    words = [0] * n_words
    for token in allowed:
        words[token // 32] |= 1 << (token % 32)
    # Emulate int32 overflow for bit 31
    words = [(w & 0xFFFFFFFF) - (1 << 32) if w & (1 << 31) else w for w in words]
    return torch.tensor([words], dtype=torch.int32)


class TestUnpackBitmask:
    def test_unpacks_expected_bits(self):
        vocab = 70
        allowed = {0, 5, 31, 32, 69}
        keep = unpack_bitmask(make_bitmask(vocab, allowed), vocab, torch.device("cpu"))
        assert keep.shape == (vocab,)
        for i in range(vocab):
            assert keep[i].item() == (i in allowed), f"token {i}"

    @pytest.mark.skipif(not HAS_XGRAMMAR, reason="xgrammar not installed")
    def test_matches_xgrammar_reference_kernel(self):
        """Our torch unpacking must agree with xgrammar's own CPU apply."""
        vocab = 100
        allowed = {1, 33, 64, 99}
        bitmask = make_bitmask(vocab, allowed)

        reference = torch.zeros(1, vocab)
        xgr.apply_token_bitmask_inplace(reference, bitmask)

        keep = unpack_bitmask(bitmask, vocab, torch.device("cpu"))
        ours = torch.zeros(1, vocab)
        ours[0].masked_fill_(~keep, float("-inf"))

        assert torch.equal(reference, ours)


class FakeGrammar:
    """Duck-typed GrammarState allowing a fixed token set."""

    def __init__(self, allowed):
        self.allowed = allowed
        self.seen_ids = []

    def apply(self, logits_row, generated_ids):
        self.seen_ids.append(list(generated_ids))
        mask = torch.ones_like(logits_row, dtype=torch.bool)
        for t in self.allowed:
            mask[t] = False
        logits_row.masked_fill_(mask, float("-inf"))


class TestSamplerIntegration:
    def test_greedy_respects_grammar_mask(self):
        logits = torch.zeros(1, 10)
        logits[0, 9] = 100.0  # unconstrained argmax would be 9
        params = SamplingParams(temperature=0.0)
        token = sample_token(logits, params, [1, 2], grammars=[FakeGrammar({7})])
        assert token.item() == 7

    def test_grammar_receives_generated_ids(self):
        grammar = FakeGrammar({0})
        sample_token(torch.zeros(1, 10), SamplingParams(temperature=0.0), [5, 6, 7], grammars=[grammar])
        assert grammar.seen_ids == [[5, 6, 7]]

    def test_batched_mixed_constrained_rows(self):
        logits = torch.zeros(2, 10)
        logits[:, 9] = 100.0
        params = [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)]
        tokens = sample_token(
            logits, params, [[1], [2]], grammars=[FakeGrammar({3}), None]
        )
        assert tokens[0].item() == 3  # constrained row
        assert tokens[1].item() == 9  # free row keeps its argmax

    def test_no_grammars_is_untouched_fast_path(self):
        logits = torch.zeros(1, 10)
        logits[0, 4] = 1.0
        token = sample_token(logits, SamplingParams(temperature=0.0), [], grammars=None)
        assert token.item() == 4

    def test_caller_logits_not_mutated(self):
        logits = torch.zeros(1, 10)
        original = logits.clone()
        sample_token(logits, SamplingParams(temperature=0.0), [], grammars=[FakeGrammar({2})])
        assert torch.equal(logits, original)


class TestGrammarState:
    def _state(self, matcher, vocab=64):
        fake_xgr = MagicMock()
        fake_xgr.allocate_token_bitmask.return_value = torch.zeros(1, 2, dtype=torch.int32)
        return GrammarState(fake_xgr, matcher, vocab)

    def test_advance_accepts_each_token_once(self):
        matcher = MagicMock()
        matcher.is_terminated.return_value = False
        matcher.accept_token.return_value = True
        state = self._state(matcher)

        state.apply(torch.zeros(64), [1, 2])
        state.apply(torch.zeros(64), [1, 2, 3])

        accepted = [c.args[0] for c in matcher.accept_token.call_args_list]
        assert accepted == [1, 2, 3]

    def test_rejection_releases_constraint(self):
        matcher = MagicMock()
        matcher.is_terminated.return_value = False
        matcher.accept_token.return_value = False
        state = self._state(matcher)

        logits = torch.full((64,), 1.0)
        state.apply(logits, [42])
        assert state.is_terminated
        # Constraint released: logits untouched afterwards
        assert torch.all(logits == 1.0)

    def test_terminated_matcher_stops_masking(self):
        matcher = MagicMock()
        matcher.is_terminated.return_value = True
        state = self._state(matcher)

        logits = torch.full((64,), 1.0)
        state.apply(logits, [])
        matcher.fill_next_token_bitmask.assert_not_called()
        assert torch.all(logits == 1.0)


class TestSchema:
    def test_response_format_accepted(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}

    def test_response_format_defaults_none(self):
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.response_format is None
