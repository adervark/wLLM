"""Unit tests for the speculative decoding engine."""

from unittest.mock import MagicMock
import pytest
import torch

from winllm.speculative import SpeculativeEngine
from winllm.config import SamplingParams
from winllm.types import GenerationRequest


# ─── Fixtures ───────────────────────────────────────────────────────────────


def _make_mock_model(eos_token_id=2, vocab_size=100, default_token=5):
    """Create a mock model that produces predictable outputs."""
    model = MagicMock()
    param = torch.zeros(1)
    model.parameters.return_value = iter([param])

    def forward_fn(input_ids, past_key_values=None, use_cache=True):
        batch, seq_len = input_ids.shape
        output = MagicMock()
        # Return logits where default_token always has the highest value
        logits = torch.full((batch, seq_len, vocab_size), -10.0)
        logits[:, :, default_token] = 10.0
        output.logits = logits
        output.past_key_values = ((torch.zeros(batch, 1, seq_len, 4), torch.zeros(batch, 1, seq_len, 4)),)
        return output

    model.side_effect = forward_fn
    return model


def _make_tokenizer(eos_token_id=2):
    tok = MagicMock()
    tok.eos_token_id = eos_token_id
    return tok


@pytest.fixture
def setup():
    """Create a speculative engine with mocked target and draft models."""
    target = _make_mock_model(default_token=5)
    draft = _make_mock_model(default_token=5)  # Same predictions → all accepted
    tokenizer = _make_tokenizer()

    engine = SpeculativeEngine(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        num_speculative_tokens=3,
    )

    request = GenerationRequest(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[10],
        sampling_params=SamplingParams(temperature=0, max_tokens=100),
    )
    # Initialize past KV caches with proper (key, value) structure per layer
    # Shape: ((batch, num_heads, seq_len, head_dim), ...)
    seq_len = len(request.prompt_token_ids) + len(request.output_token_ids)
    request._past_key_values = ((torch.zeros(1, 1, seq_len, 4), torch.zeros(1, 1, seq_len, 4)),)
    request._draft_past_key_values = None

    return engine, request


# ─── Draft proposals ────────────────────────────────────────────────────────


class TestDraftProposals:
    def test_generates_correct_number(self, setup):
        engine, request = setup
        tokens = engine._draft_proposals(request)
        # Should generate num_speculative_tokens (3) proposals
        assert len(tokens) == 3

    def test_stops_early_on_eos(self, setup):
        engine, request = setup
        # Make draft model produce EOS immediately
        engine.draft_model.side_effect = None
        eos_output = MagicMock()
        logits = torch.full((1, 1, 100), -10.0)
        logits[:, :, engine.tokenizer.eos_token_id] = 10.0
        eos_output.logits = logits
        eos_output.past_key_values = ((torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4)),)
        engine.draft_model.return_value = eos_output

        tokens = engine._draft_proposals(request)
        assert len(tokens) >= 1
        assert tokens[-1] == engine.tokenizer.eos_token_id

    def test_uses_last_output_token(self, setup):
        engine, request = setup
        request.output_token_ids = [42]
        engine._draft_proposals(request)
        # First call to draft model should use token 42
        first_call = engine.draft_model.call_args_list[0]
        input_ids = first_call[0][0]
        assert input_ids[0, 0].item() == 42


# ─── Verification ───────────────────────────────────────────────────────────


class TestVerifyProposals:
    def test_verify_input_shape(self, setup):
        engine, request = setup
        draft_tokens = [5, 5, 5]
        engine._verify_proposals(request, draft_tokens)

        # Target model should receive [last_output_token] + draft_tokens
        call_args = engine.target_model.call_args_list[0]
        input_ids = call_args[0][0]
        assert input_ids.shape == (1, 4)  # 1 (last) + 3 (draft)

    def test_updates_past_key_values(self, setup):
        engine, request = setup
        old_kv = request._past_key_values
        engine._verify_proposals(request, [5, 5, 5])
        # past_key_values should be updated
        assert request._past_key_values is not old_kv


# ─── Accept/Reject ──────────────────────────────────────────────────────────


class TestAcceptOrReject:
    def test_all_accepted_plus_bonus(self, setup):
        engine, request = setup
        # Both models produce token 5 (via greedy decoding with temp=0)
        # logits with token 5 highest → argmax returns 5
        logits = torch.full((4, 100), -10.0)  # 3 draft + 1 bonus
        logits[:, 5] = 10.0

        initial_len = len(request.output_token_ids)
        result = engine._accept_or_reject(request, [5, 5, 5], logits)

        # Should have accepted all 3 + bonus = 4 new tokens
        assert result is True
        assert len(request.output_token_ids) == initial_len + 4

    def test_rejects_at_first_mismatch(self, setup):
        engine, request = setup
        # Target model would produce token 7, not 5
        logits = torch.full((4, 100), -10.0)
        logits[:, 7] = 10.0  # Target prefers 7

        initial_len = len(request.output_token_ids)
        result = engine._accept_or_reject(request, [5, 5, 5], logits)

        # First draft (5) mismatches target (7), so only 1 token (the correction) is added
        assert result is True
        assert len(request.output_token_ids) == initial_len + 1
        assert request.output_token_ids[-1] == 7

    def test_returns_false_on_eos(self, setup):
        engine, request = setup
        eos = engine.tokenizer.eos_token_id

        # Target produces EOS at first position
        logits = torch.full((4, 100), -10.0)
        logits[:, eos] = 10.0

        result = engine._accept_or_reject(request, [eos, 5, 5], logits)
        assert result is False

    def test_resets_draft_kv_cache(self, setup):
        engine, request = setup
        request._draft_past_key_values = ((torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)),)

        logits = torch.full((4, 100), -10.0)
        logits[:, 7] = 10.0  # Mismatch

        engine._accept_or_reject(request, [5, 5, 5], logits)
        assert request._draft_past_key_values is None


# ─── Full step ──────────────────────────────────────────────────────────────


class TestSpeculativeStep:
    def test_step_adds_tokens(self, setup):
        engine, request = setup
        initial_len = len(request.output_token_ids)
        engine.step(request)
        assert len(request.output_token_ids) > initial_len

    def test_step_returns_bool(self, setup):
        engine, request = setup
        result = engine.step(request)
        assert isinstance(result, bool)
