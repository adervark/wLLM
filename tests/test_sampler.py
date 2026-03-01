"""Unit tests for the token sampler."""

import torch
import pytest
from winllm.sampler import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_token,
)
from winllm.config import SamplingParams


class TestApplyTemperature:
    def test_temperature_1(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_temperature(logits, 1.0)
        assert torch.allclose(result, logits)

    def test_temperature_higher(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_temperature(logits, 2.0)
        expected = logits / 2.0
        assert torch.allclose(result, expected)

    def test_temperature_zero_passthrough(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_temperature(logits, 0)
        # temperature=0 returns logits unchanged (greedy handled at sample stage)
        assert torch.allclose(result, logits)


class TestApplyTopK:
    def test_top_k_filters(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
        result = apply_top_k(logits, top_k=2)
        # Only top 2 values (5.0 and 4.0) should remain
        assert result[0, 1] == 5.0
        assert result[0, 3] == 4.0
        assert result[0, 0] == float("-inf")
        assert result[0, 2] == float("-inf")
        assert result[0, 4] == float("-inf")

    def test_top_k_larger_than_vocab(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_top_k(logits, top_k=10)
        assert torch.allclose(result, logits)

    def test_top_k_zero_no_filter(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_top_k(logits, top_k=0)
        assert torch.allclose(result, logits)


class TestApplyTopP:
    def test_top_p_1_no_filter(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_top_p(logits, top_p=1.0)
        assert torch.allclose(result, logits)

    def test_top_p_filters(self):
        # Very peaked distribution — low top_p should keep only the top token
        logits = torch.tensor([[0.0, 0.0, 10.0]])
        result = apply_top_p(logits, top_p=0.5)
        # Token at index 2 (highest) should survive
        assert result[0, 2] == 10.0


class TestRepetitionPenalty:
    def test_no_penalty(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits, [], 1.0)
        assert torch.allclose(result, logits)

    def test_penalty_reduces_repeated(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits.clone(), [2], 2.0)
        # Token 2 had logit 3.0 (positive), should be divided by penalty
        assert result[0, 2] == 3.0 / 2.0
        # Other tokens unchanged
        assert result[0, 0] == 1.0
        assert result[0, 1] == 2.0

    def test_penalty_on_negative_logits(self):
        logits = torch.tensor([[-1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits.clone(), [0], 2.0)
        # Token 0 had logit -1.0 (negative), should be multiplied by penalty
        assert result[0, 0] == -1.0 * 2.0


class TestSampleToken:
    def test_greedy(self):
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        params = SamplingParams(temperature=0, max_tokens=10)
        result = sample_token(logits, params)
        assert result.item() == 1  # Index of max logit

    def test_deterministic_with_seed(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        params = SamplingParams(temperature=1.0, seed=42, max_tokens=10)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        result1 = sample_token(logits, params, generator=gen)

        gen.manual_seed(42)
        result2 = sample_token(logits, params, generator=gen)

        assert result1.item() == result2.item()

    def test_returns_valid_index(self):
        vocab_size = 100
        logits = torch.randn(1, vocab_size)
        params = SamplingParams(temperature=0.8, top_k=10, top_p=0.9, max_tokens=10)
        result = sample_token(logits, params)
        assert 0 <= result.item() < vocab_size
