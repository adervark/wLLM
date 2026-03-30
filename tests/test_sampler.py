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


# ─── Full pipeline integration ──────────────────────────────────────────────


class TestSamplerFullPipeline:
    def test_all_stages_combined(self):
        """Test the full sampling pipeline: rep_penalty + temp + top_k + top_p."""
        vocab_size = 50
        logits = torch.randn(1, vocab_size)
        generated = [5, 10, 15, 20]
        params = SamplingParams(
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            repetition_penalty=1.2,
            max_tokens=10,
        )
        result = sample_token(logits, params, generated)
        assert 0 <= result.item() < vocab_size

    def test_greedy_with_repetition_penalty(self):
        """Greedy mode should still apply repetition penalty before argmax."""
        logits = torch.tensor([[5.0, 3.0, 1.0]])
        params = SamplingParams(temperature=0, repetition_penalty=100.0, max_tokens=10)
        # Without penalty, token 0 wins. With heavy penalty on token 0, token 1 should win.
        result = sample_token(logits, params, generated_ids=[0])
        assert result.item() == 1

    def test_high_temperature_increases_variety(self):
        """With high temperature, sampling should be more uniform (statistical test)."""
        vocab_size = 10
        logits = torch.tensor([[10.0] + [0.0] * (vocab_size - 1)])
        params = SamplingParams(temperature=100.0, top_k=0, top_p=1.0, max_tokens=10)

        # Sample many times — high temp should produce varied results
        results = set()
        for _ in range(100):
            token = sample_token(logits.clone(), params)
            results.add(token.item())
        # With temp=100, distribution is nearly uniform, so we expect multiple tokens
        assert len(results) > 1


# ─── Edge cases ─────────────────────────────────────────────────────────────


class TestSamplerEdgeCases:
    def test_repetition_penalty_with_duplicates_in_list(self):
        """Duplicate IDs in generated_ids should work (deduplication is internal)."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits.clone(), [2, 2, 2], 2.0)
        # Token 2 should be penalized once, not triple-penalized
        assert result[0, 2] == 3.0 / 2.0

    def test_top_p_with_uniform_distribution(self):
        """Uniform distribution with low top_p should still keep some tokens."""
        vocab_size = 5
        logits = torch.zeros(1, vocab_size)  # Uniform after softmax
        result = apply_top_p(logits, top_p=0.3)
        # At least one token should survive
        valid = (result > float("-inf")).sum()
        assert valid >= 1

    def test_top_k_equals_one(self):
        """top_k=1 should keep only the max logit."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
        result = apply_top_k(logits, top_k=1)
        valid_count = (result > float("-inf")).sum()
        assert valid_count == 1
        assert result[0, 1] == 5.0

    def test_sample_token_output_shape_1d(self):
        """Output should be 1D when input logits are 2D [1, vocab]."""
        logits = torch.randn(1, 50)
        params = SamplingParams(temperature=0, max_tokens=10)
        result = sample_token(logits, params)
        assert result.dim() == 0 or result.dim() == 1  # scalar or 1D

    def test_sample_token_clones_logits(self):
        """sample_token should not mutate the original logits tensor."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        original = logits.clone()
        params = SamplingParams(temperature=0.5, top_k=2, repetition_penalty=1.5, max_tokens=10)
        sample_token(logits, params, generated_ids=[0])
        # Original should be unchanged
        assert torch.allclose(logits, original)

    def test_very_small_temperature(self):
        """Very small (but non-zero) temperature → near-greedy behavior."""
        logits = torch.tensor([[1.0, 10.0, 5.0]])
        params = SamplingParams(temperature=0.001, top_k=0, top_p=1.0, max_tokens=10)
        result = sample_token(logits, params)
        assert result.item() == 1  # Should pick the highest


class TestBatchedSampler:
    def test_batched_repetition_penalty(self):
        logits = torch.tensor([
            [1.0, 2.0, 3.0], 
            [1.0, 2.0, 3.0]
        ])
        generated = [[2], []]
        penalties = [2.0, 1.0]
        result = apply_repetition_penalty(logits.clone(), generated, penalties)
        assert result[0, 2] == 1.5
        assert result[0, 1] == 2.0
        assert torch.allclose(result[1], logits[1])

    def test_batched_temperature(self):
        logits = torch.tensor([
            [2.0, 4.0, 6.0],
            [2.0, 4.0, 6.0]
        ])
        temps = [2.0, 0.5]
        result = apply_temperature(logits, temps)
        assert torch.allclose(result[0], logits[0] / 2.0)
        assert torch.allclose(result[1], logits[1] / 0.5)

    def test_batched_top_k(self):
        logits = torch.tensor([
            [1.0, 5.0, 3.0, 4.0, 2.0],
            [1.0, 5.0, 3.0, 4.0, 2.0]
        ])
        ks = [2, 1]
        result = apply_top_k(logits, ks)
        assert result[0, 1] == 5.0 and result[0, 3] == 4.0
        assert result[0, 0] == float('-inf')
        assert result[1, 1] == 5.0
        assert result[1, 3] == float('-inf')

    def test_batched_mixed_sample_token(self):
        logits = torch.tensor([
            [1.0, 10.0, 2.0],
            [1.0, 2.0, 10.0]
        ])
        p1 = SamplingParams(temperature=0)
        p2 = SamplingParams(temperature=1.0)
        gen = torch.Generator(device=logits.device).manual_seed(42)
        result = sample_token(logits, [p1, p2], generator=gen)
        assert result.shape == (2,)
        assert result[0].item() == 1

