"""Parity and behavior tests for the fused CUDA sampling kernel.

The native module (native/sampling/) must sample from exactly the
distribution the torch pipeline (winllm/sampling/ops.py) defines: same kept
set after top-k/top-p masking, same renormalized probabilities, bit-exact
greedy. Sampled *tokens* can never match the torch path draw-for-draw (the
kernel has its own Philox stream — that's why seeded requests are excluded),
so the tests check masking support exactly and frequencies statistically.

Everything here needs a GPU and the built module; CI (CPU-only, no nvcc)
skips the whole file.
"""

from __future__ import annotations

import pytest
import torch

try:
    import winllm_sampling
except ImportError:
    winllm_sampling = None

from winllm.config import SamplingParams
from winllm.sampling import ops, sample_token
from winllm.sampling.native import FusedSampler

requires_native = pytest.mark.skipif(
    winllm_sampling is None or not torch.cuda.is_available(),
    reason="native winllm_sampling not built or no CUDA device",
)

DEV = "cuda"


def reference_probs(logits, params: SamplingParams, generated_ids=None):
    """The distribution the torch pipeline defines, computed in fp32."""
    x = logits.clone().float()
    x = ops.apply_repetition_penalty(x, generated_ids or [], params.repetition_penalty)
    if params.temperature > 0:
        x = ops.apply_temperature(x, params.temperature)
        x = ops.apply_top_k(x, params.top_k)
        x = ops.apply_top_p(x, params.top_p)
    return torch.softmax(x, dim=-1)


def draw_many(sampler, logits, params: SamplingParams, generated_ids=None, n=20000):
    """Collect n draws by batching identical rows through the kernel."""
    vocab = logits.shape[-1]
    batch = 500
    rows = logits.expand(batch, vocab).contiguous()
    draws = []
    for _ in range(n // batch):
        if generated_ids:
            # penalty + batch>1 is rejected by design; sample rows one by one
            out = torch.cat([
                sampler.sample(
                    rows[i : i + 1],
                    params.temperature, params.top_k, params.top_p,
                    params.repetition_penalty, generated_ids,
                )
                for i in range(batch)
            ])
        else:
            out = sampler.sample(
                rows, params.temperature, params.top_k, params.top_p,
                params.repetition_penalty, None,
            )
        assert out is not None
        draws.append(out)
    return torch.cat(draws)


@requires_native
class TestFusedKernelParity:
    def setup_method(self):
        torch.manual_seed(1234)
        self.sampler = FusedSampler()
        assert self.sampler.enabled

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "params",
        [
            SamplingParams(temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.0),
            SamplingParams(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0),
            SamplingParams(temperature=1.3, top_k=200, top_p=0.95, repetition_penalty=1.0),
            SamplingParams(temperature=0.4, top_k=3, top_p=1.0, repetition_penalty=1.0),
        ],
    )
    def test_support_and_frequencies(self, dtype, params):
        vocab = 384
        logits = (torch.randn(1, vocab, device=DEV) * 2).to(dtype)
        ref = reference_probs(logits, params)[0].cpu()
        draws = draw_many(self.sampler, logits, params, n=20000).cpu()

        # Every sampled token must be inside the torch pipeline's kept set.
        # Random continuous logits keep the top-p boundary unambiguous, so
        # this holds exactly despite fp32-vs-fp16 summation differences.
        outside = [t for t in torch.unique(draws).tolist() if ref[t] == 0]
        assert outside == []

        # Frequencies within statistical noise of the reference
        # distribution: 20k draws, tolerance ~6 sigma + slack.
        emp = torch.bincount(draws, minlength=vocab).float() / draws.numel()
        sigma = (ref * (1 - ref) / draws.numel()).sqrt()
        assert ((emp - ref).abs() <= 6 * sigma + 5e-3).all()

    def test_penalty_shifts_distribution(self):
        vocab = 256
        params = SamplingParams(temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.4)
        logits = (torch.randn(1, vocab, device=DEV) * 2).half()
        generated = [1, 5, 5, 5, 9, 20, 20]  # duplicates: penalize once each
        ref = reference_probs(logits, params, generated)[0].cpu()
        draws = draw_many(self.sampler, logits, params, generated, n=10000).cpu()
        outside = [t for t in torch.unique(draws).tolist() if ref[t] == 0]
        assert outside == []
        emp = torch.bincount(draws, minlength=vocab).float() / draws.numel()
        sigma = (ref * (1 - ref) / draws.numel()).sqrt()
        assert ((emp - ref).abs() <= 6 * sigma + 5e-3).all()

    def test_greedy_with_penalty_exact(self):
        params = SamplingParams(temperature=0.0, repetition_penalty=1.25)
        generated = list(range(0, 300, 7))
        for _ in range(100):
            logits = (torch.randn(1, 500, device=DEV) * 2).half()
            x = logits.clone().float()
            x = ops.apply_repetition_penalty(x, generated, params.repetition_penalty)
            want = torch.argmax(x, dim=-1).item()
            got = self.sampler.sample(
                logits, 0.0, params.top_k, params.top_p,
                params.repetition_penalty, generated,
            )
            assert got.item() == want

    def test_top_p_exact_support_handcrafted(self):
        # probs 0.5/0.3/0.15/0.05 at temperature 1: top_p=0.8 keeps exactly
        # {0, 1} under the sorted-cumsum rule.
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]], device=DEV)
        logits = probs.log().float()
        draws = draw_many(
            self.sampler, logits,
            SamplingParams(temperature=1.0, top_k=0, top_p=0.8, repetition_penalty=1.0),
            n=4000,
        )
        assert set(torch.unique(draws).tolist()) == {0, 1}

    def test_top_k_exact_support(self):
        vocab = 300
        logits = torch.argsort(torch.rand(1, vocab, device=DEV)).float()  # distinct 0..299
        draws = draw_many(
            self.sampler, logits,
            SamplingParams(temperature=5.0, top_k=8, top_p=1.0, repetition_penalty=1.0),
            n=4000,
        )
        kept = set(torch.topk(logits[0], 8).indices.tolist())
        assert set(torch.unique(draws).tolist()) <= kept
        assert len(set(torch.unique(draws).tolist())) > 1  # actually sampling

    def test_batch_rows_independent(self):
        # Two rows peaked on different tokens must sample their own token.
        logits = torch.full((2, 64), -10.0, device=DEV).half()
        logits[0, 3] = 10.0
        logits[1, 40] = 10.0
        out = self.sampler.sample(logits, 0.7, 50, 0.9, 1.0, None)
        assert out.tolist() == [3, 40]

    def test_fully_masked_row_falls_back_to_argmax(self):
        logits = torch.full((1, 64), float("-inf"), device=DEV).half()
        out = self.sampler.sample(logits, 0.7, 50, 0.9, 1.0, None)
        assert out.item() == 0


@requires_native
class TestBitmaskKernel:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("vocab", [64, 1000, 49152])  # word-aligned and not
    def test_matches_torch_unpack(self, dtype, vocab):
        from winllm.sampling.grammar import unpack_bitmask

        torch.manual_seed(0)
        words = (vocab + 31) // 32
        packed = torch.randint(-(2**31), 2**31 - 1, (1, words), dtype=torch.int32)
        logits = torch.randn(vocab, device=DEV).to(dtype)

        ref = logits.clone()
        keep = unpack_bitmask(packed, vocab, ref.device)
        ref.masked_fill_(~keep, float("-inf"))

        got = logits.clone()
        s = FusedSampler()
        assert s.apply_bitmask(got, packed)
        assert torch.equal(got, ref)

    def test_cpu_row_falls_back(self):
        s = FusedSampler()
        assert not s.apply_bitmask(torch.randn(64), torch.zeros(1, 2, dtype=torch.int32))


@requires_native
class TestSamplerIntegration:
    def test_sample_token_uses_fused_path(self, monkeypatch):
        from winllm.sampling import native

        calls = []
        orig = native.fused_sampler.sample

        def spy(*args, **kwargs):
            r = orig(*args, **kwargs)
            calls.append(r is not None)
            return r

        monkeypatch.setattr(native.fused_sampler, "sample", spy)
        logits = torch.randn(1, 128, device=DEV).half()
        params = SamplingParams()  # chat defaults: 0.7/50/0.9/pen 1.1
        tok = sample_token(logits, params, generated_ids=[1, 2, 3])
        assert calls == [True]
        assert 0 <= tok.item() < 128

    def test_seeded_request_stays_on_torch_path(self, monkeypatch):
        from winllm.sampling import native

        def boom(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("fused path used for a seeded request")

        monkeypatch.setattr(native.fused_sampler, "sample", boom)
        logits = torch.randn(1, 128, device=DEV).half()
        gen = torch.Generator(device=DEV)
        gen.manual_seed(7)
        params = SamplingParams(temperature=0.7)
        sample_token(logits, params, generated_ids=[1], generator=gen)

    def test_seeded_reproducibility_end_to_end(self):
        # The whole point of excluding seeded requests: same seed, same tokens.
        logits = torch.randn(1, 256, device=DEV).half()
        params = SamplingParams(temperature=0.9)

        def run():
            gen = torch.Generator(device=DEV)
            gen.manual_seed(42)
            return [
                sample_token(logits, params, generated_ids=[1, 2], generator=gen).item()
                for _ in range(20)
            ]

        assert run() == run()

    def test_mixed_batch_params_fall_back(self, monkeypatch):
        from winllm.sampling import native

        calls = []
        monkeypatch.setattr(
            native.fused_sampler, "sample",
            lambda *a, **k: calls.append(1) or None,
        )
        logits = torch.randn(2, 128, device=DEV).half()
        params = [
            SamplingParams(temperature=0.7, repetition_penalty=1.0),
            SamplingParams(temperature=1.1, repetition_penalty=1.0),
        ]
        tok = sample_token(logits, params)
        assert tok.shape == (2,)
        assert calls == []  # non-uniform temperature: never offered to the kernel

    def test_cpu_logits_fall_back(self):
        logits = torch.randn(1, 64)
        tok = sample_token(logits, SamplingParams())
        assert tok.shape == (1,)

    def test_kernel_failure_disables_native_path(self, monkeypatch):
        from winllm.sampling.native import FusedSampler

        s = FusedSampler()
        assert s.enabled

        import winllm.sampling.native as native_mod

        class _Boom:
            @staticmethod
            def sample(*a, **k):
                raise RuntimeError("CUDA error: injected")

        monkeypatch.setattr(native_mod, "_module", _Boom)
        logits = torch.randn(1, 64, device=DEV).half()
        assert s.sample(logits, 0.7, 50, 0.9, 1.0, None) is None
        assert s.enabled is False
        # subsequent calls short-circuit without touching the module
        assert s.sample(logits, 0.7, 50, 0.9, 1.0, None) is None
