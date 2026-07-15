import pytest
import torch
import time
from unittest.mock import MagicMock, patch

from winllm.inference import InferenceEngine
from winllm.config import ModelConfig, KVCacheConfig, SamplingParams
from winllm.core.types import GenerationRequest, RequestStatus
from winllm.models.loader import ModelLoader

class DummyOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values
        
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Just a dummy parameter so device resolution works
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, input_ids, past_key_values=None, use_cache=True, attention_mask=None, position_ids=None):
        batch_size = input_ids.shape[0]
        vocab_size = 100
        # Return random logits
        logits = torch.randn(batch_size, 1, vocab_size, device=input_ids.device)
        
        # Emulate past key values format: tuple of (key, val) for each layer
        # Here we just use 1 layer
        num_heads = 2
        head_dim = 4
        
        if past_key_values is None:
            # Prefill: seq_len = input_ids.shape[1]
            seq_len = input_ids.shape[1]
            new_pkv = ((
                torch.zeros(batch_size, num_heads, seq_len, head_dim, device=input_ids.device),
                torch.zeros(batch_size, num_heads, seq_len, head_dim, device=input_ids.device)
            ),)
        else:
            # Decode: input_ids is length 1. seq_len = past_key_values seq_len + 1
            # Check the padded length from the provided past_key_values
            past_len = past_key_values[0][0].shape[2]
            new_pkv = ((
                torch.zeros(batch_size, num_heads, past_len + 1, head_dim, device=input_ids.device),
                torch.zeros(batch_size, num_heads, past_len + 1, head_dim, device=input_ids.device)
            ),)
            
        return DummyOutput(logits=logits, past_key_values=new_pkv)

class DummyTokenizer:
    eos_token_id = 99
    def encode(self, text, *args, **kwargs):
        return [1, 2, 3] * (len(text) // 5)
    def decode(self, ids, *args, **kwargs):
        return " ".join(str(i) for i in ids)

@pytest.fixture
def dummy_engine():
    config = ModelConfig(
        model_name_or_path="dummy",
        max_model_len=1024
    )
    
    with patch("winllm.inference.engine.ModelLoader") as MockLoader:
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.load.return_value = (DummyModel(), DummyTokenizer())
        mock_loader_instance.get_kv_cache_params.return_value = {"num_layers": 1, "num_kv_heads": 2, "head_dim": 4}
        
        engine = InferenceEngine(config)
        engine.load_model()
        yield engine

def test_decode_batch_with_varying_lengths(dummy_engine):
    """Test that _decode_batch handles varying sequence lengths seamlessly via left padding."""
    device = dummy_engine._runtime.resolve_device()
    
    req1 = GenerationRequest("prompt 1 which is short", sampling_params=SamplingParams())
    req2 = GenerationRequest("prompt 2 which is much much longer and more complex", sampling_params=SamplingParams())
    
    # Manually configure their state to simulate an ongoing generation
    req1.prompt_token_ids = [1, 2, 3] # length 3
    req1.output_token_ids = [10, 11] # total previous length = 3+2-1 = 4
    
    req2.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7] # length 7
    req2.output_token_ids = [20, 21, 22] # total previous length = 7+3-1 = 9
    
    # Create fake KV cache tensors matching those lengths
    num_heads, head_dim = 2, 4
    req1._past_key_values = ((
        torch.zeros(1, num_heads, 4, head_dim, device=device),
        torch.zeros(1, num_heads, 4, head_dim, device=device)
    ),)
    req2._past_key_values = ((
        torch.zeros(1, num_heads, 9, head_dim, device=device),
        torch.zeros(1, num_heads, 9, head_dim, device=device)
    ),)
    
    # Engine method directly
    # Call the batch method
    dummy_engine._decode_runner.decode_batch([req1, req2], device=device)
    
    # Afterwards, the new output sequences should be longer by 1
    assert len(req1.output_token_ids) == 3
    assert len(req2.output_token_ids) == 4
    
    # And their _past_key_values should correctly reflect unpadded new sizes
    # new seq len for req1 = 3 + 2 = 5
    # new seq len for req2 = 7 + 3 = 10
    k1_shape = req1._past_key_values[0][0].shape
    k2_shape = req2._past_key_values[0][0].shape
    
    assert k1_shape == (1, num_heads, 5, head_dim)
    assert k2_shape == (1, num_heads, 10, head_dim)


def _seed_request(prompt_ids, output_ids, device, num_heads=2, head_dim=4):
    req = GenerationRequest("p", sampling_params=SamplingParams())
    req.prompt_token_ids = list(prompt_ids)
    req.output_token_ids = list(output_ids)
    cache_len = len(prompt_ids) + len(output_ids) - 1
    req._past_key_values = ((
        torch.zeros(1, num_heads, cache_len, head_dim, device=device),
        torch.zeros(1, num_heads, cache_len, head_dim, device=device),
    ),)
    return req


def test_persistent_cache_reused_across_stable_steps(dummy_engine):
    """A stable batch must repack the KV cache only once, not every step."""
    device = dummy_engine._runtime.resolve_device()
    req1 = _seed_request([1, 2, 3], [10, 11], device)
    req2 = _seed_request([1, 2, 3, 4, 5, 6, 7], [20, 21, 22], device)

    cache = dummy_engine._batch_cache
    with patch.object(cache, "rebuild", wraps=cache.rebuild) as spy:
        dummy_engine._decode_runner.decode_batch([req1, req2], device)
        padded_after_first = cache.kv[0][0].shape[2]
        dummy_engine._decode_runner.decode_batch([req1, req2], device)

        # Rebuilt once (initial pack); the second stable step reused the cache.
        assert spy.call_count == 1

    # The persistent cache grew by exactly one position on the reused step.
    assert cache.kv[0][0].shape[2] == padded_after_first + 1
    # Two tokens were generated for each request.
    assert len(req1.output_token_ids) == 4
    assert len(req2.output_token_ids) == 5
    # Per-request caches remain correct (exposed as views).
    assert req1._past_key_values[0][0].shape == (1, 2, 6, 4)   # 3 + 3
    assert req2._past_key_values[0][0].shape == (1, 2, 11, 4)  # 7 + 4


def test_persistent_cache_repacks_incrementally_on_membership_change(dummy_engine):
    """A membership change repacks via the incremental path, not a full rebuild."""
    device = dummy_engine._runtime.resolve_device()
    req1 = _seed_request([1, 2, 3], [10, 11], device)
    req2 = _seed_request([1, 2, 3, 4, 5, 6, 7], [20, 21, 22], device)
    req3 = _seed_request([1, 2, 3, 4], [30, 31], device)

    cache = dummy_engine._batch_cache
    with patch.object(cache, "rebuild", wraps=cache.rebuild) as spy:
        dummy_engine._decode_runner.decode_batch([req1, req2], device)   # full rebuild (fresh)
        dummy_engine._decode_runner.decode_batch([req1, req2], device)   # reused
        dummy_engine._decode_runner.decode_batch([req1, req3], device)   # incremental (req1 survives)
        assert spy.call_count == 1

    assert cache.matches([req1, req3])
    assert not cache.matches([req1, req2])


def _colored_request(length, fill, num_heads=2, head_dim=4, num_layers=2):
    """Request whose KV cache is a recognizable constant, for content checks."""
    req = GenerationRequest("p", sampling_params=SamplingParams())
    req.prompt_token_ids = list(range(length))
    req.output_token_ids = [1]
    req._past_key_values = tuple(
        (
            torch.full((1, num_heads, length, head_dim), float(fill)),
            torch.full((1, num_heads, length, head_dim), float(fill) + 0.5),
        )
        for _ in range(num_layers)
    )
    return req


class TestPersistentBatchCacheRefresh:
    def _fresh_cache(self):
        from winllm.inference.buffers import PersistentBatchCache
        return PersistentBatchCache()

    def test_refresh_is_noop_when_membership_unchanged(self):
        cache = self._fresh_cache()
        a, b = _colored_request(3, 1), _colored_request(5, 2)
        cache.refresh([a, b], torch.device("cpu"))
        kv_before = cache.kv
        cache.refresh([a, b], torch.device("cpu"))
        assert cache.kv is kv_before  # same tensors, no repack

    def test_removal_preserves_survivor_content_and_trims_padding(self):
        cache = self._fresh_cache()
        a, b, c = _colored_request(3, 1), _colored_request(7, 2), _colored_request(4, 3)
        device = torch.device("cpu")
        cache.refresh([a, b, c], device)
        cache.expose_views([a, b, c], [3, 7, 4])

        cache.refresh([a, c], device)  # b (the longest) leaves

        k = cache.kv[0][0]
        assert k.shape[0] == 2
        assert k.shape[2] == 4  # padded length shrank to the survivors' max
        assert torch.all(k[0, :, -3:, :] == 1.0)  # a's content, right-aligned
        assert torch.all(k[1, :, -4:, :] == 3.0)  # c's content
        assert torch.all(k[0, :, :-3, :] == 0.0)  # a's left padding stays zero

    def test_addition_copies_only_new_rows_and_keeps_order(self):
        cache = self._fresh_cache()
        a, b = _colored_request(3, 1), _colored_request(5, 2)
        device = torch.device("cpu")
        cache.refresh([a, b], device)
        cache.expose_views([a, b], [3, 5])

        d = _colored_request(6, 4)
        cache.refresh([b, d, a], device)  # reordered survivors + one new request

        k, v = cache.kv[0]
        assert k.shape == (3, 2, 6, 4)
        assert torch.all(k[0, :, -5:, :] == 2.0)  # b in requested position 0
        assert torch.all(k[1, :, -6:, :] == 4.0)  # d's prefill cache
        assert torch.all(k[2, :, -3:, :] == 1.0)  # a in requested position 2
        assert torch.all(v[1, :, -6:, :] == 4.5)  # values distinct from keys

    def test_all_new_requests_falls_back_to_rebuild(self):
        cache = self._fresh_cache()
        a, b = _colored_request(3, 1), _colored_request(5, 2)
        device = torch.device("cpu")
        cache.refresh([a, b], device)
        c, d = _colored_request(2, 5), _colored_request(4, 6)
        cache.refresh([c, d], device)  # no survivors
        k = cache.kv[0][0]
        assert k.shape == (2, 2, 4, 4)
        assert torch.all(k[0, :, -2:, :] == 5.0)
        assert torch.all(k[1, :, -4:, :] == 6.0)
