import pytest
import torch
import time
from unittest.mock import MagicMock, patch

from winllm.engine import InferenceEngine
from winllm.config import ModelConfig, KVCacheConfig, SamplingParams
from winllm.types import GenerationRequest, RequestStatus
from winllm.model_loader import ModelLoader

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
    
    with patch("winllm.engine.ModelLoader") as MockLoader:
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.load.return_value = (DummyModel(), DummyTokenizer())
        mock_loader_instance.get_kv_cache_params.return_value = {"num_layers": 1, "num_kv_heads": 2, "head_dim": 4}
        
        engine = InferenceEngine(config)
        engine.load_model()
        yield engine

def test_decode_batch_with_varying_lengths(dummy_engine):
    """Test that _decode_batch handles varying sequence lengths seamlessly via left padding."""
    device = dummy_engine._resolve_device()
    
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
    dummy_engine._decode_batch([req1, req2], device=device)
    
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
