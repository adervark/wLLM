"""Shared state for a loaded model.

ModelRuntime is the single place the loaded model, tokenizer, device,
and KV cache manager live. The prefill/decode/generation components all
hold a reference to one runtime instead of reaching into the engine
(Interface Segregation: they get exactly the state they need, nothing
about scheduling, serving, or lifecycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from ..kvcache import KVCacheManager
    from .speculative import SpeculativeEngine


class ModelRuntime:
    """Holds the artifacts of a loaded model shared by inference components."""

    def __init__(self):
        self.model: Optional["PreTrainedModel"] = None
        self.tokenizer: Optional["PreTrainedTokenizerBase"] = None
        self.device: Optional[torch.device] = None
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self.speculative_engine: Optional["SpeculativeEngine"] = None
        self.graph_decoder = None  # CUDAGraphDecoder when enabled and validated

    def resolve_device(self) -> torch.device:
        """Figure out which device the model is actually on."""
        return self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(self, text: str) -> list[int]:
        """Tokenize input text."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def clear(self) -> None:
        """Drop all references to the loaded model."""
        self.model = None
        self.tokenizer = None
        self.device = None
        self.speculative_engine = None
        if self.graph_decoder is not None:
            self.graph_decoder.release()
            self.graph_decoder = None
