"""The inference layer, decomposed by responsibility (SRP):

  - runtime.py:     shared loaded-model state (ModelRuntime)
  - buffers.py:     pre-allocated CUDA buffers for the decode hot path
  - streaming.py:   token emission to stream callbacks
  - prefill.py:     prompt processing (single, chunked, batched)
  - decode.py:      token-by-token generation (single, batched, speculative)
  - generation.py:  the blocking single-request generate loop
  - speculative.py: draft-model speculative decoding
  - engine.py:      the InferenceEngine facade that wires it all together
"""

from .engine import InferenceEngine
from .runtime import ModelRuntime
from .speculative import SpeculativeEngine

__all__ = ["InferenceEngine", "ModelRuntime", "SpeculativeEngine"]
