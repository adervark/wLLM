"""Token sampling: pure logit transforms (ops), composable processors
implementing ``winllm.core.LogitsProcessor`` (OCP — add a new transform
without touching the sampler), and the sampling entry point.
"""

from .ops import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)
from .processors import (
    DEFAULT_PIPELINE,
    RepetitionPenaltyProcessor,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
)
from .sampler import sample_token

__all__ = [
    "apply_repetition_penalty",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "RepetitionPenaltyProcessor",
    "TemperatureProcessor",
    "TopKProcessor",
    "TopPProcessor",
    "DEFAULT_PIPELINE",
    "sample_token",
]
