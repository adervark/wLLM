"""The shared single-token eager decode step.

Every non-graph decode path — batched serve decode, blocking generation,
the suffix-speculation fallback, and the load-time validation probes —
performs the same forward: feed the last committed token with the KV cache,
take the final position's logits. Keeping it in one place means changes to
how the eager path invokes the model (cache_position, attention_mask,
buffer reuse) apply everywhere at once, including the probes that validate
other paths against it.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .buffers import DecodeInputBuffer


def eager_decode_step(
    model,
    token: int,
    past_key_values,
    device: torch.device,
    input_buffer: Optional["DecodeInputBuffer"] = None,
):
    """One eager decode forward for ``token``; returns (logits, cache).

    ``input_buffer`` (a DecodeInputBuffer) avoids a per-step tensor
    allocation on hot paths; cold paths (validation probes) may omit it.
    """
    if input_buffer is not None:
        input_ids = input_buffer.fill(token, device)
    else:
        input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits[:, -1, :], outputs.past_key_values
