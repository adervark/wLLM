"""Pre-allocated CUDA buffers for the decode hot path."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def trim_cache(past_key_values, length: int):
    """Trim a KV cache to ``length`` positions, whatever its representation.

    transformers Cache objects (DynamicCache, hybrid caches) expose
    ``crop``; legacy tuple-of-(k, v) caches are sliced along the seq dim.
    Used by the speculative engines to drop rejected draft positions.
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(length)
        return past_key_values
    trimmed = []
    for k, v in past_key_values:
        if k.shape[2] > length:
            trimmed.append((k[:, :, :length, :], v[:, :, :length, :]))
        else:
            trimmed.append((k, v))
    return tuple(trimmed)


class DecodeInputBuffer:
    """A static (1, 1) token buffer reused every decode step.

    Avoids a fresh CUDA allocation per generated token.
    """

    def __init__(self):
        self._buffer: Optional[torch.Tensor] = None

    def allocate(self, device: torch.device) -> None:
        self._buffer = torch.zeros((1, 1), dtype=torch.long, device=device)

    def release(self) -> None:
        self._buffer = None

    def fill(self, token_id: int, device: torch.device) -> torch.Tensor:
        """Return a (1, 1) tensor holding ``token_id`` on ``device``.

        Reuses the static buffer when possible, otherwise allocates fresh
        (e.g. before load, or if the device changed).
        """
        if self._buffer is not None and self._buffer.device == device:
            self._buffer[0, 0] = token_id
            return self._buffer
        return torch.tensor([[token_id]], dtype=torch.long, device=device)


class PersistentBatchCache:
    """A batched KV cache that survives across decode steps.

    The previous design rebuilt the batched ``past_key_values`` every decode
    step — copying every request's entire history into a shared buffer, and
    cloning each request's full history back out afterwards (~2× the whole
    batch's KV copied *per generated token*, growing with context length).

    Instead we hold the batched cache and feed it straight back each step while
    the batch membership is unchanged, so the only per-step KV traffic is the
    model's own append. The expensive repack happens only when the resident set
    changes (a request finishes or a new one is admitted). Per-request caches
    are exposed as zero-copy *views* into the batched tensor, so callers that
    read ``req._past_key_values`` (prefix promotion, single-request decode) still
    see a correct, current cache without any clone.

    Cache layout is left-padded ``[batch, kv_heads, seq, head_dim]`` per layer:
    each row's real tokens are right-aligned, padding on the left is masked out.
    """

    def __init__(self):
        self._batch_kv: Optional[tuple] = None
        self._resident_ids: list[int] = []

    @property
    def kv(self) -> Optional[tuple]:
        return self._batch_kv

    def update(self, batch_kv: tuple) -> None:
        """Adopt the cache the model returned after a forward pass."""
        self._batch_kv = batch_kv

    def release(self) -> None:
        """Drop the batched cache (frees GPU memory once views are gone)."""
        self._batch_kv = None
        self._resident_ids = []

    def matches(self, requests: list) -> bool:
        """True if the batched cache already holds exactly ``requests`` in order."""
        return self._batch_kv is not None and [id(r) for r in requests] == self._resident_ids

    def refresh(self, requests: list, device: torch.device) -> None:
        """Make the batched cache hold exactly ``requests`` (in order).

        No-op when membership is unchanged. On a membership change, rows of
        carried-over requests are moved with one fused gather+scatter per
        layer tensor instead of a per-request Python copy loop; only newly
        admitted requests pay a per-row copy of their prefill cache.
        """
        if self.matches(requests):
            return

        old_index = {rid: i for i, rid in enumerate(self._resident_ids)}
        has_survivors = self._batch_kv is not None and any(
            id(r) in old_index for r in requests
        )
        if not has_survivors:
            self.rebuild(requests, device)
            return

        lengths = [r._past_key_values[0][0].shape[2] for r in requests]
        new_max = max(lengths)
        batch_size = len(requests)

        surv_dest = [i for i, r in enumerate(requests) if id(r) in old_index]
        surv_src = [old_index[id(requests[i])] for i in surv_dest]
        new_rows = [i for i, r in enumerate(requests) if id(r) not in old_index]

        # Right-aligned rows: copying the longest survivor's span covers all
        # survivors; shorter rows drag along old left-padding zeros, which the
        # attention mask ignores.
        surv_span = min(max(lengths[i] for i in surv_dest), new_max)
        src_idx = torch.tensor(surv_src, dtype=torch.long, device=device)
        dest_idx = torch.tensor(surv_dest, dtype=torch.long, device=device)

        sample_k = self._batch_kv[0][0]
        num_kv_heads, head_dim, kv_dtype = sample_k.shape[1], sample_k.shape[3], sample_k.dtype

        batched = []
        for k_buf, v_buf in self._batch_kv:
            nk = torch.zeros(batch_size, num_kv_heads, new_max, head_dim, dtype=kv_dtype, device=device)
            nv = torch.zeros_like(nk)
            nk[dest_idx, :, -surv_span:, :] = k_buf[src_idx][:, :, -surv_span:, :]
            nv[dest_idx, :, -surv_span:, :] = v_buf[src_idx][:, :, -surv_span:, :]
            batched.append((nk, nv))

        for i in new_rows:
            req = requests[i]
            length = lengths[i]
            for layer, (nk, nv) in enumerate(batched):
                k, v = req._past_key_values[layer]
                nk[i:i + 1, :, -length:, :] = k
                nv[i:i + 1, :, -length:, :] = v

        self._batch_kv = tuple(batched)
        self._resident_ids = [id(r) for r in requests]

    def rebuild(self, requests: list, device: torch.device) -> None:
        """Repack the per-request caches into one left-padded batched cache.

        Reads each request's current ``_past_key_values`` (a prefill cache for a
        newly admitted request, or a view into the prior batched cache for a
        carried-over one). Only invoked on a membership change.
        """
        seq_lengths = [r._past_key_values[0][0].shape[2] for r in requests]
        max_len = max(seq_lengths)
        sample_k = requests[0]._past_key_values[0][0]
        num_layers = len(requests[0]._past_key_values)
        num_kv_heads = sample_k.shape[1]
        head_dim = sample_k.shape[3]
        kv_dtype = sample_k.dtype
        batch_size = len(requests)

        batched = []
        for layer in range(num_layers):
            k_buf = torch.zeros(batch_size, num_kv_heads, max_len, head_dim, dtype=kv_dtype, device=device)
            v_buf = torch.zeros(batch_size, num_kv_heads, max_len, head_dim, dtype=kv_dtype, device=device)
            for i, req in enumerate(requests):
                k, v = req._past_key_values[layer]
                length = seq_lengths[i]
                k_buf[i:i + 1, :, -length:, :] = k
                v_buf[i:i + 1, :, -length:, :] = v
            batched.append((k_buf, v_buf))

        self._batch_kv = tuple(batched)
        self._resident_ids = [id(r) for r in requests]

    def expose_views(self, requests: list, cache_lengths: list[int]) -> None:
        """Point each request's ``_past_key_values`` at its row of the batch.

        These are views (no copy): ``cache_lengths[i]`` right-aligned positions
        of row ``i``. Valid as a read-only cache for the next step or for
        per-request consumers.
        """
        num_layers = len(self._batch_kv)
        for i, req in enumerate(requests):
            length = cache_lengths[i]
            req._past_key_values = tuple(
                (
                    self._batch_kv[layer][0][i:i + 1, :, -length:, :],
                    self._batch_kv[layer][1][i:i + 1, :, -length:, :],
                )
                for layer in range(num_layers)
            )
