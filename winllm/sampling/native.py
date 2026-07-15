"""Optional fused CUDA sampling kernel (native/sampling/).

One kernel launch replaces the ~15-launch torch pipeline in the per-token
path — the launches, not the math, are the cost on Windows/WDDM (see
documentation/PERFORMANCE_NOTES.md). Importing ``winllm_sampling`` is
optional exactly like ``winllm_suffix``: without it (or with
``WINLLM_PURE_PYTHON=1``) every call falls through to the torch pipeline.

Eligibility is decided in ``sampler.sample_token``; this module only
handles the mechanics. Two hard rules enforced here:

- The kernel draws from its own Philox stream (process-random seed +
  per-call offset), so it can never serve seeded requests — their
  reproducibility contract is one ``torch`` draw per committed token.
  The sampler routes ``generator is not None`` to the torch path.
- Returned tensors are freshly allocated (callers may hold them); only the
  internal scratch buffer is reused, which is safe because all winllm work
  on a device shares one stream and reuse is therefore stream-ordered.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading

import torch

logger = logging.getLogger(__name__)

_module = None
if not os.environ.get("WINLLM_PURE_PYTHON"):
    try:
        import winllm_sampling as _module  # type: ignore[no-redef]
    except ImportError:
        _module = None

_DTYPE_CODES = {torch.float16: 0, torch.float32: 1, torch.bfloat16: 2}


class FusedSampler:
    """Stateful wrapper: RNG stream, scratch-buffer cache, failure latch."""

    def __init__(self) -> None:
        self.enabled = _module is not None
        self._seed = secrets.randbits(63)
        self._offset = 0
        self._scratch: dict[tuple, torch.Tensor] = {}
        self._lock = threading.Lock()

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        penalty: float,
        generated_ids: list[int] | None,
    ) -> torch.Tensor | None:
        """Fused sample over [batch, vocab] CUDA logits; None means
        "not handled here, take the torch path". Never mutates logits."""
        if not self.enabled or not logits.is_cuda or logits.dim() != 2:
            return None
        code = _DTYPE_CODES.get(logits.dtype)
        if code is None or logits.stride(-1) != 1:
            return None
        batch, vocab = logits.shape

        ids_tensor = None  # kept alive past the (async) launch on purpose
        ids_ptr, n_ids = 0, 0
        if penalty != 1.0 and generated_ids:
            if batch != 1:
                # Per-row penalty state (the sequential verify path) only
                # ever samples one row at a time; a multi-row call with
                # penalties belongs to the torch path.
                return None
            # Same single small H2D transfer the torch path pays; the kernel
            # requires unique ids so each logit is penalized exactly once.
            ids_tensor = torch.tensor(
                sorted(set(generated_ids)), dtype=torch.int32
            ).to(logits.device, non_blocking=True)
            ids_ptr, n_ids = ids_tensor.data_ptr(), ids_tensor.numel()

        out = torch.empty(batch, dtype=torch.int64, device=logits.device)
        key = (logits.device.index, batch, vocab)
        with self._lock:
            scratch = self._scratch.get(key)
            if scratch is None:
                scratch = torch.empty(
                    batch * 2 * vocab, dtype=torch.float32, device=logits.device
                )
                self._scratch[key] = scratch
            self._offset += 1
            offset = self._offset

        try:
            _module.sample(
                logits.data_ptr(), code, logits.stride(0), batch, vocab,
                ids_ptr, n_ids, float(penalty),
                float(temperature), int(top_k), float(top_p),
                scratch.data_ptr(), out.data_ptr(),
                self._seed, offset,
                torch.cuda.current_stream(logits.device).cuda_stream,
            )
        except Exception:
            # Fail once, loudly, then stay on the torch path for the rest of
            # the process — a broken accelerator must not take decode down.
            logger.warning(
                "winllm_sampling kernel failed; native sampling disabled",
                exc_info=True,
            )
            self.enabled = False
            return None
        return out

    def apply_bitmask(self, logits_row: torch.Tensor, packed: torch.Tensor) -> bool:
        """Mask ``logits_row`` (shape (vocab,)) in place from an xgrammar
        packed bitmask (int32 words, bit i%32 of word i//32, 1 = allowed),
        in one kernel instead of the unpack-to-bool torch chain. Returns
        False when the torch fallback should run instead."""
        if not self.enabled or not logits_row.is_cuda or logits_row.dim() != 1:
            return False
        code = _DTYPE_CODES.get(logits_row.dtype)
        if code is None or not logits_row.is_contiguous():
            return False
        vocab = logits_row.shape[0]
        words = packed.reshape(-1).to(logits_row.device, non_blocking=True)
        if words.dtype != torch.int32 or words.numel() * 32 < vocab:
            return False
        try:
            _module.apply_bitmask(
                logits_row.data_ptr(), code, vocab, 1, vocab,
                words.data_ptr(), words.numel(),
                torch.cuda.current_stream(logits_row.device).cuda_stream,
            )
        except Exception:
            logger.warning(
                "winllm_sampling bitmask kernel failed; native sampling disabled",
                exc_info=True,
            )
            self.enabled = False
            return False
        return True


#: Process-wide instance used by sample_token.
fused_sampler = FusedSampler()
