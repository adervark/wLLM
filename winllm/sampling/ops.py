"""Pure logit-transform operations used by the sampling pipeline.

These run on every decode step, so the rule is: decide everything decidable
in Python on the parameter lists BEFORE creating any tensor. On Windows
(WDDM) each ``torch.tensor(..., device="cuda")`` and each GPU boolean check
(``if torch.all(...)``) is a stream sync costing up to ~1ms — profiled at
55% of chat-path decode wall time before this rewrite. Uniform parameters
(always true for the single-request chat path) take scalar code paths that
transfer nothing to the device.

Callers pass logits they own (the sampler clones first); ops may mutate
in place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _uniform(values: list):
    """The single shared value if all entries are equal, else None."""
    first = values[0]
    for v in values[1:]:
        if v != first:
            return None
    return first


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int] | list[list[int]],
    penalty: float | list[float],
) -> torch.Tensor:
    """Apply repetition penalty to logits for already-generated tokens."""
    batch_size, _ = logits.shape
    penalties = [penalty] * batch_size if isinstance(penalty, (int, float)) else penalty

    if not generated_ids or all(p == 1.0 for p in penalties):
        return logits

    if not isinstance(generated_ids[0], list):
        gen_ids_list = [generated_ids] * batch_size
    else:
        gen_ids_list = generated_ids

    if not any(gen_ids_list):
        return logits

    device = logits.device
    for i in range(batch_size):
        ids, p = gen_ids_list[i], penalties[i]
        if not ids or p == 1.0:
            continue
        # One small H2D transfer of the (deduped) ids per penalized row,
        # instead of building a vocab-sized mask. index_copy_ requires
        # unique indices; sorted() keeps the transfer deterministic.
        idx = torch.tensor(sorted(set(ids)), dtype=torch.long).to(
            device, non_blocking=True
        )
        vals = logits[i].index_select(0, idx)
        penalized = torch.where(vals > 0, vals / p, vals * p)
        logits[i].index_copy_(0, idx, penalized)
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float | list[float]) -> torch.Tensor:
    """Scale logits by temperature."""
    batch_size = logits.shape[0]
    temps = [temperature] * batch_size if isinstance(temperature, (int, float)) else temperature

    # Replace 0s with 1s to prevent division by zero; greedy handled downstream
    safe_temps = [t if t > 0 else 1.0 for t in temps]

    t = _uniform(safe_temps)
    if t is not None:
        if t != 1.0:
            logits.div_(t)
        return logits

    temp_tensor = (
        torch.tensor(safe_temps, dtype=logits.dtype)
        .to(logits.device, non_blocking=True)
        .unsqueeze(1)
    )
    logits.div_(temp_tensor)
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int | list[int]) -> torch.Tensor:
    """Keep only top-k logits, set rest to -inf."""
    batch_size, vocab_size = logits.shape
    ks = [top_k] * batch_size if isinstance(top_k, int) else top_k

    # For any k <= 0 or k >= vocab_size, we keep all (set k=vocab_size)
    safe_ks = [k if 0 < k < vocab_size else vocab_size for k in ks]

    if all(k == vocab_size for k in safe_ks):
        return logits

    k = _uniform(safe_ks)
    if k is not None:
        # Only the k-th value is needed as the row threshold.
        # torch.topk is O(vocab) vs torch.sort's O(vocab log vocab).
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        thresholds = topk_vals[:, -1:]
        logits.masked_fill_(logits < thresholds, float("-inf"))
        return logits

    max_k = max(safe_ks)
    k_tensor = (
        torch.tensor(safe_ks, dtype=torch.long)
        .to(logits.device, non_blocking=True)
        .unsqueeze(1)
    )
    topk_vals, _ = torch.topk(logits, max_k, dim=-1)
    indices = (k_tensor - 1).clamp(min=0, max=max_k - 1)
    thresholds = topk_vals.gather(-1, indices)

    # Mask anything strictly less than the threshold
    logits.masked_fill_(logits < thresholds, float("-inf"))
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float | list[float]) -> torch.Tensor:
    """Keep tokens with cumulative probability <= top_p (nucleus sampling)."""
    batch_size = logits.shape[0]
    ps = [top_p] * batch_size if isinstance(top_p, (int, float)) else top_p

    if all(p >= 1.0 for p in ps):
        return logits

    # Uniform p stays a Python float and broadcasts; mixed p needs a column
    p = _uniform(ps)
    if p is None:
        p = (
            torch.tensor(ps, dtype=logits.dtype)
            .to(logits.device, non_blocking=True)
            .unsqueeze(1)
        )

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs_sorted = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs_sorted, dim=-1)

    # Shift cumulative probabilities to the right to keep the first token that exceeds top_p
    # The mask checks if cumulative probability BEFORE the token exceeds top_p
    # So we compute cumulative_probs - current_prob
    shifted_probs = cumulative_probs - probs_sorted
    sorted_mask = shifted_probs >= p

    sorted_logits.masked_fill_(sorted_mask, float("-inf"))

    # Scatter back to the original indices in-place
    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits
