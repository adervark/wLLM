"""Token sampling strategies for text generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import SamplingParams


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int] | list[list[int]],
    penalty: float | list[float],
) -> torch.Tensor:
    """Apply repetition penalty to logits for already-generated tokens."""
    batch_size, vocab_size = logits.shape
    
    # Handle single vs batch inputs
    penalties = [penalty] * batch_size if isinstance(penalty, (int, float)) else penalty
    if not generated_ids:
        gen_ids_list = [[] for _ in range(batch_size)]
    elif not isinstance(generated_ids[0], list):
        gen_ids_list = [generated_ids] * batch_size
    else:
        gen_ids_list = generated_ids
        
    device = logits.device
    penalty_tensor = torch.tensor(penalties, dtype=logits.dtype, device=device).unsqueeze(1)
    
    # Fast path: if all penalties are 1.0 or no generated ids, skip
    if torch.all(penalty_tensor == 1.0) or not any(gen_ids_list):
        return logits
        
    # Build mask of generated tokens
    mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    for i, ids in enumerate(gen_ids_list):
        if ids and penalties[i] != 1.0:
             # Convert list to tensor for indexing
             idx_tensor = torch.tensor(list(set(ids)), dtype=torch.long, device=device)
             mask[i].scatter_(0, idx_tensor, True)
             
    # Apply penalty
    # Apply penalty in-place (no logits.clone needed)
    penalized = torch.where(logits > 0, logits / penalty_tensor, logits * penalty_tensor)
    logits = torch.where(mask, penalized, logits)
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float | list[float]) -> torch.Tensor:
    """Scale logits by temperature."""
    batch_size = logits.shape[0]
    temps = [temperature] * batch_size if isinstance(temperature, (int, float)) else temperature
    
    # Replace 0s with 1s to prevent division by zero; greedy handled downstream
    safe_temps = [t if t > 0 else 1.0 for t in temps]
    temp_tensor = torch.tensor(safe_temps, dtype=logits.dtype, device=logits.device).unsqueeze(1)
    
    logits.div_(temp_tensor)
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int | list[int]) -> torch.Tensor:
    """Keep only top-k logits, set rest to -inf."""
    batch_size, vocab_size = logits.shape
    ks = [top_k] * batch_size if isinstance(top_k, int) else top_k
    
    # For any k <= 0 or k >= vocab_size, we keep all (set k=vocab_size)
    safe_ks = [k if 0 < k < vocab_size else vocab_size for k in ks]
    k_tensor = torch.tensor(safe_ks, dtype=torch.long, device=logits.device).unsqueeze(1)
    
    if torch.all(k_tensor == vocab_size):
        return logits
        
    # Get the value of the k-th largest logit per row
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    indices = (k_tensor - 1).clamp(min=0, max=vocab_size - 1)
    thresholds = sorted_logits.gather(-1, indices)
    
    # Mask anything strictly less than the threshold
    return torch.where(logits < thresholds, float('-inf'), logits)


def apply_top_p(logits: torch.Tensor, top_p: float | list[float]) -> torch.Tensor:
    """Keep tokens with cumulative probability <= top_p (nucleus sampling)."""
    batch_size = logits.shape[0]
    ps = [top_p] * batch_size if isinstance(top_p, (int, float)) else top_p
    
    p_tensor = torch.tensor(ps, dtype=logits.dtype, device=logits.device).unsqueeze(1)
    
    if torch.all(p_tensor >= 1.0):
        return logits
        
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Shift cumulative probabilities to the right to keep the first token that exceeds top_p
    # The mask checks if cumulative probability BEFORE the token exceeds top_p
    # So we compute cumulative_probs - current_prob
    shifted_probs = cumulative_probs - F.softmax(sorted_logits, dim=-1)
    sorted_mask = shifted_probs >= p_tensor
    
    sorted_logits.masked_fill_(sorted_mask, float('-inf'))
    
    # Scatter back to the original indices in-place
    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits


def sample_token(
    logits: torch.Tensor, 
    params: SamplingParams | list[SamplingParams],
    generated_ids: list[int] | list[list[int]] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full batched sampling pipeline: penalties → temperature → top-k → top-p → sample."""
    batch_size = logits.shape[0]
    params_list = [params] * batch_size if isinstance(params, SamplingParams) else params
    
    penalties = [p.repetition_penalty for p in params_list]
    temperatures = [p.temperature for p in params_list]
    top_ks = [p.top_k for p in params_list]
    top_ps = [p.top_p for p in params_list]
    
    # Fast path check for pure greedy (all temps == 0) and no penalties needed
    all_greedy = all(t == 0 for t in temperatures)
    if all_greedy and all(p == 1.0 for p in penalties):
        return torch.argmax(logits, dim=-1)
    
    # 1. Pipeline modifications
    logits = apply_repetition_penalty(logits, generated_ids, penalties)
    logits = apply_temperature(logits, temperatures)
    logits = apply_top_k(logits, top_ks)
    logits = apply_top_p(logits, top_ps)
    
    # 2. Multinomial sampling for non-greedy rows
    probs = F.softmax(logits, dim=-1)
    # Multinomial needs valid probabilities. Replace NaNs if all valid probs were masked:
    if torch.isnan(probs).any():
         probs = torch.nan_to_num(probs, nan=0.0)
         
    try:
        sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    except RuntimeError:
        # Fallback if multinomial fails due to 0-probability across all tokens
        sampled = torch.argmax(logits, dim=-1)
        
    # 3. Handle greedy overrides where temp == 0
    if not all_greedy:
        greedy_mask = torch.tensor([t == 0 for t in temperatures], dtype=torch.bool, device=logits.device)
        if greedy_mask.any():
            greedy_samples = torch.argmax(logits, dim=-1)
            sampled = torch.where(greedy_mask, greedy_samples, sampled)
            
    return sampled
