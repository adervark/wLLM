"""Speculative decoding engine for faster token generation.

Speculative decoding uses a small, fast "draft" model to propose multiple
tokens at once, then verifies them all with the large "target" model in
a single forward pass. When the draft model guesses correctly (which is
common for simple/predictable tokens), this can generate multiple tokens
per forward pass of the target model.

How it works:
  1. Draft model generates N candidate tokens autoregressively.
  2. Target model processes all N candidates in ONE forward pass.
  3. We compare: accept matching tokens, reject at the first mismatch.
  4. If all N matched, we bonus-sample one extra token from the target.
"""

from __future__ import annotations

import torch
from typing import Optional, List
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .types import GenerationRequest
from .sampler import sample_token


class SpeculativeEngine:
    """Implements speculative decoding logic.

    Args:
        target_model: The large, accurate model used for verification.
        draft_model: The small, fast model used for proposing tokens.
        tokenizer: Shared tokenizer (must be compatible with both models).
        num_speculative_tokens: How many tokens to draft per step.
    """

    def __init__(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_speculative_tokens: int = 4
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self.device = next(target_model.parameters()).device

    @torch.inference_mode()
    def step(self, request: GenerationRequest) -> bool:
        """Perform one speculative decoding step for a single request.

        Returns True if the request should continue generating, False if it hit EOS.
        """
        # --- Phase 1: Draft model proposes N tokens ---
        draft_tokens = self._draft_proposals(request)

        # --- Phase 2: Target model verifies all proposals at once ---
        target_logits = self._verify_proposals(request, draft_tokens)

        # --- Phase 3: Accept or reject each proposed token ---
        return self._accept_or_reject(request, draft_tokens, target_logits)

    def _draft_proposals(self, request: GenerationRequest) -> list[int]:
        """Use the draft model to quickly generate candidate tokens."""
        draft_tokens = []
        last_token = request.output_token_ids[-1]
        proposal_input_ids = torch.tensor([[last_token]], device=self.device)
        temp_past_draft = request._draft_past_key_values

        for _ in range(self.num_speculative_tokens):
            outputs = self.draft_model(
                proposal_input_ids, past_key_values=temp_past_draft, use_cache=True
            )
            temp_past_draft = outputs.past_key_values

            next_logits = outputs.logits[:, -1, :]
            next_token_id = sample_token(
                next_logits, request.sampling_params,
                request.output_token_ids + draft_tokens
            )
            next_token = next_token_id.item()
            draft_tokens.append(next_token)
            proposal_input_ids = next_token_id.unsqueeze(0)

            # Stop drafting if we hit EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        return draft_tokens

    def _verify_proposals(self, request: GenerationRequest, draft_tokens: list[int]) -> torch.Tensor:
        """Run the target model on all proposed tokens in one forward pass."""
        last_token = request.output_token_ids[-1]
        verify_input_ids = torch.tensor([[last_token] + draft_tokens], device=self.device)

        target_outputs = self.target_model(
            verify_input_ids,
            past_key_values=request._past_key_values,
            use_cache=True
        )

        request._past_key_values = target_outputs.past_key_values
        # Shape: (num_draft_tokens + 1, vocab_size)
        return target_outputs.logits[0, :, :]

    def _accept_or_reject(
        self, request: GenerationRequest,
        draft_tokens: list[int], target_logits: torch.Tensor
    ) -> bool:
        """Compare draft vs. target predictions, accept matches, reject at first mismatch."""
        num_verified = 0
        for i in range(len(draft_tokens)):
            # Sample what the target model would have chosen at position i
            target_token_id = sample_token(
                target_logits[i:i+1, :], request.sampling_params, request.output_token_ids
            )
            target_token = target_token_id.item()
            num_verified += 1

            if target_token == draft_tokens[i]:
                # Draft model guessed correctly -- accept this token
                request.output_token_ids.append(target_token)
                if target_token == self.tokenizer.eos_token_id:
                    self._trim_target_kv(request, num_verified)
                    return False
            else:
                # Mismatch: use the target model's correction instead
                request.output_token_ids.append(target_token)
                self._trim_target_kv(request, num_verified)
                break
        else:
            # All draft tokens accepted -- bonus: sample one more from target
            last_target_token_id = sample_token(
                target_logits[-1:, :], request.sampling_params, request.output_token_ids
            )
            last_target_token = last_target_token_id.item()
            request.output_token_ids.append(last_target_token)
            num_verified += 1
            if last_target_token == self.tokenizer.eos_token_id:
                return False

        # Reset draft KV cache (simplified; a production version would backtrack)
        request._draft_past_key_values = None

        return True

    def _trim_target_kv(self, request: GenerationRequest, tokens_used: int):
        """Trim target model KV cache to remove positions from rejected draft tokens.

        After verification, the KV cache may contain entries for draft tokens
        that were not accepted. This trims it to match the actual sequence length.
        """
        if request._past_key_values is None:
            return
        actual_len = len(request.prompt_token_ids) + len(request.output_token_ids)
        trimmed = []
        for k, v in request._past_key_values:
            trimmed.append((k[:, :, :actual_len, :], v[:, :, :actual_len, :]))
        request._past_key_values = tuple(trimmed)
