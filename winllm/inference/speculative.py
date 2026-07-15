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

KV-cache invariant (both models):
  Before a step, ``_past_key_values`` / ``_draft_past_key_values`` cover every
  token *except* the last one in ``output_token_ids`` (length
  ``len(prompt) + len(output) - 1``). The last output token is the input fed
  next. After a step we trim both caches back to that invariant so a rejected
  draft token never leaves a stale key/value behind.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..core.types import GenerationRequest, normalize_eos_ids
from ..sampling import sample_token
from .buffers import trim_cache


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
        num_speculative_tokens: int = 4,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self.device = next(target_model.parameters()).device
        self._eos_ids = normalize_eos_ids(tokenizer.eos_token_id)
        # Acceptance telemetry (drafted vs accepted across all steps)
        self.drafted_tokens = 0
        self.accepted_tokens = 0

    @torch.inference_mode()
    def step(self, request: GenerationRequest) -> bool:
        """Perform one speculative decoding step for a single request.

        Returns True if the request should continue generating, False if it hit EOS.
        """
        # --- Phase 0: make sure the draft model has the prompt + prior context ---
        self._ensure_draft_context(request)

        # --- Phase 1: Draft model proposes N tokens ---
        draft_tokens = self._draft_proposals(request)

        # --- Phase 2: Target model verifies all proposals at once ---
        self.drafted_tokens += len(draft_tokens)
        target_logits = self._verify_proposals(request, draft_tokens)

        # --- Phase 3: Accept or reject each proposed token ---
        return self._accept_or_reject(request, draft_tokens, target_logits)

    def _ensure_draft_context(self, request: GenerationRequest) -> None:
        """Prefill the draft model's KV cache with the committed context.

        Without this the draft model would only ever see the single last token
        and propose from no context, which destroys the acceptance rate. We only
        prefill when the draft cache is missing (first speculative step for the
        request); afterwards it is maintained incrementally by ``_accept_or_reject``.
        """
        if request._draft_past_key_values is not None:
            return

        context = request.prompt_token_ids + request.output_token_ids[:-1]
        if not context:
            return

        outputs = self.draft_model(
            torch.tensor([context], device=self.device), use_cache=True
        )
        request._draft_past_key_values = outputs.past_key_values

    def _draft_proposals(self, request: GenerationRequest) -> list[int]:
        """Use the draft model to quickly generate candidate tokens.

        Advances ``request._draft_past_key_values`` as it goes (it will cover
        the last token plus all-but-the-last proposed token afterwards).
        """
        # Never draft past the request's token budget: each accepted draft
        # plus the bonus token becomes a committed output token, and the
        # scheduler only checks max_tokens after the step. Zero drafts still
        # commits exactly one token via the bonus sample.
        budget = request.sampling_params.max_tokens - len(request.output_token_ids) - 1
        num_drafts = min(self.num_speculative_tokens, max(0, budget))

        draft_tokens: list[int] = []
        last_token = request.output_token_ids[-1]
        proposal_input_ids = torch.tensor([[last_token]], device=self.device)
        temp_past_draft = request._draft_past_key_values

        for _ in range(num_drafts):
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
            if next_token in self._eos_ids:
                break

        request._draft_past_key_values = temp_past_draft
        return draft_tokens

    def _verify_proposals(self, request: GenerationRequest, draft_tokens: list[int]) -> torch.Tensor:
        """Run the target model on all proposed tokens in one forward pass."""
        last_token = request.output_token_ids[-1]
        verify_input_ids = torch.tensor([[last_token] + draft_tokens], device=self.device)

        target_outputs = self.target_model(
            verify_input_ids,
            past_key_values=request._past_key_values,
            use_cache=True,
        )

        request._past_key_values = target_outputs.past_key_values
        # Shape: (num_draft_tokens + 1, vocab_size)
        return target_outputs.logits[0, :, :]

    def _accept_or_reject(
        self, request: GenerationRequest,
        draft_tokens: list[int], target_logits: torch.Tensor,
    ) -> bool:
        """Compare draft vs. target predictions, accept matches, reject at first mismatch."""
        for i in range(len(draft_tokens)):
            # Sample what the target model would have chosen at position i
            target_token_id = sample_token(
                target_logits[i:i+1, :], request.sampling_params, request.output_token_ids
            )
            target_token = target_token_id.item()
            request.output_token_ids.append(target_token)

            if target_token != draft_tokens[i]:
                # Mismatch: the target's correction replaces the rejected draft.
                # Drop the (now stale) draft KV for the rejected positions.
                self._sync_caches(request, draft_tokens, extend_draft=False)
                return target_token not in self._eos_ids

            self.accepted_tokens += 1
            if target_token in self._eos_ids:
                self._sync_caches(request, draft_tokens, extend_draft=False)
                return False
        else:
            # All draft tokens accepted -- bonus: sample one more from target
            last_target_token_id = sample_token(
                target_logits[-1:, :], request.sampling_params, request.output_token_ids
            )
            last_target_token = last_target_token_id.item()
            request.output_token_ids.append(last_target_token)
            self._sync_caches(request, draft_tokens, extend_draft=True)
            return last_target_token not in self._eos_ids

    def _sync_caches(
        self, request: GenerationRequest, draft_tokens: list[int], extend_draft: bool
    ) -> None:
        """Restore the cache invariant on both models after a verification step.

        ``valid_len`` is ``len(prompt) + len(output) - 1`` — the number of tokens
        whose key/values legitimately belong in the cache (the final committed
        token is fed fresh on the next step). The target cache is trimmed down to
        drop any rejected draft positions. The draft cache is brought to the same
        length: when every draft was accepted it is one token short (the last
        proposed token was never fed back into the draft), so we extend it by one
        before trimming.
        """
        valid_len = len(request.prompt_token_ids) + len(request.output_token_ids) - 1

        request._past_key_values = trim_cache(request._past_key_values, valid_len)

        if extend_draft and draft_tokens and request._draft_past_key_values is not None:
            outputs = self.draft_model(
                torch.tensor([[draft_tokens[-1]]], device=self.device),
                past_key_values=request._draft_past_key_values,
                use_cache=True,
            )
            request._draft_past_key_values = outputs.past_key_values

        request._draft_past_key_values = trim_cache(
            request._draft_past_key_values, valid_len
        )
