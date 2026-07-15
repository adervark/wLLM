"""Model-free speculative decoding driven by a SuffixCache.

Same contract as ``SpeculativeEngine`` (one ``step()`` per scheduler decode
iteration), but the draft comes from pattern matching over the request's own
prompt and output instead of a draft model — zero extra VRAM, zero extra
model forwards, works with any architecture the target runs on.

Each step:
  1. SuffixCache proposes 0..N draft tokens (0 → plain single-token decode,
     so a missed match costs nothing over the normal path).
  2. The target verifies [last_committed] + drafts in one forward pass.
  3. Committed tokens are always the target's own samples, so output is
     token-identical to non-speculative decoding; drafts only decide how
     many tokens one forward pass can commit.
  4. The KV cache is cropped back to the committed length so rejected draft
     positions never leave stale keys/values behind.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..core.types import GenerationRequest, normalize_eos_ids
from ..sampling import sample_token
from .buffers import trim_cache
from .eager import eager_decode_step
from .suffix_cache import SuffixCache

logger = logging.getLogger(__name__)


class SuffixSpeculativeEngine:
    """SuffixDecoding-style speculation: draft from history, verify once."""

    def __init__(
        self,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        alpha: float = 2.0,
        max_spec: int = 16,
        input_buffer=None,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.suffix_cache = SuffixCache(alpha=alpha, max_spec=max_spec)
        self.device = next(target_model.parameters()).device
        self._input_buffer = input_buffer  # optional DecodeInputBuffer
        self._eos_ids = normalize_eos_ids(tokenizer.eos_token_id)
        # Acceptance telemetry (drafted vs accepted across all steps)
        self.drafted_tokens = 0
        self.accepted_tokens = 0

    @torch.inference_mode()
    def validate(self, probe_token_ids: list[int]) -> bool:
        """Check that the model's KV cache rolls back losslessly via crop.

        Speculative rejection requires dropping cache entries for rejected
        draft positions. Pure-attention caches crop exactly, but hybrid
        caches with recurrent state (e.g. LFM2's conv cache) may not rewind —
        which would silently corrupt every generation after a rejected draft.
        Probe: run a context, feed junk tokens, crop them away, and require
        logits identical to a run that never saw the junk.
        """
        try:
            if len(probe_token_ids) < 2:
                return False
            ids = torch.tensor([probe_token_ids[:-1]], device=self.device)
            junk = ids[:, : min(3, ids.shape[1])]
            last = probe_token_ids[-1]

            ref_cache = self.target_model(ids, use_cache=True).past_key_values
            ref, _ = eager_decode_step(
                self.target_model, last, ref_cache, self.device
            )

            cache = self.target_model(ids, use_cache=True).past_key_values
            self.target_model(junk, past_key_values=cache, use_cache=True)
            cache = trim_cache(cache, ids.shape[1])
            test, _ = eager_decode_step(
                self.target_model, last, cache, self.device
            )

            # Sound rollback reproduces the exact same computation, so the
            # tolerance only needs to absorb float noise, not model drift.
            diff = (ref.float() - test.float()).abs().max().item()
            if diff > 1e-2:
                logger.warning(
                    "Suffix decoding disabled: this model's cache does not "
                    "roll back losslessly (crop probe logit diff %.3f) — "
                    "speculative rejection would corrupt output", diff,
                )
                return False
            return True
        except Exception as e:
            logger.warning("Suffix decoding disabled: cache rollback probe failed (%s)", e)
            return False

    @torch.inference_mode()
    def step(self, request: GenerationRequest) -> bool:
        """One decode iteration. Returns False when EOS was committed."""
        self.suffix_cache.sync(
            request.request_id,
            request.prompt_token_ids + request.output_token_ids,
        )

        draft_tokens = self.suffix_cache.propose(request.request_id)
        # Never speculate past the request's token budget: each accepted
        # draft plus the bonus token becomes a committed output token, and
        # the scheduler only checks max_tokens after the step.
        budget = request.sampling_params.max_tokens - len(request.output_token_ids) - 1
        if len(draft_tokens) > budget:
            draft_tokens = draft_tokens[:max(0, budget)]
        if not draft_tokens:
            return self._plain_step(request)

        # Verify: one forward over the last committed token + all drafts.
        last_token = request.output_token_ids[-1]
        input_ids = torch.tensor([[last_token] + draft_tokens], device=self.device)
        outputs = self.target_model(
            input_ids, past_key_values=request._past_key_values, use_cache=True
        )
        request._past_key_values = outputs.past_key_values
        logits = outputs.logits[0, :, :]  # (1 + num_drafts, vocab)

        self.drafted_tokens += len(draft_tokens)
        alive = self._accept_or_reject(request, draft_tokens, logits)

        # Drop KV entries for rejected draft positions. valid_len is the
        # committed context minus the final token, which is fed next step.
        valid_len = len(request.prompt_token_ids) + len(request.output_token_ids) - 1
        request._past_key_values = trim_cache(request._past_key_values, valid_len)

        if not alive:
            self.suffix_cache.evict(request.request_id)
        return alive

    def _accept_or_reject(
        self, request: GenerationRequest,
        draft_tokens: list[int], logits: torch.Tensor,
    ) -> bool:
        """Commit the target's samples; drafts only extend the run.

        Position i's logits are what the target predicts *after* draft i-1
        (position 0 follows the last committed token). The target's sample is
        always committed, so a rejected draft still yields one real token.
        """
        for i in range(len(draft_tokens)):
            target_token = sample_token(
                logits[i:i + 1, :], request.sampling_params, request.output_token_ids
            ).item()
            request.output_token_ids.append(target_token)

            if target_token in self._eos_ids:
                return False
            if target_token != draft_tokens[i]:
                return True
            self.accepted_tokens += 1

        # Every draft accepted — the final position's logits are a free
        # bonus prediction from the same forward pass.
        bonus = sample_token(
            logits[-1:, :], request.sampling_params, request.output_token_ids
        ).item()
        request.output_token_ids.append(bonus)
        return bonus not in self._eos_ids

    def _plain_step(self, request: GenerationRequest) -> bool:
        """Standard single-token decode for steps with no usable draft."""
        logits, request._past_key_values = eager_decode_step(
            self.target_model, request.output_token_ids[-1],
            request._past_key_values, self.device, self._input_buffer,
        )
        token = sample_token(
            logits, request.sampling_params, request.output_token_ids
        ).item()
        request.output_token_ids.append(token)
        return token not in self._eos_ids

    @property
    def acceptance_rate(self) -> Optional[float]:
        if self.drafted_tokens == 0:
            return None
        return self.accepted_tokens / self.drafted_tokens
