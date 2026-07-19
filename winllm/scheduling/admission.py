"""Memory-aware admission with prefix-cache matching."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from ..kvcache.prefix import compute_prefix_hashes

if TYPE_CHECKING:
    from ..core.types import GenerationRequest
    from ..kvcache import KVCacheManager

logger = logging.getLogger(__name__)


class AdmissionResult(Enum):
    ADMITTED = "admitted"
    NO_MEMORY = "no_memory"


class AdmissionController:
    """Decides whether a request fits in the KV cache right now.

    Also performs prefix-cache matching so admitted requests can skip
    recomputing prompt blocks that are already cached.
    """

    def __init__(self, kv_cache_manager: "KVCacheManager"):
        self._kv = kv_cache_manager

    def try_admit(self, req: "GenerationRequest") -> AdmissionResult:
        """Attempt to admit one request, allocating its KV blocks on success."""
        block_size = self._kv.block_size

        # Check KV cache admission with prefix matching
        prompt_token_ids = req.prompt_token_ids
        prefix_hashes = compute_prefix_hashes(prompt_token_ids, block_size)

        matched_blocks, matched_tensors = self._kv.match_prefix(prefix_hashes)
        matched_len = len(matched_blocks) * block_size

        req._prefix_cache_token_len = matched_len
        req._prefix_past_key_values = matched_tensors

        prompt_len = len(prompt_token_ids)
        max_new = req.sampling_params.max_tokens
        needed_tokens = prompt_len + max_new - matched_len
        if not self._kv.can_allocate(needed_tokens):
            # Cached prefixes are reclaimable budget: evict before rejecting,
            # so a request is only refused when it genuinely cannot fit.
            self._kv.ensure_free_blocks(
                self._kv.allocator.blocks_needed(needed_tokens)
            )
        if self._kv.can_allocate(needed_tokens):
            self._kv.allocate_sequence(
                req.request_id,
                prompt_len,
                prefix_blocks=matched_blocks,
            )
            return AdmissionResult.ADMITTED

        return AdmissionResult.NO_MEMORY
