"""VRAM-based KV cache capacity estimation."""

from __future__ import annotations

import logging

import torch

from ..config import KVCacheConfig

logger = logging.getLogger(__name__)


class KVMemoryEstimator:
    """Estimates how many KV blocks fit in the configured VRAM budget."""

    def __init__(self, config: KVCacheConfig):
        self.config = config

    def per_token_kv_bytes(self) -> int:
        """Estimate bytes consumed per token in KV cache.

        Uses model dimensions if available, otherwise falls back to
        a conservative estimate for a ~7B parameter model.
        """
        if (
            self.config.num_layers > 0
            and self.config.num_kv_heads > 0
            and self.config.head_dim > 0
        ):
            # Precise calculation from model architecture
            # KV per token = 2 (key+value) * num_layers * num_kv_heads * head_dim * dtype_bytes
            per_token = (
                2
                * self.config.num_layers
                * self.config.num_kv_heads
                * self.config.head_dim
                * self.config.dtype_bytes
            )
            logger.info(
                "KV cache per-token estimate (model-aware): %d bytes "
                "(layers=%d, kv_heads=%d, head_dim=%d, dtype=%dB)",
                per_token,
                self.config.num_layers,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.dtype_bytes,
            )
            return per_token
        else:
            # Fallback estimate for ~7B model: ~512 KB per token
            logger.info("KV cache per-token estimate (fallback): 524288 bytes")
            return 512 * 1024

    @staticmethod
    def total_available_vram() -> float:
        """Get total available VRAM in bytes across all GPUs."""
        if not torch.cuda.is_available():
            return 0.0

        total_available = 0.0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # Use reserved (the size of PyTorch's caching pool) rather than just
            # allocated, so cached-but-unallocated VRAM isn't double-counted as
            # free. This keeps the KV block budget conservative.
            used = max(torch.cuda.memory_allocated(i), torch.cuda.memory_reserved(i))
            total_available += props.total_memory - used

        return total_available

    def estimate_max_blocks(self) -> int:
        """Estimate how many KV blocks we can afford based on GPU memory."""
        # device_count can be 0 while is_available() is True (e.g. the driver
        # is present but CUDA_VISIBLE_DEVICES hides every GPU) — that would
        # compute a 0-byte budget and reject every request.
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return self.config.max_blocks_per_seq * 4  # Fallback for CPU

        available = self.total_available_vram() * self.config.gpu_memory_fraction
        per_token_bytes = self.per_token_kv_bytes()
        estimated_bytes_per_block = self.config.block_size * per_token_bytes

        max_blocks = max(1, int(available / estimated_bytes_per_block))

        # Scale cap based on total VRAM rather than fixed 2048
        total_vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(torch.cuda.device_count())
        )
        # ~500 blocks per 10GB as a reasonable cap
        dynamic_cap = max(2048, int(total_vram_gb * 50))

        result = min(max_blocks, dynamic_cap)
        logger.info(
            "Max KV blocks: %d (%.1f GB available VRAM, %d bytes/block, cap=%d)",
            result, available / (1024 ** 3), estimated_bytes_per_block, dynamic_cap,
        )
        return result
