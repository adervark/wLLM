"""Process-global CUDA backend configuration."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_cuda_backends() -> None:
    """Configure CUDA backends for maximum inference throughput.

    Called once before model loading. These are global PyTorch settings:
      - TF32: Enables TF32 tensor cores on Ampere+ GPUs for ~3x faster
        float32 matmuls with negligible accuracy impact for inference.
      - Expandable segments: Reduces CUDA memory fragmentation on Windows
        where the default allocator is particularly wasteful.
    """
    import torch

    if not torch.cuda.is_available():
        return

    # TF32 for Ampere+ (compute >= 8.0) -- free 2-3x speedup on float32 ops
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Expandable segments allocator -- reduces fragmentation-induced OOMs
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    if 'expandable_segments' not in alloc_conf:
        new_conf = 'expandable_segments:True'
        if alloc_conf:
            new_conf = f'{alloc_conf},{new_conf}'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = new_conf

    logger.info("CUDA backends configured: TF32=on, allocator=%s",
                os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'))
