"""Detect and display hardware info."""

from __future__ import annotations


def cmd_detect(args):
    """Detect and display hardware info."""
    from ..device import DeviceInfo
    import json

    hw = DeviceInfo.detect()

    print(f"\nHardware Detection")
    print(f"--------------------------------------------------------------")

    if hw.device_count == 0:
        print(f"  No GPUs detected — CPU-only mode")
    else:
        for gpu in hw.devices:
            print(f"  GPU {gpu.index}: {gpu.name} ({gpu.total_vram_gb} GB)")

    print(f"\n  Profile:     {hw.profile.value}")
    print(f"  Platform:    {hw.platform}")
    print(f"  Total VRAM:  {hw.total_vram_gb}")
    
    print(f"\nRecommended defaults:")
    d = hw.defaults
    print(f"  Quantization:       {d.default_quantization}")
    print(f"  Max batch size:     {d.max_batch_size}")
    print(f"  Max context length: {d.max_model_len}")
    print(f"  Tensor parallel:    {d.tensor_parallel_size}")
    print(f"  Device map:         {d.device_map_strategy}")
    print(f"--------------------------------------------------------------")

    if args.json:
        print(f"\nJSON:\n{json.dumps(hw.summary(), indent=2)}")
