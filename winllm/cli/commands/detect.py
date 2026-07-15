"""Detect and display hardware info."""

from __future__ import annotations

from rich.table import Table

from ..console import console


def cmd_detect(args):
    """Detect and display hardware info."""
    from ...hardware import DeviceInfo
    import json

    hw = DeviceInfo.detect()

    console.print()
    if hw.device_count == 0:
        console.print("[wllm.warning]No GPUs detected — CPU-only mode[/wllm.warning]")
    else:
        gpus = Table(title="Hardware", title_justify="left", border_style="wllm.border")
        gpus.add_column("GPU", justify="right")
        gpus.add_column("Name")
        gpus.add_column("VRAM", justify="right")
        for gpu in hw.devices:
            gpus.add_row(str(gpu.index), gpu.name, f"{gpu.total_vram_gb} GB")
        console.print(gpus)

    console.print(f"Platform: {hw.platform} · Total VRAM: {hw.total_vram_gb} GB")

    if args.json:
        print(f"\nJSON:\n{json.dumps(hw.summary(), indent=2)}")
