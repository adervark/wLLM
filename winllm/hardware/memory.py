"""GPU memory introspection utilities."""

from __future__ import annotations


class MemoryUtils:
    """Utilities for tracking GPU memory usage."""

    @staticmethod
    def get_all_info() -> list[dict[str, float]]:
        """Get memory info for all GPUs."""
        import torch
        if not torch.cuda.is_available():
            return []
        result = []
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            result.append({
                "device": i,
                "name": torch.cuda.get_device_properties(i).name,
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(total - reserved, 2),
            })
        return result

    @staticmethod
    def get_total_vram() -> float:
        """Get total VRAM across all GPUs in bytes."""
        import torch
        if not torch.cuda.is_available():
            return 0.0
        total = 0.0
        for i in range(torch.cuda.device_count()):
            total += torch.cuda.get_device_properties(i).total_memory
        return total

    @staticmethod
    def get_info(device_index: int = 0) -> dict[str, float]:
        """Get current GPU memory usage in GB for a single device."""
        import torch
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}
        total = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        return {
            "total": round(total, 2),
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(total - reserved, 2),
        }

    @staticmethod
    def get_aggregate_info() -> dict[str, float]:
        """Get aggregate GPU memory across all devices."""
        import torch
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "reserved": 0, "free": 0, "device_count": 0}
        total = allocated = reserved = 0.0
        count = torch.cuda.device_count()
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            total += props.total_memory / (1024 ** 3)
            allocated += torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved += torch.cuda.memory_reserved(i) / (1024 ** 3)
        return {
            "total": round(total, 2),
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(total - reserved, 2),
            "device_count": count,
        }


# Function-style aliases — the rest of the codebase consumes these names.
get_all_gpu_memory_info = MemoryUtils.get_all_info
get_total_gpu_memory = MemoryUtils.get_total_vram
get_gpu_memory_info = MemoryUtils.get_info
get_aggregate_gpu_memory = MemoryUtils.get_aggregate_info
