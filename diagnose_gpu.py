"""Standalone diagnostic tool for Windows GPU issues."""
import os
import sys
import subprocess
import platform

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running '{' '.join(cmd)}': {str(e)}"

def main():
    print("="*60)
    print("      WinLLM GPU Diagnostic Tool")
    print("="*60)
    
    # 1. System Info
    print(f"\n[1] System info")
    print(f"  OS:       {platform.system()} {platform.release()}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Venv:     {os.environ.get('VIRTUAL_ENV', 'None')}")

    # 2. Driver Info
    print(f"\n[2] NVIDIA Driver (via nvidia-smi)")
    smi = run_cmd("nvidia-smi")
    if "NVIDIA-SMI has failed" in smi:
        print("  !!! NVIDIA-SMI FAILED: Driver not installed or GPU not active.")
    elif "not found" in smi or "not recognized" in smi:
        print("  !!! NVIDIA-SMI NOT FOUND: Are NVIDIA drivers installed?")
    else:
        # Extract driver version and CUDA version
        lines = smi.split('\n')
        for line in lines:
            if "Driver Version" in line:
                print(f"  {line.strip()}")
                break
        else:
            print("  SMI output found but version info missing.")

    # 3. PyTorch Info
    print(f"\n[3] PyTorch status")
    try:
        import torch
        print(f"  Version:         {torch.__version__}")
        print(f"  CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device Name:     {torch.cuda.get_device_name(0)}")
            print(f"  Device Count:    {torch.cuda.device_count()}")
            print(f"  CUDA Built with: {torch.version.cuda}")
        else:
            print("  !!! CUDA NOT AVAILABLE IN TORCH")
            # Check for common mismatch
            if "+cu" not in torch.__version__ and "dev" not in torch.__version__:
                print("  HINT: You likely have the CPU-only version from PyPI.")
    except ImportError:
        print("  !!! TORCH NOT INSTALLED")

    # 4. ONNX Runtime Info
    print(f"\n[4] ONNX Runtime status")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Available Providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            print("  CUDA Provider:       SUCCESS")
        else:
            print("  CUDA Provider:       MISSING (Check onnxruntime-gpu installation)")
    except ImportError:
        print("  !!! ONNXRUNTIME NOT INSTALLED")

    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
