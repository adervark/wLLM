import argparse
import subprocess
import sys
import os

def check_optimum_installed():
    try:
        import optimum
    except ImportError:
        print("Error: 'optimum' is not installed.")
        print("Please run: uv pip install optimum optimum[onnxruntime-gpu]")
        sys.exit(1)

def export_onnx(model_id: str, output_dir: str, quantize: str = None):
    check_optimum_installed()
    
    # We use optimum-cli to perform the export
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", model_id,
        "--task", "text-generation-with-past"
    ]
    
    if quantize == "fp16":
        cmd.extend(["--dtype", "fp16", "--optimize", "O4", "--device", "cuda"])
    elif quantize == "fp32":
        cmd.extend(["--dtype", "fp32", "--optimize", "O2"])

    cmd.append(output_dir)

    print(f"Running ONNX export for {model_id}...")
    print("Command:", " ".join(cmd))
    print("This may take several minutes depending on the model size.")
    
    # Run the command and stream output
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

    if process.returncode == 0:
        print(f"\n[Success] Model successfully exported to {os.path.abspath(output_dir)}")
        print(f"You can now run this model in wLLM by pointing to the exported folder:")
        print(f"  wllm chat -m {os.path.abspath(output_dir)} --inference-backend onnxruntime")
    else:
        print("\n[Error] Failed to export model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile/Export a HuggingFace model to ONNX for wLLM")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model ID (e.g. LiquidAI/LFM2.5-1.2B-Thinking)")
    parser.add_argument("--output", "-o", required=True, help="Output directory for the ONNX model files")
    parser.add_argument("--quantize", "-q", choices=["fp16", "fp32"], default="fp16", help="Precision and optimization level (fp16 applies O4 GPU optimizations)")
    
    args = parser.parse_args()
    export_onnx(args.model, args.output, args.quantize)
