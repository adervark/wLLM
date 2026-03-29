import subprocess
import sys

def main():
    print("Forcing torch reinstall...")
    cmd = [
        ".venv\\Scripts\\uv.exe", "pip", "install",
        "torch", "torchvision", "torchaudio", "--reinstall",
        "--extra-index-url", "https://download.pytorch.org/whl/cu124"
    ]
    with open("fix_torch.log", "w") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\nExit code: {res.returncode}\n")

if __name__ == "__main__":
    main()
