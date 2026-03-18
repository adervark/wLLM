import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    print("Checking imports...")
    from winllm.cli import main
    print("winllm.cli imported successfully")
    from winllm.commands.serve import cmd_serve
    print("winllm.commands.serve imported successfully")
    from winllm.commands.chat import cmd_chat
    print("winllm.commands.chat imported successfully")
    from winllm.commands.benchmark import cmd_benchmark
    print("winllm.commands.benchmark imported successfully")
    from winllm.commands.list import cmd_list
    print("winllm.commands.list imported successfully")
    from winllm.commands.detect import cmd_detect
    print("winllm.commands.detect imported successfully")
    print("All imports SUCCESSFUL")
except Exception as e:
    print(f"IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
