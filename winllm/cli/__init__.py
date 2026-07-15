"""Command-line interface for WinLLM.

``winllm.cli:main`` is the console-script entry point (see pyproject.toml).
"""

from .main import main, setup_logging, _add_common_model_args, _add_scaling_args

__all__ = ["main", "setup_logging", "_add_common_model_args", "_add_scaling_args"]
