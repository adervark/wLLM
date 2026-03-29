"""Command implementations for the CLI."""

from .serve import cmd_serve
from .chat import cmd_chat
from .benchmark import cmd_benchmark
from .list import cmd_list
from .detect import cmd_detect
from .remove import cmd_remove


__all__ = [
    "cmd_serve",
    "cmd_chat",
    "cmd_benchmark",
    "cmd_list",
    "cmd_detect",
    "cmd_remove",
]
