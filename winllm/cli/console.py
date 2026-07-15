"""Shared rich console for all CLI output.

A single Console instance is shared by every command and by the logging
handler so spinners, live regions, and log lines can coexist without
corrupting each other's terminal state. Rich handles TTY/color detection
itself, so callers never check ``isatty`` or emit raw ANSI.

Styling is centralized in ``GITHUB_DARK`` (GitHub Primer dark-mode
palette): commands reference semantic ``wllm.*`` style names instead of
colors, so the whole look changes in this one file. Rich downgrades the
truecolor values automatically on terminals that can't show them.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

GITHUB_DARK = Theme(
    {
        # Semantic wLLM styles
        "wllm.user": "bold #58a6ff",
        "wllm.assistant": "bold #3fb950",
        "wllm.error": "bold #f85149",
        "wllm.warning": "#d29922",
        "wllm.muted": "#8b949e",
        "wllm.border": "#30363d",
        "wllm.accent": "#58a6ff",
        # Overrides of rich's built-in style names
        "logging.level.debug": "#8b949e",
        "logging.level.info": "#58a6ff",
        "logging.level.warning": "#d29922",
        "logging.level.error": "bold #f85149",
        "logging.level.critical": "bold #f85149",
        "rule.line": "#30363d",
        "table.caption": "#8b949e",
        "markdown.link": "#58a6ff",
        "status.spinner": "#58a6ff",
    }
)

console = Console(theme=GITHUB_DARK)
