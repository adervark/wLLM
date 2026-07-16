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
        "status.spinner": "#58a6ff",
        # Markdown: rich's defaults (yellow bullets, magenta block quotes,
        # cyan inline code) clash with the palette — mute them to match.
        "markdown.link": "#58a6ff",
        "markdown.link_url": "#58a6ff underline",
        "markdown.code": "#79c0ff on #161b22",
        "markdown.block_quote": "#8b949e",
        "markdown.item.bullet": "#8b949e",
        "markdown.item.number": "#8b949e",
        "markdown.hr": "#30363d",
        # Repr highlighting (applied to log lines and printed objects):
        # rich's defaults color numbers cyan, strings green, and paths
        # magenta. Restrict to blue for values and gray for structure.
        "repr.number": "#79c0ff",
        "repr.number_complex": "#79c0ff",
        "repr.str": "#a5d6ff",
        "repr.bool_true": "#79c0ff italic",
        "repr.bool_false": "#79c0ff italic",
        "repr.none": "#79c0ff italic",
        "repr.url": "#58a6ff underline",
        "repr.path": "#8b949e",
        "repr.filename": "#c9d1d9",
        "repr.call": "#e6edf3",
        "repr.attrib_name": "#79c0ff",
        "repr.attrib_value": "#a5d6ff",
        "repr.tag_start": "#8b949e",
        "repr.tag_end": "#8b949e",
        "repr.tag_name": "#79c0ff",
        "repr.tag_contents": "#c9d1d9",
        "repr.uuid": "#79c0ff",
        "repr.ipv4": "#79c0ff",
        "repr.ipv6": "#79c0ff",
        "repr.eui48": "#79c0ff",
        "repr.eui64": "#79c0ff",
        "repr.ellipsis": "#8b949e",
        "repr.indent": "#8b949e",
    }
)

console = Console(theme=GITHUB_DARK)
