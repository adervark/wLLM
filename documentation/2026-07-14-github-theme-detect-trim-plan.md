# GitHub Dark Theme + detect Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle all rich CLI output to GitHub's dark-mode palette (including `github-dark` code highlighting in chat) and slim `wllm detect` to GPU + system info.

**Architecture:** A `GITHUB_DARK` `rich.theme.Theme` in `winllm/cli/console.py` defines semantic `wllm.*` styles plus overrides of rich's built-in style names; the shared Console is constructed with it. Call sites switch from hardcoded colors to semantic names. `detect` loses the Profile display and Recommended-defaults table.

**Tech Stack:** Python 3.12, rich (Theme/Markdown/Table), pygments `github-dark` (verified present in the venv), pytest.

**Spec:** `documentation/2026-07-14-github-theme-detect-trim-design.md`

## Global Constraints

- Environment is uv-managed, no pip; nothing needs installing. Never `uv sync`/`uv run`.
- Run tests with `.venv/Scripts/python.exe -m pytest tests/test_cli.py -q` (full suite: `tests -q`).
- **Any test that constructs its own `Console` and renders `wllm.*` styles MUST pass `theme=GITHUB_DARK`** (import from `winllm.cli.console`), or rich raises `MissingStyle`.
- `detect --json` output is unchanged: `print(f"\nJSON:\n{json.dumps(hw.summary(), indent=2)}")` with all fields (profile, defaults) intact.
- Palette values (verbatim from spec): user `bold #58a6ff`, assistant `bold #3fb950`, error `bold #f85149`, warning `#d29922`, muted `#8b949e`, border `#30363d`, accent `#58a6ff`.

---

### Task 1: GITHUB_DARK theme on the shared console

**Files:**
- Modify: `winllm/cli/console.py` (full new content below)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `winllm.cli.console.GITHUB_DARK` (a `rich.theme.Theme`) and the shared `console` now constructed as `Console(theme=GITHUB_DARK)`. Tasks 2–3 use style names `wllm.user`, `wllm.assistant`, `wllm.error`, `wllm.warning`, `wllm.muted`, `wllm.border`, `wllm.accent` and import `GITHUB_DARK` in tests.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
# ─── GitHub theme ───────────────────────────────────────────────────────────


class TestGithubTheme:
    def test_console_resolves_semantic_styles(self):
        from winllm.cli.console import console

        for name in (
            "wllm.user",
            "wllm.assistant",
            "wllm.error",
            "wllm.warning",
            "wllm.muted",
            "wllm.border",
            "wllm.accent",
        ):
            # get_style raises rich.errors.MissingStyle if undefined.
            assert console.get_style(name) is not None

    def test_user_style_is_github_blue(self):
        from winllm.cli.console import console

        style = console.get_style("wllm.user")
        assert style.color.triplet.hex.lower() == "#58a6ff"
        assert style.bold
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestGithubTheme -q`
Expected: FAIL with `MissingStyle` (or assertion error on `get_style`)

- [ ] **Step 3: Implement the theme**

Replace `winllm/cli/console.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all PASS (existing tests unaffected — the theme only adds/overrides named styles)

- [ ] **Step 5: Commit**

```bash
git add winllm/cli/console.py tests/test_cli.py
git commit -m "feat(cli): GITHUB_DARK theme on the shared console"
```

---

### Task 2: Restyle chat, renderer, list, benchmark

**Files:**
- Modify: `winllm/cli/commands/chat.py:46,65,94,107` (markup swaps below)
- Modify: `winllm/cli/formatting.py:143-151` (`_renderable`) and its class docstring
- Modify: `winllm/cli/commands/list.py:69` (add `border_style`)
- Modify: `winllm/cli/commands/benchmark.py:52,81` (add `border_style`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `GITHUB_DARK` and the `wllm.*` style names from Task 1.
- Produces: `RichChatRenderer` renders answers with `Markdown(..., code_theme="github-dark")`; thinking rules use `wllm.border`, thinking text uses `wllm.muted`.

- [ ] **Step 1: Theme the existing renderer-test consoles**

In `tests/test_cli.py`, every `Console(record=True, ...)` used with `RichChatRenderer` must carry the theme. Update `_render_chat` and the two standalone renderer tests (`test_close_is_idempotent`, `test_feed_after_close_is_ignored`) to construct:

```python
from winllm.cli.console import GITHUB_DARK
console = Console(record=True, width=80, force_terminal=False, theme=GITHUB_DARK)
```

(Only the constructor line changes in each; assertions stay.)

- [ ] **Step 2: Write the failing code-theme test**

Append inside `TestGithubTheme`:

```python
    def test_chat_code_blocks_use_github_dark(self):
        from rich.console import Console
        from winllm.cli.console import GITHUB_DARK
        from winllm.cli.formatting import RichChatRenderer

        console = Console(
            record=True,
            width=80,
            force_terminal=True,
            color_system="truecolor",
            theme=GITHUB_DARK,
        )
        renderer = RichChatRenderer(console=console)
        renderer.feed("```python\ndef f():\n    return 1\n```")
        renderer.close()
        ansi = console.export_text(styles=True)
        # github-dark renders Python keywords in #ff7b72 (255;123;114).
        assert "255;123;114" in ansi
```

Contingency: if the truecolor export proves unreliable under `Live`, assert
structurally instead — feed the same input, then
`md = renderer._renderable().renderables[-1]` and
`assert md.code_theme == "github-dark"` (and note the swap in your report).

- [ ] **Step 3: Run to verify the new test fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestGithubTheme -q`
Expected: `test_chat_code_blocks_use_github_dark` FAILS (monokai is the default code theme; no `255;123;114`)

- [ ] **Step 4: Implement the restyles**

`winllm/cli/formatting.py` — replace `_renderable` (and update the class docstring's "dim" wording to "muted"):

```python
    def _renderable(self) -> Group:
        parts = []
        if self._thinking:
            parts.append(Rule("thinking", style="wllm.border", align="left"))
            parts.append(Text(self._thinking.strip("\n"), style="wllm.muted"))
            parts.append(Rule(style="wllm.border"))
        if self._answer:
            parts.append(Markdown(self._answer, code_theme="github-dark"))
        return Group(*parts)
```

`winllm/cli/commands/chat.py` — four markup swaps:

```python
                console.print("\n[wllm.user]You:[/] ", end="")
```
```python
            console.print("\n[wllm.assistant]Assistant:[/]")
```
```python
                console.print(f"\n[wllm.error]Error:[/] {result.error}")
```
```python
            console.print(f"[wllm.muted]── {stats}[/wllm.muted]")
```

`winllm/cli/commands/list.py:69` — the Table constructor becomes:

```python
    table = Table(
        caption_justify="left",
        border_style="wllm.border",
    )
```

`winllm/cli/commands/benchmark.py:52` and `:81`:

```python
    table = Table(title="Benchmark results", title_justify="left", border_style="wllm.border")
```
```python
    console.print(Panel(summary, title="Summary", expand=False, border_style="wllm.border"))
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all PASS (list test renders `wllm.border` via its themed console from Step 1 — if it wasn't themed, fix it the same way)

Note: `TestListOutput.test_models_render_as_table` constructs its own recorded console — it must also get `theme=GITHUB_DARK` now that list.py renders `wllm.border`.

- [ ] **Step 6: Commit**

```bash
git add winllm/cli/formatting.py winllm/cli/commands/chat.py winllm/cli/commands/list.py winllm/cli/commands/benchmark.py tests/test_cli.py
git commit -m "feat(cli): GitHub Dark styling for chat, renderer, list, benchmark"
```

---

### Task 3: Trim detect + changelog

**Files:**
- Modify: `winllm/cli/commands/detect.py` (full new content below)
- Modify: `tests/test_cli.py` (`TestDetectOutput`)
- Modify: `documentation/CHANGELOG.md` (extend the 2026-07-14 entry)

**Interfaces:**
- Consumes: `wllm.warning`, `wllm.border` styles (Task 1); themed test consoles (Task 2 pattern).
- Produces: nothing new — output change only. `--json` payload untouched.

- [ ] **Step 1: Update the detect tests (failing first)**

In `tests/test_cli.py`, `TestDetectOutput.test_tables_render`: theme the recorded console and replace the assertion block:

```python
        recorded = Console(record=True, width=100, force_terminal=False, theme=GITHUB_DARK)
```
(add `from winllm.cli.console import GITHUB_DARK` to the test's imports), and:

```python
        assert "RTX 4070 Laptop" in out
        assert "Windows" in out                      # system line
        assert "single_gpu" not in out               # profile removed
        assert "Recommended defaults" not in out     # defaults table removed
        assert "4096" not in out
```

Also theme the console in `test_json_flag_prints_plain_json` the same way; its JSON assertion stays unchanged.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestDetectOutput -q`
Expected: `test_tables_render` FAILS (`single_gpu` and `Recommended defaults` still present)

- [ ] **Step 3: Rewrite detect.py**

Replace `winllm/cli/commands/detect.py` with:

```python
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
```

- [ ] **Step 4: Run the full suite and smoke check**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all PASS

Run: `.venv/Scripts/python.exe -m winllm detect`
Expected: GPU table with GitHub-gray borders, then `Platform: Windows · Total VRAM: 8.0 GB`; no Profile, no Recommended defaults

- [ ] **Step 5: Changelog**

In `documentation/CHANGELOG.md`, inside the `## [Unreleased] - 2026-07-14` entry: add to `#### Added`:

```markdown
- **GitHub Dark theme** — central rich `Theme` (`GITHUB_DARK` in `cli/console.py`, GitHub Primer dark palette) drives labels, rules, tables, panels, spinner, and log-level badges via semantic `wllm.*` style names; chat code blocks highlight with pygments' `github-dark`.
```

and to `#### Changed`:

```markdown
- **`detect` slimmed** — now shows the GPU table plus a `Platform … · Total VRAM …` line; the Profile display and Recommended-defaults table are gone from human output (both remain in `--json`).
```

- [ ] **Step 6: Commit**

```bash
git add winllm/cli/commands/detect.py tests/test_cli.py documentation/CHANGELOG.md
git commit -m "feat(cli): github-dark detect trim to GPU and system info"
```
