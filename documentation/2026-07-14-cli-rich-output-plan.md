# Rich CLI Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace wLLM's hand-rolled ANSI CLI output with `rich`: live markdown chat rendering, a model-loading spinner, tables for `detect`/`list`/`benchmark`, and quieter restyled logging.

**Architecture:** One shared `rich.console.Console` (new `winllm/cli/console.py`) is used by every command and by the log handler, so spinners/live regions/log lines coexist. The existing `<think>`-tag streaming logic is extracted into a presentation-free `ThinkTagParser`; a new `RichChatRenderer` consumes its segments inside `rich.live.Live` to render markdown as tokens arrive.

**Tech Stack:** Python 3.12, `rich`, argparse CLI, pytest.

**Spec:** `documentation/2026-07-14-cli-rich-output-design.md`

## Global Constraints

- Environment is **uv-managed, no pip**: install with `uv pip install <pkg> --python .venv/Scripts/python.exe`. Never use `uv sync`/`uv run` (they pull ~2.5 GB CUDA wheels).
- Run tests with `.venv/Scripts/python.exe -m pytest tests/test_cli.py -q` (full suite: `tests -q`, ~340 tests, CPU-only, ~5 s).
- Out of scope: `winllm/server/`, the `remove` command, `apply_auto_config`'s prints, any generation behavior.
- The `--json` output of `detect` must stay plain `json.dumps` (machine-readable).
- Keep the `You:` label printed separately from `input()` (line-editing column counting).
- The stats footer content must stay exactly: `N tok · Xs · Y tok/s` plus optional ` · spec A/D (P%)`.

---

### Task 1: rich dependency + shared console

**Files:**
- Modify: `pyproject.toml` (dependencies list, lines 9–22)
- Modify: `.github/workflows/ci.yml:35` (explicit test-dep list)
- Create: `winllm/cli/console.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `winllm.cli.console.console` — a module-level `rich.console.Console` singleton. All later tasks import it as `from ..console import console` (from `winllm/cli/commands/*`) or `from .console import console` (from `winllm/cli/*`).

- [ ] **Step 1: Install rich into the venv**

Run: `uv pip install rich --python .venv/Scripts/python.exe`
Expected: `+ rich==13.x` (plus `markdown-it-py`, `pygments` if not present)

- [ ] **Step 2: Add rich to project + CI dependencies**

In `pyproject.toml`, add to `[project] dependencies` after `"onnxruntime-gpu>=1.24.3",`:

```toml
    "rich>=13.7.0",
```

In `.github/workflows/ci.yml` line 35, add `rich` to the explicit install list (CI installs deps by name, not from pyproject):

```yaml
        run: uv pip install -e . --no-deps --python ${{ matrix.python }} && uv pip install transformers accelerate fastapi "uvicorn[standard]" sse-starlette pydantic pytest httpx rich --python ${{ matrix.python }}
```

- [ ] **Step 3: Write the failing test**

Append to `tests/test_cli.py`:

```python
# ─── Shared console ─────────────────────────────────────────────────────────


class TestSharedConsole:
    def test_console_is_singleton_rich_console(self):
        from rich.console import Console
        from winllm.cli.console import console
        assert isinstance(console, Console)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestSharedConsole -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'winllm.cli.console'`

- [ ] **Step 5: Create the console module**

Create `winllm/cli/console.py`:

```python
"""Shared rich console for all CLI output.

A single Console instance is shared by every command and by the logging
handler so spinners, live regions, and log lines can coexist without
corrupting each other's terminal state. Rich handles TTY/color detection
itself, so callers never check ``isatty`` or emit raw ANSI.
"""

from __future__ import annotations

from rich.console import Console

console = Console()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestSharedConsole -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml winllm/cli/console.py tests/test_cli.py
git commit -m "feat(cli): add rich dependency and shared console"
```

---

### Task 2: ThinkTagParser — presentation-free segment parser

**Files:**
- Modify: `winllm/cli/formatting.py` (add `ThinkTagParser`; leave `ThinkingStreamFormatter` untouched — it is deleted in Task 4)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `winllm.cli.formatting.ThinkTagParser` with:
  - `__init__(self, on_segment: Callable[[str, str], None])` — callback receives `(kind, text)` where kind is `"thinking"` or `"answer"`
  - `feed(self, text: str) -> None`
  - `close(self) -> None`
- Behavior later tasks rely on: tags split across deltas are recognized; a lone partial tail (`"a < b"`) is flushed on close; pure-whitespace gap between `</think>` and the answer is dropped and the first answer chunk is `lstrip("\n")`-ed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
# ─── Think tag parser ───────────────────────────────────────────────────────


def _run_parser(deltas):
    """Feed deltas to a parser and return the emitted (kind, text) segments."""
    from winllm.cli.formatting import ThinkTagParser
    segs = []
    parser = ThinkTagParser(lambda kind, text: segs.append((kind, text)))
    for d in deltas:
        parser.feed(d)
    parser.close()
    return segs


def _joined(segs, kind):
    return "".join(t for k, t in segs if k == kind)


class TestThinkTagParser:
    def test_plain_stream_is_all_answer(self):
        segs = _run_parser(["Hello, ", "world!"])
        assert all(k == "answer" for k, _ in segs)
        assert _joined(segs, "answer") == "Hello, world!"

    def test_thinking_is_segregated_from_answer(self):
        segs = _run_parser(["<think>reason here</think>", "The answer."])
        assert _joined(segs, "thinking") == "reason here"
        assert _joined(segs, "answer") == "The answer."

    def test_tag_split_across_deltas(self):
        segs = _run_parser(["<th", "ink>", "secret", "</thi", "nk>", "done"])
        assert _joined(segs, "thinking") == "secret"
        assert _joined(segs, "answer") == "done"

    def test_no_false_positive_on_partial_tail(self):
        segs = _run_parser(["a < b"])
        assert _joined(segs, "answer") == "a < b"

    def test_whitespace_gap_after_thinking_dropped(self):
        # Models often emit "\n\n" between </think> and the answer; the gap
        # must be dropped even when it arrives as its own delta.
        segs = _run_parser(["<think>x</think>", "\n\n", "Answer"])
        assert _joined(segs, "answer") == "Answer"

    def test_no_segment_callbacks_with_empty_text(self):
        segs = _run_parser(["<think>", "</think>", "hi"])
        assert all(text for _, text in segs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestThinkTagParser -q`
Expected: FAIL with `ImportError: cannot import name 'ThinkTagParser'`

- [ ] **Step 3: Implement ThinkTagParser**

Add to `winllm/cli/formatting.py` (below the `THINK_OPEN`/`THINK_CLOSE` constants, above `ThinkingStreamFormatter`):

```python
class ThinkTagParser:
    """Splits streamed text deltas into (kind, text) segments.

    ``kind`` is ``"thinking"`` for text inside ``<think>...</think>`` and
    ``"answer"`` otherwise. Presentation-free: the callback decides how to
    render each segment. Tags are detected even when split across delta
    boundaries by holding back any trailing substring that could be the
    start of a tag.

    The blank gap models often leave between ``</think>`` and the answer is
    dropped so the answer starts cleanly.
    """

    def __init__(self, on_segment: Callable[[str, str], None]):
        self._on_segment = on_segment
        self._in_thinking = False
        self._buf = ""
        # Whether the answer gap after thinking still needs trimming.
        self._answer_pending = False

    def feed(self, text: str) -> None:
        """Process a streamed text delta, emitting any complete segments."""
        if not text:
            return
        self._buf += text
        while self._buf:
            tag = THINK_CLOSE if self._in_thinking else THINK_OPEN
            idx = self._buf.find(tag)
            if idx != -1:
                self._emit(self._buf[:idx])
                self._buf = self._buf[idx + len(tag):]
                if self._in_thinking:
                    self._answer_pending = True
                self._in_thinking = not self._in_thinking
                continue

            # No complete tag: emit everything except a possible partial tag
            # tail, which we hold back until the next delta completes it.
            hold = self._partial_tail_len(self._buf, tag)
            split = len(self._buf) - hold
            self._emit(self._buf[:split])
            self._buf = self._buf[split:]
            break

    def close(self) -> None:
        """Flush any held-back text."""
        if self._buf:
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, text: str) -> None:
        if not text:
            return
        if self._answer_pending and not self._in_thinking:
            if not text.strip():
                return  # drop the pure-whitespace gap after </think>
            self._answer_pending = False
            text = text.lstrip("\n")
        self._on_segment("thinking" if self._in_thinking else "answer", text)

    @staticmethod
    def _partial_tail_len(buf: str, tag: str) -> int:
        """Length of the longest suffix of ``buf`` that prefixes ``tag``."""
        max_k = min(len(buf), len(tag) - 1)
        for k in range(max_k, 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0
```

Note: `Callable` is already imported in `formatting.py` (`from typing import Callable, Optional`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -q`
Expected: all PASS (old `TestThinkingStreamFormatter` tests still pass — the class is untouched)

- [ ] **Step 5: Commit**

```bash
git add winllm/cli/formatting.py tests/test_cli.py
git commit -m "feat(cli): extract presentation-free ThinkTagParser"
```

---

### Task 3: RichChatRenderer — live markdown streaming

**Files:**
- Modify: `winllm/cli/formatting.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ThinkTagParser` (Task 2), `winllm.cli.console.console` (Task 1)
- Produces: `winllm.cli.formatting.RichChatRenderer` with:
  - `__init__(self, console: Optional[Console] = None)` — starts the live region immediately; defaults to the shared console
  - `feed(self, text: str) -> None`
  - `close(self) -> None` — idempotent (safe to call twice)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
# ─── Rich chat renderer ─────────────────────────────────────────────────────


def _render_chat(deltas):
    """Render deltas through RichChatRenderer on a recorded console."""
    from rich.console import Console
    from winllm.cli.formatting import RichChatRenderer

    console = Console(record=True, width=80, force_terminal=False)
    renderer = RichChatRenderer(console=console)
    for d in deltas:
        renderer.feed(d)
    renderer.close()
    return console.export_text()


class TestRichChatRenderer:
    def test_markdown_is_rendered_not_raw(self):
        out = _render_chat(["some **bold** text"])
        assert "bold" in out
        assert "**" not in out  # markdown consumed, not echoed raw

    def test_thinking_precedes_answer(self):
        out = _render_chat(["<think>pondering</think>", "The answer."])
        assert "thinking" in out    # rule label rendered
        assert "pondering" in out
        assert "The answer." in out
        assert out.index("pondering") < out.index("The answer.")

    def test_plain_stream_without_tags(self):
        out = _render_chat(["Hello, ", "world!"])
        assert "Hello, world!" in out
        assert "thinking" not in out

    def test_close_is_idempotent(self):
        from rich.console import Console
        from winllm.cli.formatting import RichChatRenderer

        console = Console(record=True, width=80, force_terminal=False)
        renderer = RichChatRenderer(console=console)
        renderer.feed("hi")
        renderer.close()
        renderer.close()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestRichChatRenderer -q`
Expected: FAIL with `ImportError: cannot import name 'RichChatRenderer'`

- [ ] **Step 3: Implement RichChatRenderer**

In `winllm/cli/formatting.py`, extend the imports and add the class:

```python
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
```

```python
class RichChatRenderer:
    """Streams a chat response to the terminal via rich.

    Thinking text (``<think>...</think>``) streams as dim text under a dim
    "thinking" rule; the answer streams as live-rendered Markdown, so code
    blocks and emphasis appear formatted as tokens arrive. Refresh is capped
    at 10/s to limit re-render flicker.
    """

    def __init__(self, console: Optional[Console] = None):
        if console is None:
            from .console import console as shared_console
            console = shared_console
        self._console = console
        self._parser = ThinkTagParser(self._on_segment)
        self._thinking = ""
        self._answer = ""
        self._closed = False
        self._live = Live(
            console=self._console,
            refresh_per_second=10,
            vertical_overflow="visible",
        )
        self._live.start()

    def feed(self, text: str) -> None:
        self._parser.feed(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._parser.close()
        self._live.update(self._renderable(), refresh=True)
        self._live.stop()

    def _on_segment(self, kind: str, text: str) -> None:
        if kind == "thinking":
            self._thinking += text
        else:
            self._answer += text
        self._live.update(self._renderable())

    def _renderable(self):
        parts = []
        if self._thinking:
            parts.append(Rule("thinking", style="dim", align="left"))
            parts.append(Text(self._thinking.strip("\n"), style="dim"))
            parts.append(Rule(style="dim"))
        if self._answer:
            parts.append(Markdown(self._answer))
        return Group(*parts)
```

Contingency: if `test_markdown_is_rendered_not_raw` finds the export empty (a
non-terminal `Live` that doesn't print its final frame on stop), add this to
the end of `close()` instead of the `refresh=True` update:

```python
        if not self._console.is_terminal:
            self._console.print(self._renderable())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add winllm/cli/formatting.py tests/test_cli.py
git commit -m "feat(cli): live markdown chat renderer on rich"
```

---

### Task 4: Rewire chat command; delete ThinkingStreamFormatter

**Files:**
- Modify: `winllm/cli/commands/chat.py` (full rewrite below)
- Modify: `winllm/cli/formatting.py` (delete `ThinkingStreamFormatter`, `_DIM`, `_RESET`; update module docstring)
- Modify: `tests/test_cli.py` (delete `TestThinkingStreamFormatter`, `_run_formatter`, and the `ThinkingStreamFormatter` import)

**Interfaces:**
- Consumes: `RichChatRenderer` (Task 3), `console` (Task 1)
- Produces: nothing new — behavior change only. After this task `ThinkingStreamFormatter` no longer exists anywhere.

- [ ] **Step 1: Delete the superseded tests**

In `tests/test_cli.py`: remove the `from winllm.cli.formatting import ThinkingStreamFormatter` import (line 8), the `_run_formatter` helper, and the entire `TestThinkingStreamFormatter` class (its behaviors are covered by `TestThinkTagParser` + `TestRichChatRenderer`).

- [ ] **Step 2: Rewrite chat.py**

Replace `winllm/cli/commands/chat.py` with:

```python
"""Interactive chat in the terminal."""

from __future__ import annotations

from .common import build_model_config, apply_auto_config
from ..console import console
from ..formatting import RichChatRenderer


def cmd_chat(args):
    """Interactive chat in the terminal."""
    from ...config import SamplingParams, SchedulerConfig, KVCacheConfig
    from ...core.types import GenerationRequest
    from ...inference import InferenceEngine
    from ...models.chat_template import format_chat_prompt

    model_config = build_model_config(args)
    kv_cache_config = KVCacheConfig()

    if args.auto_config:
        scheduler_config = SchedulerConfig()
        apply_auto_config(model_config, scheduler_config, kv_cache_config)

    engine = InferenceEngine(model_config, kv_cache_config)

    with console.status(
        f"Loading [bold]{args.model}[/bold] ({model_config.quantization.value})..."
    ):
        engine.load_model()
    console.print("Model loaded. Type [bold]quit[/bold] or [bold]exit[/bold] to stop.\n")

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    try:
        while True:
            try:
                # Label printed separately so styling never sits inside the
                # input() prompt, where line editing can miscount columns.
                console.print("\n[bold cyan]You:[/] ", end="")
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            prompt = format_chat_prompt(engine.tokenizer, messages)

            # Label on its own line: the renderer's live region draws
            # block-level markdown, which can't share the label's line.
            console.print("\n[bold green]Assistant:[/]")
            renderer = RichChatRenderer()

            def on_token(text: str, finished: bool):
                if finished:
                    renderer.close()
                else:
                    renderer.feed(text)

            request = GenerationRequest(
                prompt=prompt,
                sampling_params=sampling,
                _stream_callback=on_token,
            )

            # Speculation counters are lifetime totals on the engine; snapshot
            # around generate() to report this response's share.
            spec = engine.speculative_engine
            drafted_before = spec.drafted_tokens if spec else 0
            accepted_before = spec.accepted_tokens if spec else 0

            result = engine.generate(request)
            renderer.close()  # idempotent; ensures the live region is released

            if result.error:
                # Nothing streamed and there is no assistant turn to keep;
                # drop the user turn too so the failed exchange doesn't
                # poison the context of every later prompt.
                messages.pop()
                console.print(f"\n[bold red]Error:[/] {result.error}")
                continue

            stats = (
                f"{result.generation_tokens} tok · "
                f"{result.elapsed:.1f}s · "
                f"{result.tokens_per_second:.1f} tok/s"
            )
            if spec:
                drafted = spec.drafted_tokens - drafted_before
                accepted = spec.accepted_tokens - accepted_before
                if drafted:
                    stats += f" · spec {accepted}/{drafted} ({accepted / drafted:.0%})"
            console.print(f"[dim]── {stats}[/dim]")

            messages.append({"role": "assistant", "content": result.output_text})

    finally:
        engine.unload_model()
```

- [ ] **Step 3: Delete ThinkingStreamFormatter**

In `winllm/cli/formatting.py`: delete the `ThinkingStreamFormatter` class, the `_DIM`/`_RESET` constants, and the `import sys` (no longer used). Update the module docstring's last paragraph to say rendering is done with rich instead of ANSI. Keep `THINK_OPEN`/`THINK_CLOSE`, `ThinkTagParser`, `RichChatRenderer`.

Check nothing else references it:

Run: `git grep -n ThinkingStreamFormatter -- "*.py"`
Expected: no output

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add winllm/cli/commands/chat.py winllm/cli/formatting.py tests/test_cli.py
git commit -m "feat(cli): chat streams via rich live markdown with load spinner"
```

---

### Task 5: RichHandler logging + third-party quieting

**Files:**
- Modify: `winllm/cli/main.py:20-27` (`setup_logging`)
- Test: `tests/test_cli.py` (replace `TestSetupLogging`)

**Interfaces:**
- Consumes: `console` (Task 1)
- Produces: `setup_logging(verbose: bool = False)` — same signature, now installs a `RichHandler` on the root logger (with `force=True` so repeated calls reconfigure) and caps `transformers`, `urllib3`, `filelock`, `accelerate` at WARNING unless verbose.

- [ ] **Step 1: Replace the logging tests**

In `tests/test_cli.py`, replace the whole `TestSetupLogging` class with:

```python
class TestSetupLogging:
    def test_installs_rich_handler(self):
        from rich.logging import RichHandler
        setup_logging(verbose=False)
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(isinstance(h, RichHandler) for h in root.handlers)

    def test_third_party_quieted_by_default(self):
        setup_logging(verbose=False)
        for name in ("transformers", "urllib3", "filelock", "accelerate"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_verbose_enables_debug_everywhere(self):
        setup_logging(verbose=True)
        assert logging.getLogger().level == logging.DEBUG
        for name in ("transformers", "urllib3", "filelock", "accelerate"):
            assert logging.getLogger(name).level == logging.NOTSET
```

Add `import logging` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestSetupLogging -q`
Expected: FAIL (no RichHandler on root; third-party loggers at NOTSET)

- [ ] **Step 3: Rewrite setup_logging**

Replace `setup_logging` in `winllm/cli/main.py`:

```python
# Third-party libraries that flood INFO during model load; capped at
# WARNING unless the user asks for verbose output.
_NOISY_LOGGERS = ("transformers", "urllib3", "filelock", "accelerate")


def setup_logging(verbose: bool = False):
    """Configure logging on the shared rich console."""
    from rich.logging import RichHandler
    from .console import console

    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(
        console=console,
        show_path=verbose,
        show_time=verbose,
        rich_tracebacks=True,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(
            logging.NOTSET if verbose else logging.WARNING
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add winllm/cli/main.py tests/test_cli.py
git commit -m "feat(cli): rich log handler, quiet third-party loggers by default"
```

---

### Task 6: detect — tables

**Files:**
- Modify: `winllm/cli/commands/detect.py` (full rewrite below)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `console` (Task 1); `DeviceInfo` fields already used today: `device_count`, `devices[]` (`.index`, `.name`, `.total_vram_gb`), `profile.value`, `platform`, `total_vram_gb`, `defaults` (`.default_quantization`, `.max_batch_size`, `.max_model_len`, `.tensor_parallel_size`, `.device_map_strategy`), `summary()`.
- Produces: nothing new — output change only.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
# ─── detect output ──────────────────────────────────────────────────────────


def _fake_hw():
    from types import SimpleNamespace
    gpu = SimpleNamespace(index=0, name="RTX 4070 Laptop", total_vram_gb=8.0)
    return SimpleNamespace(
        device_count=1,
        devices=[gpu],
        profile=SimpleNamespace(value="single_gpu"),
        platform="Windows",
        total_vram_gb=8.0,
        defaults=SimpleNamespace(
            default_quantization="4bit",
            max_batch_size=4,
            max_model_len=4096,
            tensor_parallel_size=1,
            device_map_strategy="auto",
        ),
        summary=lambda: {"device_count": 1},
    )


class TestDetectOutput:
    def test_tables_render(self, monkeypatch):
        from types import SimpleNamespace
        from rich.console import Console
        import winllm.cli.commands.detect as detect_mod

        recorded = Console(record=True, width=100, force_terminal=False)
        monkeypatch.setattr(detect_mod, "console", recorded)
        monkeypatch.setattr(
            "winllm.hardware.DeviceInfo.detect", staticmethod(_fake_hw)
        )

        detect_mod.cmd_detect(SimpleNamespace(json=False, verbose=False))
        out = recorded.export_text()

        assert "RTX 4070 Laptop" in out
        assert "single_gpu" in out
        assert "Recommended defaults" in out
        assert "4096" in out

    def test_json_flag_prints_plain_json(self, monkeypatch, capsys):
        from types import SimpleNamespace
        from rich.console import Console
        import winllm.cli.commands.detect as detect_mod

        monkeypatch.setattr(
            detect_mod, "console", Console(record=True, width=100, force_terminal=False)
        )
        monkeypatch.setattr(
            "winllm.hardware.DeviceInfo.detect", staticmethod(_fake_hw)
        )

        detect_mod.cmd_detect(SimpleNamespace(json=True, verbose=False))
        assert '"device_count": 1' in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestDetectOutput -q`
Expected: FAIL with `AttributeError: ... has no attribute 'console'` (detect.py doesn't import it yet)

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
        console.print("[yellow]No GPUs detected — CPU-only mode[/yellow]")
    else:
        gpus = Table(title="Hardware", title_justify="left")
        gpus.add_column("GPU", justify="right")
        gpus.add_column("Name")
        gpus.add_column("VRAM", justify="right")
        for gpu in hw.devices:
            gpus.add_row(str(gpu.index), gpu.name, f"{gpu.total_vram_gb} GB")
        console.print(gpus)

    console.print(
        f"Profile: [bold]{hw.profile.value}[/bold]   "
        f"Platform: {hw.platform}   Total VRAM: {hw.total_vram_gb} GB"
    )

    d = hw.defaults
    defaults = Table(title="Recommended defaults", title_justify="left")
    defaults.add_column("Setting")
    defaults.add_column("Value", justify="right")
    defaults.add_row("Quantization", str(d.default_quantization))
    defaults.add_row("Max batch size", str(d.max_batch_size))
    defaults.add_row("Max context length", str(d.max_model_len))
    defaults.add_row("Tensor parallel", str(d.tensor_parallel_size))
    defaults.add_row("Device map", str(d.device_map_strategy))
    console.print()
    console.print(defaults)

    if args.json:
        print(f"\nJSON:\n{json.dumps(hw.summary(), indent=2)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -q`
Expected: all PASS

- [ ] **Step 5: Manual smoke check**

Run: `.venv/Scripts/python.exe -m winllm detect`
Expected: box-drawn Hardware and Recommended defaults tables, real GPU listed

- [ ] **Step 6: Commit**

```bash
git add winllm/cli/commands/detect.py tests/test_cli.py
git commit -m "feat(cli): detect renders hardware and defaults as rich tables"
```

---

### Task 7: list — table with caption

**Files:**
- Modify: `winllm/cli/commands/list.py` (only the printing block, lines 65–79; the cache-scanning logic is unchanged)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `console` (Task 1)
- Produces: nothing new — output change only. Empty-cache / missing-directory messages keep their wording (switched to `console.print`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
# ─── list output ────────────────────────────────────────────────────────────


class TestListOutput:
    def test_models_render_as_table(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from rich.console import Console
        import winllm.cli.commands.list as list_mod

        hub = tmp_path / "hub"
        model_dir = hub / "models--org--tiny-model" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 2048)

        monkeypatch.setenv("HF_HOME", str(tmp_path))
        recorded = Console(record=True, width=100, force_terminal=False)
        monkeypatch.setattr(list_mod, "console", recorded)

        list_mod.cmd_list(SimpleNamespace(verbose=False))
        out = recorded.export_text()

        assert "org/tiny-model" in out
        assert "2 KB" in out
        assert "1 model(s)" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py::TestListOutput -q`
Expected: FAIL with `AttributeError: ... has no attribute 'console'`

- [ ] **Step 3: Switch list.py to a rich table**

In `winllm/cli/commands/list.py`, add imports at the top:

```python
from rich.table import Table

from ..console import console
```

Replace the two early-return `print(...)` calls with `console.print(...)` (same
strings). Then replace everything from `# Print table` (line 65) to the end of
the function with:

```python
    table = Table(
        caption_justify="left",
    )
    table.add_column("Model")
    table.add_column("Size", justify="right")
    table.add_column("Modified", justify="right")

    total_size = 0
    for name, size_bytes, modified in models:
        total_size += size_bytes
        table.add_row(name, _fmt_size(size_bytes), modified)

    table.caption = (
        f"{len(models)} model(s), {_fmt_size(total_size)} total · {cache_dir}"
    )
    console.print()
    console.print(table)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -q`
Expected: all PASS

- [ ] **Step 5: Manual smoke check**

Run: `.venv/Scripts/python.exe -m winllm list`
Expected: box-drawn table of cached models with size/modified columns and a caption line

- [ ] **Step 6: Commit**

```bash
git add winllm/cli/commands/list.py tests/test_cli.py
git commit -m "feat(cli): list renders cached models as a rich table"
```

---

### Task 8: benchmark — live results table, summary panel, spinner; changelog

**Files:**
- Modify: `winllm/cli/commands/benchmark.py` (full rewrite below)
- Modify: `documentation/CHANGELOG.md` (new entry at top)

**Interfaces:**
- Consumes: `console` (Task 1)
- Produces: nothing new — output change only.

No unit test for this task: `cmd_benchmark` requires a loaded engine and its
logic is untouched (only prints change). The whole-suite run plus a real-model
manual check gate it instead.

- [ ] **Step 1: Rewrite benchmark.py**

Replace `winllm/cli/commands/benchmark.py` with:

```python
"""Run a simple throughput benchmark."""

from __future__ import annotations

from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .common import build_model_config, apply_auto_config
from ..console import console


def cmd_benchmark(args):
    """Run a simple throughput benchmark."""
    from ...config import SamplingParams, SchedulerConfig, KVCacheConfig
    from ...core.types import GenerationRequest
    from ...hardware import get_aggregate_gpu_memory
    from ...inference import InferenceEngine

    model_config = build_model_config(args)
    kv_cache_config = KVCacheConfig()
    scheduler_config = SchedulerConfig()

    if args.auto_config:
        apply_auto_config(model_config, scheduler_config, kv_cache_config)

    engine = InferenceEngine(model_config, kv_cache_config)
    with console.status(
        f"Loading [bold]{args.model}[/bold] ({model_config.quantization.value})..."
    ):
        engine.load_model()

    mem = get_aggregate_gpu_memory()
    console.print(f"\nGPU memory: {mem}")
    console.print(
        f"Running benchmark ({args.num_prompts} prompts, "
        f"{args.max_tokens} max tokens each)...\n"
    )

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to sort a list using quicksort.",
        "What are the main differences between TCP and UDP?",
        "Describe the process of photosynthesis step by step.",
        "Write a haiku about artificial intelligence.",
    ]

    sampling = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    total_tokens = 0
    total_time = 0.0

    table = Table(title="Benchmark results", title_justify="left")
    table.add_column("Prompt", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("tok/s", justify="right")

    # Rows appear in the terminal as each prompt finishes.
    with Live(table, console=console, refresh_per_second=4):
        for i in range(min(args.num_prompts, len(prompts))):
            request = GenerationRequest(
                prompt=prompts[i],
                sampling_params=sampling,
            )
            result = engine.generate(request)
            total_tokens += result.generation_tokens
            total_time += result.elapsed

            table.add_row(
                str(i + 1),
                str(result.generation_tokens),
                f"{result.elapsed:.2f}s",
                f"{result.tokens_per_second:.1f}",
            )

    summary = (
        f"Total: {total_tokens} tokens in {total_time:.2f}s\n"
        f"Average: {total_tokens / total_time:.1f} tokens/sec\n"
        f"GPU memory: {get_aggregate_gpu_memory()}"
    )
    console.print(Panel(summary, title="Summary", expand=False))

    engine.unload_model()
```

- [ ] **Step 2: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all PASS

- [ ] **Step 3: End-to-end manual check with a real model**

Run: `.venv/Scripts/python.exe -m winllm benchmark -m HuggingFaceTB/SmolLM2-135M-Instruct --max-tokens 32 --num-prompts 2`
Expected: spinner during load, results table filling in live, Summary panel at the end, no noisy transformers INFO logs

Then: `.venv/Scripts/python.exe -m winllm chat -m HuggingFaceTB/SmolLM2-135M-Instruct --max-tokens 64`
Expected: spinner during load; ask for "a python function that adds two numbers" — the code block renders syntax-highlighted; stats footer appears dim; `quit` exits cleanly

- [ ] **Step 4: Add changelog entry**

In `documentation/CHANGELOG.md`, insert directly below the `---` on line 14 (above the `## [Unreleased] - 2026-07-07` entry):

```markdown
## [Unreleased] - 2026-07-14
### Rich CLI Output

CLI presentation moved from hand-rolled ANSI to the `rich` library (new
runtime dependency). Shared `Console` in `winllm/cli/console.py`; spec in
`documentation/2026-07-14-cli-rich-output-design.md`.

#### Added
- **Live markdown chat streaming** (`cli/formatting.py`) — the assistant's reply renders as markdown while it streams (syntax-highlighted code blocks, emphasis, lists); `<think>` reasoning streams as dim text under a dim rule. `ThinkingStreamFormatter` replaced by a presentation-free `ThinkTagParser` plus a `RichChatRenderer` built on `rich.live.Live` (capped at 10 refreshes/s).
- **Model-load spinner** — `chat` and `benchmark` show an animated status while `load_model()` runs.
- **Tables** — `detect` (hardware + recommended defaults), `list` (cached models with size/modified and a totals caption), `benchmark` (per-prompt results filling in live, summary panel).

#### Changed
- **Logging** (`cli/main.py`) — `RichHandler` on the shared console; `transformers`/`urllib3`/`filelock`/`accelerate` capped at WARNING unless `-v` (which also restores timestamps and module paths).

```

- [ ] **Step 5: Commit**

```bash
git add winllm/cli/commands/benchmark.py documentation/CHANGELOG.md
git commit -m "feat(cli): benchmark live results table and summary panel; changelog"
```
