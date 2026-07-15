# CLI Output Formatting with rich — Design

**Date:** 2026-07-14
**Status:** Approved

## Goal

Replace wLLM's hand-rolled ANSI terminal output with the `rich` library across the
CLI: live markdown rendering in chat, a loading spinner, box-drawn tables for
`detect`/`list`/`benchmark`, and quieter, restyled logging.

## Scope

In scope: `winllm/cli/` (main.py, formatting.py, commands/chat.py, detect.py,
list.py, benchmark.py) and `tests/test_cli.py`.

Out of scope: server/API output (`winllm/server/`), the `remove` command, and any
behavior change to generation, scheduling, or model loading itself.

## Design

### 1. Dependency & shared console

- Add `rich` to `[project.dependencies]` in `pyproject.toml` (pure Python; no
  build steps, compatible with the uv-only environment).
- New module `winllm/cli/console.py` exposing a single shared
  `rich.console.Console` instance. All CLI commands and the logging handler use
  this one console so spinners, live regions, and log lines coexist without
  corrupting each other.
- Rich's own TTY/color detection replaces the hand-rolled `sys.stdout.isatty()`
  checks and ANSI escape constants. Piped/redirected output degrades to plain
  text automatically.

### 2. Logging

- `setup_logging()` in `winllm/cli/main.py` switches from `basicConfig` with a
  `%(asctime)s | %(levelname)-8s | ...` format to `RichHandler` on the shared
  console: colored level badges, message-only format, no module path clutter.
- Default (no `-v`): wLLM's own loggers stay at INFO; noisy third-party loggers
  (`transformers`, `urllib3`, `filelock`, `accelerate`) are capped at WARNING.
- `-v`: DEBUG for everything, with timestamps enabled on the handler.

### 3. Loading spinner

- `chat` and `benchmark` wrap `engine.load_model()` in
  `console.status("Loading <model> (<quant>)…")`. Log lines emitted during load
  print above the spinner (RichHandler shares the console, so rich manages the
  live region).
- `serve` keeps plain sequential logs — it is a long-running server process and
  a spinner adds nothing.

### 4. Chat: live markdown + thinking panel

- The incremental `<think>`-tag parser currently inside
  `ThinkingStreamFormatter` (`winllm/cli/formatting.py`) is refactored into a
  presentation-free segment parser: it consumes streamed text deltas and emits
  `(kind, text)` segments where kind is `thinking` or `answer`. The
  tag-split-across-deltas logic and partial-tail hold-back behavior are kept
  as-is.
- A new renderer in `formatting.py` consumes segments inside `rich.live.Live`:
  - Thinking text streams as dim text under a dim "thinking" rule (rich-drawn
    replacement for today's `┌─ thinking ─` box).
  - Answer text streams as a live-rendered `rich.markdown.Markdown` — bold,
    lists, and syntax-highlighted code blocks appear as tokens arrive. Refresh
    rate capped (~10/s) to limit flicker.
  - The blank-gap trim between `</think>` and the answer is preserved.
- The stats footer keeps its exact content
  (`N tok · Xs · Y tok/s · spec A/D (P%)`) rendered as a dim caption line.
- `You:` / `Assistant:` labels keep their bold cyan/green styling via rich
  markup. The label is still printed separately from `input()` so line editing
  never miscounts columns.
- Error handling is unchanged: on `result.error` the user turn is popped and the
  error printed (now styled bold red).

### 5. Tables: detect / list / benchmark

- `detect`: a GPU table (index, name, VRAM) plus a recommended-defaults table;
  profile/platform/total-VRAM as labeled lines above. `--json` output stays
  plain `json.dumps` (machine-readable escape hatch).
- `list`: model/size/modified columns become a `rich.table.Table`; the
  "N model(s), X GB total / Cache: path" summary becomes the table caption.
  Empty-cache and missing-directory messages unchanged in content.
- `benchmark`: per-prompt results as table rows (prompt #, tokens, time,
  tok/s); the total/average/GPU-memory summary as a closing panel.

### 6. Tests

- The six `TestThinkingStreamFormatter` tests in `tests/test_cli.py` are ported
  to the segment parser (pass-through, segregation, tag split across deltas,
  partial-tail false positive). The two ANSI on/off tests are replaced by
  renderer tests.
- Rendering tests use `Console(record=True, width=80, force_terminal=...)` with
  a fixed width so output is deterministic and independent of the real
  terminal.
- Existing argparse/logging tests are updated where `setup_logging` changes
  (handler type, third-party logger levels).

## Error handling

- No new failure modes: rich falls back to plain text on dumb terminals and
  redirected output. If `rich` is missing the CLI fails at import with a normal
  dependency error, same as any other dependency.

## Risks / trade-offs

- Live markdown re-renders can flicker on some terminals; mitigated by the
  capped refresh rate. Windows Terminal handles it well.
- `ThinkingStreamFormatter`'s public shape changes; it is only used by
  `cmd_chat` and tests, both updated in the same change.
