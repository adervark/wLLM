# GitHub Dark Theme + detect Trim — Design

**Date:** 2026-07-14
**Status:** Approved

## Goal

1. Restyle all rich CLI output to GitHub's dark-mode palette, including
   GitHub-style syntax highlighting in chat code blocks.
2. Slim `wllm detect` down to GPU and related system info: drop the
   `Profile:` display and the Recommended-defaults table.

## Scope

In scope: `winllm/cli/console.py`, `winllm/cli/formatting.py`,
`winllm/cli/commands/{chat,detect,list,benchmark}.py`, `tests/test_cli.py`,
changelog. Out of scope: server output, `--json` payload (unchanged),
any generation/detection behavior.

## Design

### 1. Central theme (`winllm/cli/console.py`)

Define `GITHUB_DARK = rich.theme.Theme({...})` and construct the shared
console as `Console(theme=GITHUB_DARK)`. Semantic styles (GitHub Primer
dark palette):

| Style name       | Value              | Used for                          |
|------------------|--------------------|-----------------------------------|
| `wllm.user`      | `#58a6ff bold`     | `You:` label                      |
| `wllm.assistant` | `#3fb950 bold`     | `Assistant:` label                |
| `wllm.error`     | `#f85149 bold`     | error messages                    |
| `wllm.warning`   | `#d29922`          | CPU-only fallback notice          |
| `wllm.muted`     | `#8b949e`          | thinking text, stats footer       |
| `wllm.border`    | `#30363d`          | rules, table/panel borders        |
| `wllm.accent`    | `#58a6ff`          | spinner                           |

Overrides of rich's built-in style names in the same Theme:

- `logging.level.debug` → `#8b949e`, `.info` → `#58a6ff`,
  `.warning` → `#d29922`, `.error`/`.critical` → `bold #f85149`
- `rule.line` → `#30363d`
- `table.caption` → `#8b949e`
- `markdown.link` → `#58a6ff`
- `status.spinner` → `#58a6ff`

Rich downgrades truecolor automatically on non-truecolor terminals; no
capability checks in our code.

### 2. Call-site changes

- `formatting.py` (`RichChatRenderer`): `Markdown(self._answer,
  code_theme="github-dark")` (pygments built-in style); thinking `Rule`s
  use `style="wllm.border"`, thinking `Text` uses `style="wllm.muted"`.
- `chat.py`: `[bold cyan]You:[/]` → `[wllm.user]You:[/]`;
  `[bold green]Assistant:[/]` → `[wllm.assistant]Assistant:[/]`;
  `[bold red]Error:[/]` → `[wllm.error]Error:[/]`;
  stats footer `[dim]…[/dim]` → `[wllm.muted]…[/wllm.muted]`.
- `detect.py`, `list.py`, `benchmark.py`: tables get
  `border_style="wllm.border"`; benchmark's Summary `Panel` gets
  `border_style="wllm.border"`; detect's `[yellow]No GPUs…[/yellow]` →
  `[wllm.warning]…[/wllm.warning]`.

### 3. detect trim

`cmd_detect` output becomes:

1. GPU table (unchanged columns: GPU / Name / VRAM) — or the CPU-only
   fallback message.
2. One system line: `Platform: {hw.platform} · Total VRAM:
   {hw.total_vram_gb} GB`.
3. `--json` unchanged: `print(f"\nJSON:\n{json.dumps(hw.summary(),
   indent=2)}")` still contains every field including profile and
   defaults.

Removed: the `Profile:` display and the entire "Recommended defaults"
table (and its `hw.defaults` access).

### 4. Tests

- `TestDetectOutput.test_tables_render`: drop the `"single_gpu"` and
  `"Recommended defaults"`/`"4096"` assertions; assert the profile string
  and `"Recommended defaults"` are ABSENT; keep GPU-name assertion; add
  `"Windows"` platform-line assertion. `test_json_flag_prints_plain_json`
  unchanged (JSON keeps all fields).
- New `TestGithubTheme`: the shared console resolves `wllm.user` /
  `wllm.muted` / `wllm.border` (`console.get_style(...)` doesn't raise);
  chat renderer passes `code_theme="github-dark"` (feed a fenced code
  block on a recorded truecolor console and assert a GitHub-dark token
  color appears in `export_text(styles=True)` output, or assert via the
  built `Markdown` object's `code_theme`).
- Existing rendering tests assert text content, not colors — they keep
  passing untouched.

## Risks / trade-offs

- Hex colors assume a dark terminal background; on a light terminal the
  muted grays are low-contrast. Accepted — user chose GitHub Dark.
- `pygments` must ship `github-dark` (present since pygments 2.15;
  transformers already pins a newer pygments). Verified at implementation.

## Changelog

Extend the 2026-07-14 entry: GitHub Dark palette via central rich Theme;
`github-dark` code highlighting; `detect` slimmed to GPU + system line.
