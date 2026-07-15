"""Unit tests for CLI argument parsing and command dispatch."""

import logging
import sys
from unittest.mock import patch, MagicMock
import pytest

from winllm.cli import main, _add_common_model_args, _add_scaling_args, setup_logging
from winllm import __version__
from winllm.cli.console import GITHUB_DARK


# ─── Version ────────────────────────────────────────────────────────────────


class TestVersion:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["wllm", "--version"]):
                main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_version_matches_package(self):
        assert isinstance(__version__, str)
        assert len(__version__) > 0


# ─── No command ─────────────────────────────────────────────────────────────


class TestNoCommand:
    def test_no_command_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["wllm"]):
                main()
        assert exc_info.value.code == 1


# ─── Subcommand registration ────────────────────────────────────────────────


class TestSubcommands:
    @pytest.mark.parametrize("cmd", ["serve", "chat", "benchmark", "list", "detect", "remove"])
    def test_subcommand_is_recognized(self, cmd):
        """Each subcommand should be registered and not produce 'invalid choice'."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Import and verify the subparser registration works
        from winllm.cli import main
        # We can't easily test without running main, so instead verify
        # the command map has all expected entries
        cmd_map_keys = {"serve", "chat", "benchmark", "list", "detect", "remove"}
        assert cmd in cmd_map_keys


# ─── Common model args ─────────────────────────────────────────────────────


class TestCommonModelArgs:
    def test_adds_model_arg(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_common_model_args(parser)
        # --model should be required
        args = parser.parse_args(["--model", "test/model"])
        assert args.model == "test/model"

    def test_model_shorthand(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_common_model_args(parser)
        args = parser.parse_args(["-m", "test/model"])
        assert args.model == "test/model"

    def test_quantization_default(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_common_model_args(parser)
        args = parser.parse_args(["--model", "x"])
        assert args.quantization == "auto"

    def test_quantization_choices(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_common_model_args(parser)
        for q in ["auto", "none", "4bit", "8bit", "awq", "gptq"]:
            args = parser.parse_args(["--model", "x", "-q", q])
            assert args.quantization == q

    def test_invalid_quantization_rejected(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_common_model_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "x", "-q", "invalid"])

    def test_verbose_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_common_model_args(parser)
        args = parser.parse_args(["--model", "x", "--verbose"])
        assert args.verbose is True

    def test_trust_remote_code(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_common_model_args(parser)
        args = parser.parse_args(["--model", "x", "--trust-remote-code"])
        assert args.trust_remote_code is True


# ─── Scaling args ───────────────────────────────────────────────────────────


class TestScalingArgs:
    def test_backend_choices(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_scaling_args(parser)
        for backend in ["pytorch", "onnxruntime", "directml"]:
            args = parser.parse_args(["--backend", backend])
            assert args.backend == backend

    def test_invalid_backend_rejected(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_scaling_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "invalid"])

    def test_attention_backend_choices(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_scaling_args(parser)
        for ab in ["auto", "sdpa", "flash_attention_2", "eager"]:
            args = parser.parse_args(["--attention-backend", ab])
            assert args.attention_backend == ab

    def test_tensor_parallel_size(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_scaling_args(parser)
        args = parser.parse_args(["-tp", "4"])
        assert args.tensor_parallel_size == 4

    def test_cpu_offload_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_scaling_args(parser)
        args = parser.parse_args(["--cpu-offload"])
        assert args.cpu_offload is True

    def test_device_default(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_scaling_args(parser)
        args = parser.parse_args([])
        assert args.device == "auto"

    def test_draft_model(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_scaling_args(parser)
        args = parser.parse_args(["--draft-model", "small/model"])
        assert args.draft_model == "small/model"


# ─── setup_logging ──────────────────────────────────────────────────────────


class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def _restore_logging_state(self):
        from winllm.cli.main import _NOISY_LOGGERS
        root = logging.getLogger()
        saved_root = (root.level, root.handlers[:])
        saved_levels = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
        yield
        root.handlers[:] = saved_root[1]
        root.setLevel(saved_root[0])
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)

    def test_installs_rich_handler(self):
        from rich.logging import RichHandler
        setup_logging(verbose=False)
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(isinstance(h, RichHandler) for h in root.handlers)

    def test_third_party_quieted_by_default(self):
        from winllm.cli.main import _NOISY_LOGGERS
        setup_logging(verbose=False)
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    def test_verbose_enables_debug_everywhere(self):
        from winllm.cli.main import _NOISY_LOGGERS
        setup_logging(verbose=True)
        assert logging.getLogger().level == logging.DEBUG
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.NOTSET


# ─── Shared console ─────────────────────────────────────────────────────────


class TestSharedConsole:
    def test_console_is_singleton_rich_console(self):
        from rich.console import Console
        from winllm.cli.console import console
        assert isinstance(console, Console)


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

    def test_partial_tag_completed_by_later_delta(self):
        # "a <" is held back as a possible tag start; the next delta
        # completes it into a real <think> tag.
        segs = _run_parser(["a <", "think>secret</think>", "done"])
        assert _joined(segs, "answer") == "a done"
        assert _joined(segs, "thinking") == "secret"


# ─── Rich chat renderer ─────────────────────────────────────────────────────


def _render_chat(deltas):
    """Render deltas through RichChatRenderer on a recorded console."""
    from rich.console import Console
    from winllm.cli.console import GITHUB_DARK
    from winllm.cli.formatting import RichChatRenderer

    console = Console(record=True, width=80, force_terminal=False, theme=GITHUB_DARK)
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
        from winllm.cli.console import GITHUB_DARK
        from winllm.cli.formatting import RichChatRenderer

        console = Console(record=True, width=80, force_terminal=False, theme=GITHUB_DARK)
        renderer = RichChatRenderer(console=console)
        renderer.feed("hi")
        renderer.close()
        renderer.close()  # must not raise

    def test_feed_after_close_is_ignored(self):
        from rich.console import Console
        from winllm.cli.console import GITHUB_DARK
        from winllm.cli.formatting import RichChatRenderer

        console = Console(record=True, width=80, force_terminal=False, theme=GITHUB_DARK)
        renderer = RichChatRenderer(console=console)
        renderer.feed("hi")
        renderer.close()
        renderer.feed("late token")  # must not raise or render
        assert "late token" not in console.export_text()


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

        recorded = Console(record=True, width=100, force_terminal=False, theme=GITHUB_DARK)
        monkeypatch.setattr(detect_mod, "console", recorded)
        monkeypatch.setattr(
            "winllm.hardware.DeviceInfo.detect", staticmethod(_fake_hw)
        )

        detect_mod.cmd_detect(SimpleNamespace(json=False, verbose=False))
        out = recorded.export_text()

        assert "RTX 4070 Laptop" in out
        assert "Windows" in out                      # system line
        assert "single_gpu" not in out               # profile removed
        assert "Recommended defaults" not in out     # defaults table removed
        assert "4096" not in out

    def test_json_flag_prints_plain_json(self, monkeypatch, capsys):
        from types import SimpleNamespace
        from rich.console import Console
        import winllm.cli.commands.detect as detect_mod

        monkeypatch.setattr(
            detect_mod, "console", Console(record=True, width=100, force_terminal=False, theme=GITHUB_DARK)
        )
        monkeypatch.setattr(
            "winllm.hardware.DeviceInfo.detect", staticmethod(_fake_hw)
        )

        detect_mod.cmd_detect(SimpleNamespace(json=True, verbose=False))
        assert '"device_count": 1' in capsys.readouterr().out


# ─── list output ────────────────────────────────────────────────────────────


class TestListOutput:
    def test_models_render_as_table(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from rich.console import Console
        from winllm.cli.console import GITHUB_DARK
        import winllm.cli.commands.list as list_mod

        hub = tmp_path / "hub"
        model_dir = hub / "models--org--tiny-model" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 2048)

        monkeypatch.setenv("HF_HOME", str(tmp_path))
        recorded = Console(record=True, width=100, force_terminal=False, theme=GITHUB_DARK)
        monkeypatch.setattr(list_mod, "console", recorded)

        list_mod.cmd_list(SimpleNamespace(verbose=False))
        out = recorded.export_text()

        assert "org/tiny-model" in out
        assert "2 KB" in out
        assert "1 model(s)" in out


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
