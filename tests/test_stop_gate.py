"""Unit tests for StopStringGate: streamed text must never contain a stop string."""

from winllm.server.streaming import StopStringGate


class TestStopStringGate:
    def test_no_stops_passthrough(self):
        gate = StopStringGate([])
        assert gate.push("hello") == "hello"
        assert gate.flush() == ""

    def test_holds_back_potential_stop_prefix(self):
        gate = StopStringGate(["###"])
        # "abc##" — the trailing "##" could be the start of "###"
        assert gate.push("abc##") == "abc"
        # completing the stop: nothing more is emitted
        assert gate.push("#done") == ""
        assert gate.triggered
        assert gate.flush() == ""

    def test_stop_spanning_two_pushes_never_emitted(self):
        gate = StopStringGate(["STOP"])
        out = gate.push("hello ST")
        out += gate.push("OP world")
        out += gate.flush()
        assert "STOP" not in out
        assert out == "hello "

    def test_false_alarm_text_is_released(self):
        gate = StopStringGate(["###"])
        out = gate.push("abc##")
        out += gate.push("x more text ")
        out += gate.flush()
        assert out == "abc##x more text "

    def test_flush_releases_clean_tail(self):
        gate = StopStringGate(["<end>"])
        out = gate.push("final answer")
        assert len(out) < len("final answer")  # tail held back
        assert out + gate.flush() == "final answer"

    def test_multiple_stops_earliest_wins(self):
        gate = StopStringGate(["YY", "XXXX"])
        out = gate.push("abYYcdXXXX")
        out += gate.flush()
        assert out == "ab"

    def test_single_char_stop(self):
        gate = StopStringGate(["\n"])
        assert gate.push("line one") == "line one"  # holdback = 0
        assert gate.push("\nrest") == ""
        assert gate.triggered
