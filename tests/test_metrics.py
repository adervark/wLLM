"""Tests for latency percentile tracking and Prometheus metrics rendering."""

import asyncio

from winllm.config import SamplingParams, SchedulerConfig
from winllm.core.types import GenerationRequest
from winllm.scheduling.scheduler import Scheduler
from winllm.scheduling.stats import LatencyWindow, SchedulerStats
from winllm.server.metrics import render_prometheus

from test_scheduler_loop import EOS_ID, ScriptedEngine


class TestLatencyWindow:
    def test_empty_percentile_is_zero(self):
        w = LatencyWindow()
        assert w.percentile(0.5) == 0.0

    def test_single_value(self):
        w = LatencyWindow()
        w.record(3.0)
        assert w.percentile(0.5) == 3.0
        assert w.percentile(0.99) == 3.0

    def test_nearest_rank(self):
        w = LatencyWindow()
        for v in range(1, 101):  # 1..100
            w.record(float(v))
        assert w.percentile(0.5) == 50.0
        assert w.percentile(0.9) == 90.0
        assert w.percentile(0.99) == 99.0

    def test_bounded_window_drops_oldest(self):
        w = LatencyWindow(maxlen=10)
        for v in range(100):
            w.record(float(v))
        assert len(w) == 10
        assert w.percentile(0.01) == 90.0  # oldest retained value

    def test_percentiles_dict(self):
        w = LatencyWindow()
        w.record(2.0)
        d = w.percentiles()
        assert set(d) == {"p50", "p90", "p99"}


class TestStatsRecording:
    def test_record_request_populates_windows(self):
        stats = SchedulerStats()
        req = GenerationRequest(prompt="x")
        req.created_at = 100.0
        req.started_at = 100.5
        req.first_token_at = 101.0
        req.finished_at = 102.0
        req.output_token_ids = [1, 2, 3]
        stats.record_request(req)
        assert stats.ttft.percentile(0.5) == 1.0
        assert stats.e2e_latency.percentile(0.5) == 2.0
        assert stats.request_tokens_per_second.percentile(0.5) == 2.0  # 3 tokens / 1.5s

    def test_to_dict_includes_percentiles(self):
        d = SchedulerStats().to_dict()
        assert "ttft_seconds" in d
        assert "e2e_latency_seconds" in d
        assert "tokens_per_second" in d


class TestPrometheusRendering:
    def _run_one(self, engine):
        scheduler = Scheduler(engine, SchedulerConfig())

        async def _go():
            req = GenerationRequest(prompt="hi", sampling_params=SamplingParams(max_tokens=10))
            return await asyncio.wait_for(scheduler.submit(req), timeout=10)

        asyncio.run(_go())
        return scheduler

    def test_render_after_completed_request(self):
        engine = ScriptedEngine(script=[1, 2, EOS_ID])
        scheduler = self._run_one(engine)
        text = render_prometheus(scheduler)

        assert "winllm_requests_total 1" in text
        assert "winllm_requests_completed_total 1" in text
        assert 'winllm_ttft_seconds{quantile="0.5"}' in text
        assert "winllm_kv_cache_blocks_total" in text
        # Prometheus format: every non-comment line is "name[{labels}] value"
        for line in text.strip().splitlines():
            if not line.startswith("#"):
                assert len(line.rsplit(" ", 1)) == 2

    def test_speculation_metrics_rendered_when_engine_present(self):
        engine = ScriptedEngine(script=[1, 2, EOS_ID])

        class FakeSpec:
            drafted_tokens = 10
            accepted_tokens = 4

        engine.speculative_engine = FakeSpec()
        scheduler = self._run_one(engine)
        text = render_prometheus(scheduler)
        assert "winllm_spec_drafted_tokens_total 10" in text
        assert "winllm_spec_accepted_tokens_total 4" in text
        assert "winllm_spec_acceptance_rate 0.4" in text

    def test_no_speculation_metrics_without_engine(self):
        engine = ScriptedEngine(script=[1, EOS_ID])
        scheduler = self._run_one(engine)
        assert "winllm_spec_" not in render_prometheus(scheduler)

    def test_ttft_recorded_on_scheduled_path(self):
        engine = ScriptedEngine(script=[1, 2, EOS_ID])
        scheduler = self._run_one(engine)
        assert len(scheduler.stats.ttft) == 1
        # Scripted engine completes within a clock tick; just verify it's sane
        assert scheduler.stats.ttft.percentile(0.5) >= 0.0
