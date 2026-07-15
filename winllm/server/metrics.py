"""Prometheus text-format metrics rendering.

Hand-rolls the (stable, line-oriented) exposition format rather than
pulling in prometheus-client for a handful of gauges and counters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..scheduling.scheduler import Scheduler

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_QUANTILES = (0.5, 0.9, 0.99)


def _metric(lines: list[str], name: str, kind: str, help_text: str, value) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {kind}")
    lines.append(f"{name} {value}")


def _summary(lines: list[str], name: str, help_text: str, window) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} summary")
    for q in _QUANTILES:
        lines.append(f'{name}{{quantile="{q}"}} {window.percentile(q):.6f}')


def render_prometheus(scheduler: "Scheduler") -> str:
    """Render scheduler + KV cache metrics in Prometheus text format."""
    stats = scheduler.stats
    lines: list[str] = []

    _metric(lines, "winllm_requests_total", "counter",
            "Total requests submitted.", stats.total_requests)
    _metric(lines, "winllm_requests_completed_total", "counter",
            "Requests completed successfully.", stats.completed_requests)
    _metric(lines, "winllm_requests_failed_total", "counter",
            "Requests that failed.", stats.failed_requests)
    _metric(lines, "winllm_prompt_tokens_total", "counter",
            "Prompt tokens processed.", stats.total_prompt_tokens)
    _metric(lines, "winllm_generation_tokens_total", "counter",
            "Tokens generated.", stats.total_generation_tokens)

    _metric(lines, "winllm_requests_waiting", "gauge",
            "Requests queued for admission.", scheduler.num_waiting)
    _metric(lines, "winllm_requests_running", "gauge",
            "Requests in the active batch.", scheduler.num_running)
    _metric(lines, "winllm_avg_tokens_per_second", "gauge",
            "Lifetime average generation throughput.",
            round(stats.avg_tokens_per_second, 3))

    _summary(lines, "winllm_ttft_seconds",
             "Time to first token (includes queue wait), recent window.",
             stats.ttft)
    _summary(lines, "winllm_e2e_latency_seconds",
             "End-to-end request latency, recent window.",
             stats.e2e_latency)
    _summary(lines, "winllm_request_tokens_per_second",
             "Per-request decode throughput, recent window.",
             stats.request_tokens_per_second)

    spec = getattr(scheduler.engine, "speculative_engine", None)
    if spec is not None:
        drafted = getattr(spec, "drafted_tokens", 0)
        accepted = getattr(spec, "accepted_tokens", 0)
        _metric(lines, "winllm_spec_drafted_tokens_total", "counter",
                "Speculative draft tokens proposed.", drafted)
        _metric(lines, "winllm_spec_accepted_tokens_total", "counter",
                "Speculative draft tokens accepted by the target model.", accepted)
        _metric(lines, "winllm_spec_acceptance_rate", "gauge",
                "Fraction of drafted tokens accepted (lifetime).",
                round(accepted / drafted, 4) if drafted else 0.0)

    kv = scheduler.engine.kv_cache_manager
    if kv is not None:
        kv_stats = kv.get_stats()
        _metric(lines, "winllm_kv_cache_blocks_total", "gauge",
                "Total KV cache blocks.", kv_stats.get("total_blocks", 0))
        _metric(lines, "winllm_kv_cache_blocks_free", "gauge",
                "Free KV cache blocks.", kv_stats.get("free_blocks", 0))
        _metric(lines, "winllm_kv_cache_utilization", "gauge",
                "KV cache utilization fraction.", kv_stats.get("utilization", 0.0))
        _metric(lines, "winllm_active_sequences", "gauge",
                "Sequences with live KV allocations.", kv_stats.get("active_sequences", 0))

    return "\n".join(lines) + "\n"
