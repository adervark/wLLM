"""Unit tests for common data types (GenerationRequest, RequestStatus)."""

import time
import threading
import pytest
from winllm.types import GenerationRequest, RequestStatus
from winllm.config import SamplingParams


# ─── RequestStatus ──────────────────────────────────────────────────────────


class TestRequestStatus:
    def test_all_values(self):
        assert RequestStatus.PENDING.value == "pending"
        assert RequestStatus.RUNNING.value == "running"
        assert RequestStatus.COMPLETED.value == "completed"
        assert RequestStatus.FAILED.value == "failed"
        assert RequestStatus.CANCELLED.value == "cancelled"

    def test_str_enum_comparison(self):
        """RequestStatus is a str enum, so it should compare with strings."""
        assert RequestStatus.PENDING == "pending"
        assert RequestStatus.COMPLETED == "completed"


# ─── GenerationRequest defaults ─────────────────────────────────────────────


class TestGenerationRequestDefaults:
    def test_auto_generated_request_id(self):
        r = GenerationRequest()
        assert isinstance(r.request_id, str)
        assert len(r.request_id) == 12  # uuid hex[:12]

    def test_unique_request_ids(self):
        r1 = GenerationRequest()
        r2 = GenerationRequest()
        assert r1.request_id != r2.request_id

    def test_default_status(self):
        r = GenerationRequest()
        assert r.status == RequestStatus.PENDING

    def test_default_empty_lists(self):
        r = GenerationRequest()
        assert r.prompt_token_ids == []
        assert r.output_token_ids == []

    def test_default_timestamps(self):
        r = GenerationRequest()
        assert r.created_at > 0
        assert r.started_at is None
        assert r.finished_at is None

    def test_default_callbacks(self):
        r = GenerationRequest()
        assert r._stream_callback is None
        assert r._token_callback is None

    def test_default_internal_state(self):
        r = GenerationRequest()
        assert r._past_key_values is None
        assert r._prefix_cache_token_len == 0
        assert r._stream_text_cursor == 0
        assert r._prefix_past_key_values is None
        assert r._prefill_cursor == 0
        assert r._draft_past_key_values is None


# ─── is_prefill_complete ────────────────────────────────────────────────────


class TestIsPrefillComplete:
    def test_not_complete_when_no_tokens(self):
        r = GenerationRequest(prompt_token_ids=[1, 2, 3])
        r._prefill_cursor = 0
        assert not r.is_prefill_complete

    def test_not_complete_when_partial(self):
        r = GenerationRequest(prompt_token_ids=[1, 2, 3, 4, 5])
        r._prefill_cursor = 3
        assert not r.is_prefill_complete

    def test_complete_when_cursor_equals_length(self):
        r = GenerationRequest(prompt_token_ids=[1, 2, 3])
        r._prefill_cursor = 3
        assert r.is_prefill_complete

    def test_complete_when_cursor_exceeds_length(self):
        r = GenerationRequest(prompt_token_ids=[1, 2, 3])
        r._prefill_cursor = 5
        assert r.is_prefill_complete

    def test_complete_when_empty_prompt(self):
        r = GenerationRequest(prompt_token_ids=[])
        r._prefill_cursor = 0
        assert r.is_prefill_complete


# ─── Cancellation ───────────────────────────────────────────────────────────


class TestCancellation:
    def test_not_cancelled_by_default(self):
        r = GenerationRequest()
        assert not r.is_cancelled

    def test_cancel_sets_flag(self):
        r = GenerationRequest()
        r.cancel()
        assert r.is_cancelled

    def test_cancel_is_idempotent(self):
        r = GenerationRequest()
        r.cancel()
        r.cancel()
        assert r.is_cancelled

    def test_cancel_is_thread_safe(self):
        """Verify cancellation can be triggered from another thread safely."""
        r = GenerationRequest()
        results = []

        def cancel_from_thread():
            r.cancel()
            results.append(r.is_cancelled)

        t = threading.Thread(target=cancel_from_thread)
        t.start()
        t.join(timeout=2.0)

        assert results == [True]
        assert r.is_cancelled


# ─── Token count properties ────────────────────────────────────────────────


class TestTokenCounts:
    def test_total_tokens(self):
        r = GenerationRequest(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[4, 5],
        )
        assert r.total_tokens == 5

    def test_total_tokens_empty(self):
        r = GenerationRequest()
        assert r.total_tokens == 0

    def test_generation_tokens(self):
        r = GenerationRequest(output_token_ids=[10, 20, 30])
        assert r.generation_tokens == 3

    def test_generation_tokens_empty(self):
        r = GenerationRequest()
        assert r.generation_tokens == 0


# ─── Timing properties ─────────────────────────────────────────────────────


class TestTimingProperties:
    def test_elapsed_before_start(self):
        r = GenerationRequest()
        assert r.elapsed == 0.0

    def test_elapsed_running(self):
        r = GenerationRequest()
        r.started_at = time.time() - 1.0  # Started 1 second ago
        elapsed = r.elapsed
        assert 0.9 < elapsed < 2.0

    def test_elapsed_finished(self):
        r = GenerationRequest()
        r.started_at = 1000.0
        r.finished_at = 1002.5
        assert r.elapsed == 2.5

    def test_tokens_per_second_zero_elapsed(self):
        r = GenerationRequest()
        assert r.tokens_per_second == 0.0

    def test_tokens_per_second_calculation(self):
        r = GenerationRequest(output_token_ids=[1, 2, 3, 4, 5])
        r.started_at = 1000.0
        r.finished_at = 1001.0  # 1 second elapsed
        assert r.tokens_per_second == 5.0

    def test_tokens_per_second_no_output(self):
        r = GenerationRequest()
        r.started_at = 1000.0
        r.finished_at = 1001.0
        assert r.tokens_per_second == 0.0


# ─── SamplingParams integration ────────────────────────────────────────────


class TestRequestSamplingParams:
    def test_default_sampling_params(self):
        r = GenerationRequest()
        assert r.sampling_params.temperature == 0.7
        assert r.sampling_params.max_tokens == 512

    def test_custom_sampling_params(self):
        params = SamplingParams(temperature=0.0, max_tokens=100, top_k=10)
        r = GenerationRequest(sampling_params=params)
        assert r.sampling_params.temperature == 0.0
        assert r.sampling_params.max_tokens == 100
        assert r.sampling_params.top_k == 10
