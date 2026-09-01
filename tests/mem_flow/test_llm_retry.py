from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest

from mem_flow.config import LLMConfig, StorageConfig
from mem_flow.llm import (
    LLMConcurrencyLimiter,
    complete_structured,
    is_retryable_llm_error,
    retry_wait_seconds,
)
from mem_flow.models import ChatMessage, LLMRequest
from mem_flow.observability import FlowMetrics


def test_mem_flow_defaults_to_s3_and_five_llm_attempts() -> None:
    assert StorageConfig().type == "s3"
    assert LLMConfig().retry_attempts == 5
    assert LLMConfig().structured_retry_attempts == 2
    assert LLMConfig().request_timeout_seconds == 300.0
    assert LLMConfig().retry_max_seconds == 8.0
    assert LLMConfig().max_concurrency == 50


def test_adaptive_limiter_decreases_on_errors_and_recovers_cautiously() -> None:
    limiter = LLMConcurrencyLimiter(
        min_limit=1,
        initial_limit=3,
        max_limit=5,
        adaptive=True,
        error_window=5,
        failure_worker_ratio_threshold=0.5,
        decrease_ratio=0.7,
        increase_after_successes=2,
    )

    limiter.acquire()
    limiter.release(success=False, retryable=True)
    assert limiter.snapshot()["current_workers"] == 3
    limiter.acquire()
    limiter.release(success=False, retryable=True)
    assert limiter.snapshot()["current_workers"] == 3
    limiter.acquire()
    limiter.release(success=False, retryable=True)
    assert limiter.snapshot()["current_workers"] == 2

    for _ in range(2):
        limiter.acquire()
        limiter.release(success=True)

    snapshot = limiter.snapshot()
    assert snapshot["current_workers"] == 3
    assert snapshot["requests"] == 5
    assert snapshot["errors"] == 3
    assert snapshot["adjustments"] == 2


def test_adaptive_limiter_accepts_structured_validation_feedback() -> None:
    limiter = LLMConcurrencyLimiter(
        min_limit=1,
        initial_limit=3,
        max_limit=5,
        adaptive=True,
        error_window=4,
        failure_worker_ratio_threshold=0.5,
    )

    limiter.record_external_outcome(success=False, retryable=True)
    limiter.record_external_outcome(success=False, retryable=True)

    snapshot = limiter.snapshot()
    assert snapshot["current_workers"] == 3

    limiter.record_external_outcome(success=False, retryable=True)

    snapshot = limiter.snapshot()
    assert snapshot["current_workers"] == 2
    assert snapshot["requests"] == 3
    assert snapshot["errors"] == 3
    assert snapshot["active_requests"] == 0


def test_limiter_bounds_parallel_requests() -> None:
    limiter = LLMConcurrencyLimiter(max_limit=2)
    lock = Lock()
    active = 0
    maximum_active = 0

    def request() -> None:
        nonlocal active, maximum_active
        limiter.acquire()
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        limiter.release(success=True)

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _index: request(), range(6)))

    assert maximum_active == 2
    assert limiter.snapshot()["active_requests"] == 0


def test_retryable_error_applies_shared_cooldown() -> None:
    limiter = LLMConcurrencyLimiter(max_limit=1)

    limiter.acquire()
    limiter.release(
        success=False,
        retryable=True,
        cooldown_seconds=0.03,
    )
    started = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - started
    limiter.release(success=True)

    assert elapsed >= 0.02
    assert limiter.snapshot()["cooldown_remaining_seconds"] == 0.0


def test_limiter_serves_waiters_in_fifo_order() -> None:
    limiter = LLMConcurrencyLimiter(max_limit=1)
    limiter.acquire()
    acquired: list[int] = []
    ready = [Event() for _ in range(3)]

    def request(index: int) -> None:
        ready[index].set()
        limiter.acquire()
        acquired.append(index)
        limiter.release(success=True)

    threads: list[Thread] = []
    for index in range(3):
        thread = Thread(target=request, args=(index,))
        thread.start()
        threads.append(thread)
        assert ready[index].wait(timeout=1)
        deadline = time.monotonic() + 1
        while limiter.snapshot()["waiting_requests"] < index + 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)

    limiter.release(success=True)
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    assert acquired == [0, 1, 2]
    assert limiter.snapshot()["waiting_requests"] == 0


def test_limiter_spaces_request_starts_when_rate_is_configured() -> None:
    limiter = LLMConcurrencyLimiter(max_limit=2, requests_per_minute=1200)

    started = time.monotonic()
    limiter.acquire()
    limiter.release(success=True)
    limiter.acquire()
    elapsed = time.monotonic() - started
    limiter.release(success=True)

    assert elapsed >= 0.04
    assert limiter.snapshot()["requests_per_minute"] == 1200


def test_retry_wait_is_capped_and_http_classification_handles_maas_400() -> None:
    config = LLMConfig(
        retry_initial_seconds=2,
        retry_max_seconds=5,
        retry_jitter=0.5,
    )

    assert retry_wait_seconds(config, 10, random_value=lambda: 0.5) == 5

    class HTTPError(RuntimeError):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    assert is_retryable_llm_error(HTTPError(400))
    assert is_retryable_llm_error(HTTPError(429))
    assert not is_retryable_llm_error(HTTPError(401))


def test_openai_adapter_releases_slot_before_retry() -> None:
    from mem_flow.llm import OpenAIFlowLLM

    class Completions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                error = RuntimeError("overloaded")
                error.status_code = 400  # type: ignore[attr-defined]
                raise error
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    completions = Completions()
    limiter = LLMConcurrencyLimiter(
        min_limit=1,
        initial_limit=2,
        max_limit=2,
        adaptive=True,
    )
    llm = OpenAIFlowLLM.create(
        LLMConfig(retry_attempts=2, retry_initial_seconds=0),
        FlowMetrics.create(enabled=False),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        limiter=limiter,
    )

    result = llm.complete(
        LLMRequest(
            operation="retry_test",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert result == "ok"
    assert completions.calls == 2
    assert limiter.snapshot()["active_requests"] == 0
    # One failed request is not a full two-worker cohort, so it must not
    # trigger the majority-failure reduction policy.
    assert limiter.snapshot()["current_workers"] == 2


def test_langchain_adapter_uses_shared_limiter(monkeypatch) -> None:
    from langchain_openai import ChatOpenAI

    from mem_flow.llm import OpenAIFlowLLM

    monkeypatch.setattr(
        ChatOpenAI,
        "_generate",
        lambda _self, *_args, **_kwargs: "agent-result",
    )
    limiter = LLMConcurrencyLimiter(max_limit=2)
    llm = OpenAIFlowLLM.create(
        LLMConfig(api_key="test-key", base_url="https://llm.example.test/v1"),
        FlowMetrics.create(enabled=False),
        client=SimpleNamespace(),
        limiter=limiter,
    )

    model = llm.as_langchain_model()
    assert llm.as_langchain_model() is model
    assert model._generate([]) == "agent-result"
    assert limiter.snapshot()["requests"] == 1
    assert limiter.snapshot()["errors"] == 0
    assert limiter.snapshot()["active_requests"] == 0


def test_complete_structured_retries_invalid_model_output() -> None:
    class InvalidThenValidLLM:
        config = LLMConfig(
            retry_attempts=5,
            structured_retry_attempts=3,
            retry_initial_seconds=0,
        )

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _request: LLMRequest) -> str:
            self.calls += 1
            return "not json" if self.calls < 3 else '{"answer": "ok"}'

    llm = InvalidThenValidLLM()
    request = LLMRequest(
        operation="structured_test",
        messages=[ChatMessage(role="user", content="return json")],
    )

    result = complete_structured(llm, request, json.loads)

    assert result == {"answer": "ok"}
    assert llm.calls == 3


def test_complete_structured_reports_validation_failure_to_limiter() -> None:
    class InvalidThenValidLLM:
        config = LLMConfig(
            structured_retry_attempts=2,
            retry_initial_seconds=0,
        )

        def __init__(self) -> None:
            self.calls = 0
            self.limiter = LLMConcurrencyLimiter(
                min_limit=1,
                initial_limit=3,
                max_limit=3,
                adaptive=True,
            )

        def complete(self, _request: LLMRequest) -> str:
            self.calls += 1
            return "not json" if self.calls == 1 else '{"answer": "ok"}'

    llm = InvalidThenValidLLM()
    result = complete_structured(
        llm,
        LLMRequest(
            operation="structured_test",
            messages=[ChatMessage(role="user", content="return json")],
        ),
        json.loads,
    )

    assert result == {"answer": "ok"}
    # A single invalid structured response is recorded, but the limiter waits
    # for a full three-worker cohort before deciding whether failures exceed 50%.
    assert llm.limiter.snapshot()["current_workers"] == 3


def test_complete_structured_does_not_multiply_transport_failures() -> None:
    class FailedLLM:
        config = LLMConfig(retry_attempts=5, retry_initial_seconds=0)

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _request: LLMRequest) -> str:
            self.calls += 1
            raise RuntimeError("transport retries already exhausted")

    llm = FailedLLM()
    request = LLMRequest(
        operation="structured_test",
        messages=[ChatMessage(role="user", content="return json")],
    )

    with pytest.raises(RuntimeError, match="transport retries already exhausted"):
        complete_structured(llm, request, json.loads)

    assert llm.calls == 1
