"""OpenAI-compatible LLM adapter used by mem_flow."""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from collections.abc import Callable
from threading import Condition, Lock
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .config import LLMConfig
from .models import LLMRequest
from .observability import FlowMetrics

T = TypeVar("T")


class LLMConcurrencyLimiter:
    """Bound in-flight MaaS calls and optionally adapt capacity using AIMD.

    The limit applies only while an HTTP request is active. A request releases
    its slot before sleeping for a retry so one unhealthy call cannot block
    unrelated work. Adaptive mode decreases promptly on retryable service
    errors and increases cautiously after a sustained successful period. A
    decrease decision waits for one current-capacity cohort and requires the
    failed-worker ratio to exceed its configured threshold.
    """

    def __init__(
        self,
        *,
        max_limit: int,
        initial_limit: int | None = None,
        min_limit: int = 1,
        adaptive: bool = False,
        error_window: int = 20,
        failure_worker_ratio_threshold: float = 0.5,
        decrease_ratio: float = 0.7,
        increase_after_successes: int = 20,
        requests_per_minute: float | None = None,
    ) -> None:
        initial = max_limit if initial_limit is None else initial_limit
        if not 1 <= min_limit <= initial <= max_limit:
            raise ValueError("LLM concurrency must satisfy 1 <= min <= initial <= max")
        if error_window < 1 or increase_after_successes < 1:
            raise ValueError("adaptive concurrency windows must be positive")
        if not 0 < failure_worker_ratio_threshold <= 1:
            raise ValueError("failure_worker_ratio_threshold must be in (0, 1]")
        if not 0 < decrease_ratio < 1:
            raise ValueError("decrease_ratio must be in (0, 1)")
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive when set")
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.adaptive = adaptive
        self.failure_worker_ratio_threshold = failure_worker_ratio_threshold
        self.decrease_ratio = decrease_ratio
        self.increase_after_successes = increase_after_successes
        self.requests_per_minute = requests_per_minute
        self._request_interval = (
            60.0 / requests_per_minute if requests_per_minute is not None else 0.0
        )
        self._limit = initial
        self._active = 0
        self._condition = Condition()
        self._outcomes: deque[bool] = deque(maxlen=max(error_window, max_limit))
        self._success_streak = 0
        self._requests = 0
        self._errors = 0
        self._adjustments = 0
        self._cooldown_until = 0.0
        self._next_request_at = 0.0
        # ``Condition.notify_all`` alone does not provide ordering.  At low
        # adaptive limits a thread that just completed a fast extraction call
        # can repeatedly reacquire the only slot while older maintenance or
        # retrieval calls remain asleep.  Keep an explicit FIFO so a busy
        # pipeline makes progress at every stage instead of starving later
        # stages indefinitely.
        self._waiters: deque[object] = deque()

    def acquire(self) -> None:
        with self._condition:
            ticket = object()
            self._waiters.append(ticket)
            while True:
                now = time.monotonic()
                cooldown_remaining = self._cooldown_until - now
                rate_remaining = self._next_request_at - now
                is_next = self._waiters[0] is ticket
                if (
                    is_next
                    and self._active < self._limit
                    and cooldown_remaining <= 0
                    and rate_remaining <= 0
                ):
                    self._waiters.popleft()
                    self._active += 1
                    if self._request_interval:
                        self._next_request_at = now + self._request_interval
                    # When the adaptive limit is greater than one, let the
                    # next FIFO waiter claim another available slot promptly.
                    self._condition.notify_all()
                    return
                waits = [
                    value
                    for value in (cooldown_remaining, rate_remaining)
                    if value > 0
                ]
                self._condition.wait(timeout=min(waits) if waits else None)

    def release(
        self,
        *,
        success: bool,
        retryable: bool = True,
        cooldown_seconds: float = 0.0,
    ) -> None:
        with self._condition:
            if self._active < 1:
                raise RuntimeError("LLM concurrency limiter released without acquire")
            self._active -= 1
            self._record_outcome_locked(
                success=success,
                retryable=retryable,
                cooldown_seconds=cooldown_seconds,
            )
            self._condition.notify_all()

    def record_external_outcome(
        self,
        *,
        success: bool,
        retryable: bool = True,
        cooldown_seconds: float = 0.0,
    ) -> None:
        """Feed post-response validation failures into adaptive capacity.

        A structured response can complete at the HTTP layer and still be
        unusable.  Reporting that signal separately lets dynamic mode reduce
        pressure when MaaS concurrency causes truncated or malformed JSON,
        without pretending that another in-flight slot was released.
        """

        with self._condition:
            self._record_outcome_locked(
                success=success,
                retryable=retryable,
                cooldown_seconds=cooldown_seconds,
            )
            self._condition.notify_all()

    def _record_outcome_locked(
        self,
        *,
        success: bool,
        retryable: bool,
        cooldown_seconds: float,
    ) -> None:
        """Record one adaptive feedback signal while holding the condition."""

        self._requests += 1
        is_overload_error = not success and retryable
        self._outcomes.append(is_overload_error)
        reported_error_rate = sum(self._outcomes) / len(self._outcomes)
        if not success:
            self._errors += 1
        if is_overload_error and cooldown_seconds > 0:
            self._cooldown_until = max(
                self._cooldown_until,
                time.monotonic() + cooldown_seconds,
            )

        previous = self._limit
        reason = ""
        if not self.adaptive:
            pass
        elif is_overload_error:
            self._success_streak = 0
            worker_outcomes = list(self._outcomes)[-self._limit :]
            failed_worker_ratio = sum(worker_outcomes) / len(worker_outcomes)
            if (
                len(worker_outcomes) >= self._limit
                and failed_worker_ratio > self.failure_worker_ratio_threshold
            ):
                reduced = math.floor(self._limit * self.decrease_ratio)
                self._limit = max(self.min_limit, min(self._limit - 1, reduced))
                reason = "failed_worker_ratio"
                self._outcomes.clear()
        elif success:
            self._success_streak += 1
            error_rate = sum(self._outcomes) / len(self._outcomes)
            if (
                self._success_streak >= self.increase_after_successes
                and error_rate <= self.failure_worker_ratio_threshold
                and self._limit < self.max_limit
            ):
                self._limit += 1
                self._success_streak = 0
                reason = "success_streak"
        else:
            self._success_streak = 0

        if self._limit != previous:
            self._adjustments += 1
            logging.getLogger("mem_flow.llm").warning(
                "llm_concurrency_adjusted previous=%d current=%d reason=%s "
                "recent_error_rate=%.3f",
                previous,
                self._limit,
                reason,
                reported_error_rate,
            )

    def snapshot(self) -> dict[str, int | float | bool]:
        with self._condition:
            recent_error_rate = (
                sum(self._outcomes) / len(self._outcomes) if self._outcomes else 0.0
            )
            worker_outcomes = list(self._outcomes)[-self._limit :]
            failed_worker_ratio = (
                sum(worker_outcomes) / len(worker_outcomes)
                if worker_outcomes
                else 0.0
            )
            return {
                "adaptive": self.adaptive,
                "min_workers": self.min_limit,
                "current_workers": self._limit,
                "max_workers": self.max_limit,
                "requests_per_minute": self.requests_per_minute or 0.0,
                "active_requests": self._active,
                "waiting_requests": len(self._waiters),
                "requests": self._requests,
                "errors": self._errors,
                "recent_error_rate": round(recent_error_rate, 4),
                "failed_worker_ratio": round(failed_worker_ratio, 4),
                "failure_worker_ratio_threshold": self.failure_worker_ratio_threshold,
                "adjustments": self._adjustments,
                "cooldown_remaining_seconds": round(
                    max(0.0, self._cooldown_until - time.monotonic()), 3
                ),
            }


def is_retryable_llm_error(error: BaseException) -> bool:
    """Classify transport/service errors without coupling to one SDK version."""

    status = getattr(error, "status_code", None)
    if status is None:
        # Connection/timeouts, empty responses, and invalid structured model
        # output do not reliably expose an HTTP status and are safe to retry.
        return True
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        return True
    # The MaaS gateway intermittently reports overloaded requests as HTTP 400
    # InvalidParameter, so 400 remains retryable but is bounded by the explicit
    # attempt and backoff caps below. Authentication and not-found are terminal.
    return (
        status_code == 400 or status_code in {408, 409, 425, 429} or status_code >= 500
    )


def retry_wait_seconds(
    config: object,
    attempt: int,
    error: BaseException | None = None,
    *,
    random_value: Callable[[], float] = random.random,
) -> float:
    """Return capped exponential backoff with jitter and Retry-After support."""

    initial = max(0.0, float(getattr(config, "retry_initial_seconds", 0.0)))
    maximum = max(initial, float(getattr(config, "retry_max_seconds", initial)))
    jitter = min(1.0, max(0.0, float(getattr(config, "retry_jitter", 0.0))))
    wait = min(maximum, initial * (2 ** max(0, attempt - 1)))
    if error is not None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                wait = max(wait, float(headers.get("retry-after", 0) or 0))
            except (TypeError, ValueError):
                pass
    if jitter and wait:
        wait *= 1 - jitter + (2 * jitter * random_value())
    return min(maximum, max(0.0, wait))


@runtime_checkable
class FlowLLM(Protocol):
    def complete(self, request: LLMRequest) -> str: ...


def complete_structured(
    llm: FlowLLM,
    request: LLMRequest,
    validator: Callable[[str], T],
) -> T:
    """Retry model responses that fail downstream structured validation.

    ``OpenAIFlowLLM.complete`` already owns transport and empty-response retry.
    This boundary adds retries only after a non-empty response was returned but
    could not be parsed or failed a conservation check.  That distinction keeps
    API failures from accidentally multiplying the configured retry budget.
    """

    config = getattr(llm, "config", None)
    attempts = max(1, int(getattr(config, "structured_retry_attempts", 1)))
    logger = logging.getLogger("mem_flow.llm")
    for attempt in range(1, attempts + 1):
        raw = llm.complete(request)
        try:
            return validator(raw)
        except Exception as error:
            limiter = getattr(llm, "limiter", None)
            if isinstance(limiter, LLMConcurrencyLimiter):
                limiter.record_external_outcome(success=False, retryable=True)
            if attempt >= attempts:
                logger.error(
                    "llm_structured_failed operation=%s attempts=%d "
                    "error_type=%s error=%s",
                    request.operation,
                    attempt,
                    type(error).__name__,
                    error,
                )
                raise
            wait = retry_wait_seconds(config, attempt, error)
            logger.warning(
                "llm_structured_retry operation=%s attempt=%d wait_seconds=%.2f "
                "error_type=%s error=%s",
                request.operation,
                attempt,
                wait,
                type(error).__name__,
                error,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


class OpenAIFlowLLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: LLMConfig
    metrics: FlowMetrics
    client: object
    limiter: LLMConcurrencyLimiter
    _langchain_model: object | None = PrivateAttr(default=None)
    _langchain_model_lock: Lock = PrivateAttr(default_factory=Lock)

    @classmethod
    def create(
        cls,
        config: LLMConfig,
        metrics: FlowMetrics,
        *,
        client: object | None = None,
        limiter: LLMConcurrencyLimiter | None = None,
    ) -> "OpenAIFlowLLM":
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.base_url or None,
                # Keep one explicit retry policy. The SDK's hidden retries made
                # a single failed request block before this loop ran.
                max_retries=0,
                timeout=config.request_timeout_seconds,
            )
        limiter = limiter or LLMConcurrencyLimiter(max_limit=config.max_concurrency)
        return cls(config=config, metrics=metrics, client=client, limiter=limiter)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.llm")

    def complete(self, request: LLMRequest) -> str:
        self.logger.info(
            "llm_started operation=%s model=%s messages=%d",
            request.operation,
            self.config.model,
            len(request.messages),
        )
        started = time.monotonic()
        for attempt in range(1, self.config.retry_attempts + 1):
            self.limiter.acquire()
            try:
                kwargs: dict[str, object] = {
                    "model": self.config.model,
                    "messages": [message.model_dump() for message in request.messages],
                }
                temperature = (
                    request.temperature
                    if request.temperature is not None
                    else self.config.temperature
                )
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = self.client.chat.completions.create(**kwargs)
                content = (
                    response.choices[0].message.content if response.choices else ""
                )
                if not content:
                    raise RuntimeError("LLM returned empty content")
            except BaseException as error:
                if not isinstance(error, Exception):
                    self.limiter.release(success=False, retryable=False)
                    raise
                retryable = is_retryable_llm_error(error)
                wait = (
                    retry_wait_seconds(self.config, attempt, error)
                    if retryable
                    else 0.0
                )
                # A retryable MaaS overload applies to the shared endpoint, not
                # just this thread.  Publish the same backoff to the limiter so
                # queued calls do not immediately replace the failed request
                # and continuously hammer the service at concurrency one.
                self.limiter.release(
                    success=False,
                    retryable=retryable,
                    cooldown_seconds=wait,
                )
                if attempt >= self.config.retry_attempts or not retryable:
                    self.metrics.llm_total.labels(
                        operation=request.operation, status="error"
                    ).inc()
                    self.logger.exception(
                        "llm_failed operation=%s model=%s attempts=%d",
                        request.operation,
                        self.config.model,
                        attempt,
                    )
                    raise
                self.logger.warning(
                    "llm_retry operation=%s attempt=%d wait_seconds=%.2f error_type=%s",
                    request.operation,
                    attempt,
                    wait,
                    type(error).__name__,
                )
                time.sleep(wait)
                continue
            self.limiter.release(success=True)
            elapsed = time.monotonic() - started
            self.metrics.llm_total.labels(
                operation=request.operation, status="success"
            ).inc()
            self.logger.info(
                "llm_completed operation=%s model=%s duration_seconds=%.3f attempt=%d",
                request.operation,
                self.config.model,
                elapsed,
                attempt,
            )
            return str(content).strip()
        raise AssertionError("unreachable")

    def as_langchain_model(self) -> object:
        """Build the explicitly configured model used by Agentic retrieval."""

        with self._langchain_model_lock:
            if self._langchain_model is not None:
                return self._langchain_model

            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "AGENTIC retrieval requires infini-memory[deepagents]"
                ) from exc
            flow_config = self.config
            limiter = self.limiter
            metrics = self.metrics
            logger = self.logger

            class LimitedChatOpenAI(ChatOpenAI):
                """Route LangChain agent calls through the shared MaaS limiter."""

                def _generate(self, *args: object, **kwargs: object) -> object:
                    started = time.monotonic()
                    for attempt in range(1, flow_config.retry_attempts + 1):
                        limiter.acquire()
                        try:
                            result = super()._generate(*args, **kwargs)
                        except BaseException as error:
                            if not isinstance(error, Exception):
                                limiter.release(success=False, retryable=False)
                                raise
                            retryable = is_retryable_llm_error(error)
                            wait = (
                                retry_wait_seconds(flow_config, attempt, error)
                                if retryable
                                else 0.0
                            )
                            limiter.release(
                                success=False,
                                retryable=retryable,
                                cooldown_seconds=wait,
                            )
                            if attempt >= flow_config.retry_attempts or not retryable:
                                metrics.llm_total.labels(
                                    operation="agentic_retrieve", status="error"
                                ).inc()
                                logger.exception(
                                    "llm_failed operation=agentic_retrieve model=%s attempts=%d",
                                    flow_config.model,
                                    attempt,
                                )
                                raise
                            logger.warning(
                                "llm_retry operation=agentic_retrieve attempt=%d "
                                "wait_seconds=%.2f error_type=%s",
                                attempt,
                                wait,
                                type(error).__name__,
                            )
                            time.sleep(wait)
                            continue
                        limiter.release(success=True)
                        metrics.llm_total.labels(
                            operation="agentic_retrieve", status="success"
                        ).inc()
                        logger.info(
                            "llm_completed operation=agentic_retrieve model=%s "
                            "duration_seconds=%.3f attempt=%d",
                            flow_config.model,
                            time.monotonic() - started,
                            attempt,
                        )
                        return result
                    raise AssertionError("unreachable")

            self._langchain_model = LimitedChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url or None,
                temperature=self.config.temperature,
                max_retries=0,
                timeout=self.config.request_timeout_seconds,
                disable_streaming=True,
            )
            return self._langchain_model


__all__ = [
    "FlowLLM",
    "LLMConcurrencyLimiter",
    "OpenAIFlowLLM",
    "complete_structured",
    "is_retryable_llm_error",
    "retry_wait_seconds",
]
