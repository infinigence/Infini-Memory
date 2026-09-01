"""Structured logging helpers and Prometheus metrics for every flow."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from pydantic import BaseModel, ConfigDict


_METRICS_CACHE: dict[tuple[str, int], "FlowMetrics"] = {}


class _NoopMetric:
    def labels(self, **_labels: str) -> "_NoopMetric":
        return self

    def inc(self, _amount: float = 1) -> None:
        return None

    def observe(self, _amount: float) -> None:
        return None


class FlowMetrics(BaseModel):
    """Metric bundle. A custom registry can be supplied by tests or applications."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    flow_total: object
    flow_duration: object
    documents_total: object
    llm_total: object
    s3_total: object

    @classmethod
    def create(
        cls,
        *,
        namespace: str = "mem_flow",
        enabled: bool = True,
        registry: object | None = None,
    ) -> "FlowMetrics":
        if not enabled:
            noop = _NoopMetric()
            return cls(
                flow_total=noop,
                flow_duration=noop,
                documents_total=noop,
                llm_total=noop,
                s3_total=noop,
            )
        try:
            from prometheus_client import Counter, Histogram, REGISTRY
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RuntimeError(
                "prometheus-client is required when metrics are enabled"
            ) from exc
        target_registry = REGISTRY if registry is None else registry
        cache_key = (namespace, id(target_registry))
        if cache_key in _METRICS_CACHE:
            return _METRICS_CACHE[cache_key]
        metrics = cls(
            flow_total=Counter(
                "flow_total",
                "Completed mem_flow operations",
                ["module", "flow", "status"],
                namespace=namespace,
                registry=target_registry,
            ),
            flow_duration=Histogram(
                "flow_duration_seconds",
                "mem_flow operation duration",
                ["module", "flow"],
                namespace=namespace,
                registry=target_registry,
            ),
            documents_total=Counter(
                "documents_total",
                "Documents processed by mem_flow",
                ["module", "flow", "kind"],
                namespace=namespace,
                registry=target_registry,
            ),
            llm_total=Counter(
                "llm_calls_total",
                "LLM calls made by mem_flow",
                ["operation", "status"],
                namespace=namespace,
                registry=target_registry,
            ),
            s3_total=Counter(
                "s3_operations_total",
                "S3 operations made by mem_flow",
                ["operation", "status"],
                namespace=namespace,
                registry=target_registry,
            ),
        )
        _METRICS_CACHE[cache_key] = metrics
        return metrics


@contextmanager
def observe_flow(
    metrics: FlowMetrics,
    logger: logging.Logger,
    *,
    module: str,
    flow: str,
    context: dict[str, object] | None = None,
) -> Iterator[None]:
    started = time.monotonic()
    extra = context or {}
    logger.info("flow_started module=%s flow=%s context=%s", module, flow, extra)
    try:
        yield
    except Exception:
        elapsed = time.monotonic() - started
        metrics.flow_total.labels(module=module, flow=flow, status="error").inc()
        metrics.flow_duration.labels(module=module, flow=flow).observe(elapsed)
        logger.exception(
            "flow_failed module=%s flow=%s duration_seconds=%.3f context=%s",
            module,
            flow,
            elapsed,
            extra,
        )
        raise
    else:
        elapsed = time.monotonic() - started
        metrics.flow_total.labels(module=module, flow=flow, status="success").inc()
        metrics.flow_duration.labels(module=module, flow=flow).observe(elapsed)
        logger.info(
            "flow_completed module=%s flow=%s duration_seconds=%.3f context=%s",
            module,
            flow,
            elapsed,
            extra,
        )


def start_metrics_server(port: int, address: str = "0.0.0.0") -> None:
    """Expose the default Prometheus registry over HTTP."""

    from prometheus_client import start_http_server

    start_http_server(port, addr=address)


__all__ = ["FlowMetrics", "observe_flow", "start_metrics_server"]
