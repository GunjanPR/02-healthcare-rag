"""Prometheus metrics and OpenTelemetry tracing.

Metrics are served at /metrics. With OTEL_EXPORTER_OTLP_ENDPOINT set, every request is
traced: the FastAPI server span, one span per RAG stage (retrieve, rerank, generate,
verify) and every httpx call. The httpx calls to the LLM gateway carry the W3C
`traceparent` header, and the gateway exports to the same collector (Jaeger), so a single
trace runs from /ask through the gateway to Ollama.
"""

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from prometheus_client import CollectorRegistry, Counter, Histogram

from healthcare_rag import __version__

logger = logging.getLogger(__name__)

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 60)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        r = self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "rag_requests_total", "API requests", ["endpoint", "outcome"], registry=r
        )
        self.latency = Histogram(
            "rag_request_latency_seconds",
            "End-to-end latency of /ask and /summarize",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.stage_latency = Histogram(
            "rag_stage_latency_seconds",
            "Latency per RAG stage",
            ["stage"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.answers = Counter(
            "rag_answers_total",
            "Outcome of /ask: answered or the abstention reason",
            ["reason"],
            registry=r,
        )
        self.tokens = Counter("rag_tokens_total", "LLM tokens", ["type"], registry=r)
        self.cost = Counter("rag_cost_usd_total", "LLM cost reported by the gateway", registry=r)
        self.llm_calls = Counter("rag_llm_calls_total", "Chat calls to the gateway", registry=r)
        self.citations = Histogram(
            "rag_citations_per_answer",
            "Passages cited per answer",
            buckets=(0, 1, 2, 3, 4, 5),
            registry=r,
        )
        self.top_score = Histogram(
            "rag_rerank_top_score",
            "Best reranker score per question (MedCPT logit)",
            buckets=(-10, -5, -2, 0, 2, 4, 6, 8, 10, 15),
            registry=r,
        )

    def observe_result(self, endpoint: str, result: Any) -> None:
        """Record one RAGResult."""
        self.requests.labels(endpoint, "ok").inc()
        self.latency.labels(endpoint).observe(result.timings.get("total", 0.0))
        for stage, seconds in result.timings.items():
            if stage != "total":
                self.stage_latency.labels(stage).observe(seconds)
        u = result.usage
        self.tokens.labels("prompt").inc(u.prompt_tokens)
        self.tokens.labels("completion").inc(u.completion_tokens)
        self.cost.inc(u.cost_usd)
        self.llm_calls.inc(u.llm_calls)
        if endpoint == "ask":
            self.answers.labels(result.reason).inc()
            if result.top_score is not None:
                self.top_score.observe(result.top_score)
            if not result.abstained:
                self.citations.observe(len(result.cited))


metrics = Metrics()

_provider: TracerProvider | None = None


def setup_tracing(
    app: Any, endpoint: str, service_name: str, exporter: SpanExporter | None = None
) -> TracerProvider | None:
    """Instrument FastAPI and httpx. `exporter` is for tests (exported synchronously)."""
    global _provider
    if not endpoint and exporter is None:
        return None
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.version": __version__})
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        url = endpoint.rstrip("/")
        if not url.endswith("/v1/traces"):
            url += "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
    if _provider is None:
        trace.set_tracer_provider(provider)
    _provider = provider
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls="health,ready,metrics"
    )
    instrumentor = HTTPXClientInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=provider)
    logger.info("tracing enabled for %s -> %s", service_name, endpoint or "test exporter")
    return provider


def get_tracer() -> trace.Tracer:
    if _provider is not None:
        return _provider.get_tracer("healthcare_rag", __version__)
    return trace.get_tracer("healthcare_rag", __version__)
