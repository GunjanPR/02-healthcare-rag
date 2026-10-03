import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from fakes import FakeChat, FakeReranker, FakeRetriever, hit
from healthcare_rag import main, observability
from healthcare_rag.factory import get_pipeline
from healthcare_rag.observability import Metrics
from healthcare_rag.rag import RAGConfig, RAGPipeline


def _value(registry: CollectorRegistry, name: str, labels: dict | None = None) -> float:
    return registry.get_sample_value(name, labels or {}) or 0.0


def test_observe_result_records_outcome_tokens_and_stages() -> None:
    m = Metrics(CollectorRegistry())
    pipe = RAGPipeline(
        FakeRetriever([hit(1), hit(2)]),
        FakeChat("Answer [1]."),
        FakeReranker([5.0, 1.0]),
        RAGConfig(abstain_threshold=0.0),
    )
    m.observe_result("ask", pipe.ask("q?"))
    weak = RAGPipeline(
        FakeRetriever([hit(1)]), FakeChat("x"), FakeReranker([-3.0]), RAGConfig()
    ).ask("q?")
    m.observe_result("ask", weak)

    r = m.registry
    assert _value(r, "rag_requests_total", {"endpoint": "ask", "outcome": "ok"}) == 2
    assert _value(r, "rag_answers_total", {"reason": "answered"}) == 1
    assert _value(r, "rag_answers_total", {"reason": "low_score"}) == 1
    assert _value(r, "rag_tokens_total", {"type": "prompt"}) == 100
    assert _value(r, "rag_llm_calls_total") == 1
    assert _value(r, "rag_stage_latency_seconds_count", {"stage": "generate"}) == 1
    assert _value(r, "rag_stage_latency_seconds_count", {"stage": "retrieve"}) == 2
    assert _value(r, "rag_rerank_top_score_count") == 2
    assert _value(r, "rag_citations_per_answer_sum") == 1


def test_metrics_endpoint_exposes_rag_metrics() -> None:
    main.app.dependency_overrides[get_pipeline] = lambda: RAGPipeline(
        FakeRetriever([hit(1)]), FakeChat("Answer [1].")
    )
    try:
        client = TestClient(main.app)
        assert client.post("/ask", json={"question": "what is it?"}).status_code == 200
        body = client.get("/metrics").text
    finally:
        main.app.dependency_overrides.clear()
    assert 'rag_requests_total{endpoint="ask",outcome="ok"}' in body
    assert "rag_request_latency_seconds_bucket" in body


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_provider", provider)
    return exporter


def test_ask_emits_one_trace_with_a_span_per_stage(spans: InMemorySpanExporter) -> None:
    pipe = RAGPipeline(
        FakeRetriever([hit(1), hit(2)]),
        FakeChat("Answer [1]."),
        FakeReranker([4.0, 1.0]),
        RAGConfig(abstain_threshold=0.0),
    )
    pipe.ask("q?")
    by_name = {s.name: s for s in spans.get_finished_spans()}
    assert {"rag.ask", "rag.retrieve", "rag.rerank", "rag.generate"} <= set(by_name)
    root = by_name["rag.ask"]
    for name in ("rag.retrieve", "rag.rerank", "rag.generate"):
        assert by_name[name].parent.span_id == root.context.span_id
        assert by_name[name].context.trace_id == root.context.trace_id
    assert by_name["rag.generate"].attributes["gen_ai.usage.input_tokens"] == 100
    assert by_name["rag.rerank"].attributes["rag.rerank.top_score"] == 4.0
    assert root.attributes["rag.reason"] == "answered"


def test_hybrid_workers_keep_the_trace_context(spans: InMemorySpanExporter) -> None:
    from healthcare_rag.hybrid import HybridRetriever

    class TracedRetriever:
        def search(self, query, k=5, filters=None):
            with observability.get_tracer().start_as_current_span("child.search"):
                return [hit(1)]

    with observability.get_tracer().start_as_current_span("parent") as parent:
        HybridRetriever(TracedRetriever(), TracedRetriever()).search("q", k=1)
    children = [s for s in spans.get_finished_spans() if s.name == "child.search"]
    assert len(children) == 2
    assert all(s.parent.span_id == parent.context.span_id for s in children)
