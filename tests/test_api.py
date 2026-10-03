from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from fakes import FakeChat, FakeRetriever, hit
from healthcare_rag import main
from healthcare_rag.config import Settings, get_settings
from healthcare_rag.factory import get_index, get_pipeline
from healthcare_rag.gateway import GatewayError
from healthcare_rag.rag import RAGPipeline


@pytest.fixture
def client() -> Iterator[TestClient]:
    main.app.dependency_overrides[get_settings] = lambda: Settings(
        opensearch_url="http://opensearch.test:9200",
        qdrant_url="http://qdrant.test:6333",
        gateway_base_url="http://gateway.test:8000/v1",
    )
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_health_reports_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "healthcare-rag", "version": "0.1.0"}


def _fake_get(statuses: dict[str, int | Exception]):
    def fake_get(url: str, **_: object) -> httpx.Response:
        for host, outcome in statuses.items():
            if host in url:
                if isinstance(outcome, Exception):
                    raise outcome
                return httpx.Response(outcome, request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected probe {url}")

    return fake_get


def test_ready_when_backends_up(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake = _fake_get({"opensearch.test": 200, "qdrant.test": 200, "gateway.test": 200})

    def recording_get(url: str, **kw: object) -> httpx.Response:
        calls.append(url)
        return fake(url, **kw)

    monkeypatch.setattr(main.httpx, "get", recording_get)
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert "http://gateway.test:8000/livez" in calls


def test_not_ready_when_qdrant_down(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main.httpx,
        "get",
        _fake_get(
            {
                "opensearch.test": 200,
                "qdrant.test": httpx.ConnectError("refused"),
                "gateway.test": 200,
            }
        ),
    )
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["checks"]["qdrant"] == "unreachable: ConnectError"
    assert body["checks"]["opensearch"] == "ok"


def test_gateway_down_does_not_block_readiness(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main.httpx,
        "get",
        _fake_get({"opensearch.test": 200, "qdrant.test": 200, "gateway.test": 502}),
    )
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["gateway"] == "http 502"


class FakeIndex:
    def __init__(self, chunks: list) -> None:
        self.chunks = chunks
        self.asked: list[str] = []

    def chunks_for_doc(self, doc_id: str, limit: int = 200) -> list:
        self.asked.append(doc_id)
        return self.chunks


@pytest.fixture
def rag_client() -> Iterator[tuple[TestClient, FakeChat, FakeRetriever]]:
    chat = FakeChat("Statins inhibit HMG-CoA reductase [2].")
    retriever = FakeRetriever([hit(1), hit(2, "Statins inhibit HMG-CoA reductase.")])
    main.app.dependency_overrides[get_pipeline] = lambda: RAGPipeline(retriever, chat)
    main.app.dependency_overrides[get_index] = lambda: FakeIndex([hit(1), hit(2)])
    yield TestClient(main.app), chat, retriever
    main.app.dependency_overrides.clear()


def test_ask_returns_answer_with_citations(rag_client) -> None:
    client, _, retriever = rag_client
    resp = client.post(
        "/ask",
        json={"question": "What do statins inhibit?", "filters": {"year_min": 2015}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False
    assert body["answer"].endswith("[2].")
    assert [c["chunk_id"] for c in body["citations"]] == ["c2"]
    assert body["citations"][0]["url"] == "https://pmc.ncbi.nlm.nih.gov/articles/PMC2/"
    assert body["usage"]["llm_calls"] == 1
    assert retriever.calls[0][2].year_min == 2015


def test_ask_validates_input(rag_client) -> None:
    client, _, _ = rag_client
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_maps_gateway_errors_to_502(rag_client) -> None:
    client, chat, _ = rag_client

    def boom(*_, **__):
        raise GatewayError("gateway returned 500")

    chat.chat = boom
    resp = client.post("/ask", json={"question": "What do statins inhibit?"})
    assert resp.status_code == 502


def test_summarize_normalizes_pmcid(rag_client) -> None:
    client, _, _ = rag_client
    resp = client.post("/summarize", json={"pmcid": "1234"})
    assert resp.status_code == 200
    assert resp.json()["citations"][0]["pmcid"] == "PMC2"
    assert client.post("/summarize", json={"pmcid": "not-an-id"}).status_code == 422


def test_summarize_404_when_not_indexed() -> None:
    main.app.dependency_overrides[get_pipeline] = lambda: RAGPipeline(
        FakeRetriever([]), FakeChat("x")
    )
    main.app.dependency_overrides[get_index] = lambda: FakeIndex([])
    try:
        resp = TestClient(main.app).post("/summarize", json={"pmcid": "PMC9"})
        assert resp.status_code == 404
    finally:
        main.app.dependency_overrides.clear()
