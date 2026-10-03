from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from healthcare_rag import main
from healthcare_rag.config import Settings, get_settings


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
