"""FastAPI app. M1: liveness and readiness endpoints only."""

from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Response, status

from healthcare_rag import __version__
from healthcare_rag.config import Settings, get_settings

app = FastAPI(title="Healthcare RAG", version=__version__)

SettingsDep = Annotated[Settings, Depends(get_settings)]


def _probe(url: str, timeout: float, auth: tuple[str, str] | None = None) -> str:
    """Return "ok" if the URL answers with a 2xx status, otherwise a short error string."""
    try:
        resp = httpx.get(url, timeout=timeout, auth=auth)
    except httpx.HTTPError as exc:
        return f"unreachable: {type(exc).__name__}"
    return "ok" if resp.is_success else f"http {resp.status_code}"


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Does not touch backing services."""
    return {"status": "ok", "service": "healthcare-rag", "version": __version__}


@app.get("/ready")
def ready(settings: SettingsDep, response: Response) -> dict[str, object]:
    """Readiness: OpenSearch and Qdrant are reachable. The gateway is reported, not required."""
    timeout = settings.probe_timeout_seconds
    os_auth = (
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
    )
    checks = {
        "opensearch": _probe(
            f"{settings.opensearch_url.rstrip('/')}/_cluster/health", timeout, os_auth
        ),
        "qdrant": _probe(f"{settings.qdrant_url.rstrip('/')}/readyz", timeout),
    }
    gateway = _probe(f"{settings.gateway_base_url.rstrip('/').removesuffix('/v1')}/livez", timeout)
    is_ready = all(v == "ok" for v in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if is_ready else "not ready", "checks": checks, "gateway": gateway}
