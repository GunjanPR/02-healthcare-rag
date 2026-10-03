"""FastAPI app: /ask, /summarize, /health and /ready."""

from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response, status
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from pydantic import BaseModel, Field

from healthcare_rag import __version__
from healthcare_rag.bm25 import BM25Index
from healthcare_rag.config import Settings, get_settings
from healthcare_rag.factory import get_index, get_pipeline
from healthcare_rag.gateway import GatewayError
from healthcare_rag.rag import RAGPipeline, RAGResult
from healthcare_rag.types import Filters

app = FastAPI(title="Healthcare RAG", version=__version__)

SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]
IndexDep = Annotated[BM25Index, Depends(get_index)]


# ---- schemas -----------------------------------------------------------------------------


class FilterIn(BaseModel):
    sections: list[str] = Field(default_factory=list, examples=[["abstract", "results"]])
    topics: list[str] = Field(default_factory=list)
    pmcids: list[str] = Field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None

    def to_filters(self) -> Filters:
        return Filters(
            sections=tuple(self.sections),
            topics=tuple(self.topics),
            doc_ids=tuple(self.pmcids),
            year_min=self.year_min,
            year_max=self.year_max,
        )


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: FilterIn | None = None


class SummarizeIn(BaseModel):
    pmcid: str = Field(pattern=r"^(?i:pmc)?\d+$", examples=["PMC1234567"])


class Citation(BaseModel):
    n: int
    chunk_id: str
    pmcid: str
    title: str
    section: str
    year: int | None
    url: str
    snippet: str


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    llm_calls: int


class AnswerOut(BaseModel):
    answer: str
    abstained: bool
    reason: str
    citations: list[Citation]
    timings_ms: dict[str, float]
    usage: UsageOut


def _url(pmcid: str) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid.startswith("PMC") else ""


def _answer_out(result: RAGResult) -> AnswerOut:
    citations = [
        Citation(
            n=n,
            chunk_id=h.id,
            pmcid=h.doc_id,
            title=h.title,
            section=h.section,
            year=h.year,
            url=_url(h.doc_id),
            snippet=h.text[:300],
        )
        for n, h in zip(result.cited, result.cited_passages, strict=True)
    ]
    u = result.usage
    return AnswerOut(
        answer=result.answer,
        abstained=result.abstained,
        reason=result.reason,
        citations=citations,
        timings_ms={k: round(v * 1000, 1) for k, v in result.timings.items()},
        usage=UsageOut(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            cost_usd=u.cost_usd,
            llm_calls=u.llm_calls,
        ),
    )


def _backend_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, GatewayError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM gateway error: {exc}")
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "search backend unavailable")


# ---- endpoints ---------------------------------------------------------------------------


@app.post("/ask", response_model=AnswerOut)
def ask(body: AskIn, pipeline: PipelineDep) -> AnswerOut:
    """Answer a question from the indexed literature, citing passages, or abstain."""
    filters = body.filters.to_filters() if body.filters else None
    try:
        result = pipeline.ask(body.question, filters=filters)
    except (GatewayError, OpenSearchConnectionError) as exc:
        raise _backend_errors(exc) from exc
    return _answer_out(result)


@app.post("/summarize", response_model=AnswerOut)
def summarize(body: SummarizeIn, pipeline: PipelineDep, index: IndexDep) -> AnswerOut:
    """Summarize one indexed article, with citations to its passages."""
    pmcid = body.pmcid.upper() if body.pmcid.upper().startswith("PMC") else f"PMC{body.pmcid}"
    try:
        chunks = index.chunks_for_doc(pmcid)
        if not chunks:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"{pmcid} is not indexed")
        result = pipeline.summarize(chunks[0].title, chunks)
    except (GatewayError, OpenSearchConnectionError) as exc:
        raise _backend_errors(exc) from exc
    return _answer_out(result)


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
