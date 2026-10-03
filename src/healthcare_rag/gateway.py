"""Client for the LLM Gateway (01-llm-gateway), which speaks the OpenAI API."""

import time
from dataclasses import dataclass

import httpx

from healthcare_rag.config import Settings


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float  # from the gateway's x-gateway-cost-usd header
    latency_s: float  # wall clock, measured by this client
    backend: str
    cache: str


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    prompt_tokens: int
    cost_usd: float
    latency_s: float


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        embedding_model: str = "nomic-embed-text",
        timeout: float = 120.0,
        bypass_cache: bool = False,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model
        self.bypass_cache = bypass_cache
        self.http = http or httpx.Client(timeout=timeout)
        if api_key:
            self.http.headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: object) -> "GatewayClient":
        kwargs: dict = {
            "base_url": settings.gateway_base_url,
            "api_key": settings.gateway_api_key,
            "model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "timeout": settings.gateway_timeout_seconds,
            "bypass_cache": settings.gateway_bypass_cache,
        }
        return cls(**{**kwargs, **overrides})

    def _post(self, path: str, payload: dict, bypass_cache: bool) -> tuple[httpx.Response, float]:
        headers = {"Cache-Control": "no-cache"} if bypass_cache else {}
        start = time.perf_counter()
        try:
            resp = self.http.post(f"{self.base_url}{path}", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise GatewayError(f"gateway unreachable: {type(exc).__name__}") from exc
        latency = time.perf_counter() - start
        if not resp.is_success:
            raise GatewayError(f"gateway returned {resp.status_code}: {resp.text[:200]}")
        return resp, latency

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        bypass_cache: bool | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        bypass = self.bypass_cache if bypass_cache is None else bypass_cache
        resp, latency = self._post("/chat/completions", payload, bypass)
        body = resp.json()
        usage = body.get("usage") or {}
        return ChatResult(
            text=body["choices"][0]["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cost_usd=float(resp.headers.get("x-gateway-cost-usd", 0.0)),
            latency_s=latency,
            backend=resp.headers.get("x-gateway-backend", ""),
            cache=resp.headers.get("x-gateway-cache", ""),
        )

    def embed(self, texts: list[str]) -> EmbeddingResult:
        payload = {"model": self.embedding_model, "input": texts}
        resp, latency = self._post("/embeddings", payload, bypass_cache=True)
        body = resp.json()
        data = sorted(body["data"], key=lambda d: d["index"])
        return EmbeddingResult(
            vectors=[d["embedding"] for d in data],
            prompt_tokens=int((body.get("usage") or {}).get("prompt_tokens", 0)),
            cost_usd=float(resp.headers.get("x-gateway-cost-usd", 0.0)),
            latency_s=latency,
        )
