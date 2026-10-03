"""Client for the LLM Gateway (01-llm-gateway), which speaks the OpenAI chat API."""

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


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = http or httpx.Client(timeout=timeout)
        self.http.headers.update(headers)

    @classmethod
    def from_settings(cls, settings: Settings) -> "GatewayClient":
        return cls(settings.gateway_base_url, settings.gateway_api_key, settings.chat_model)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        bypass_cache: bool = True,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Cache-Control": "no-cache"} if bypass_cache else {}
        start = time.perf_counter()
        resp = self.http.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        latency = time.perf_counter() - start
        if not resp.is_success:
            raise GatewayError(f"gateway returned {resp.status_code}: {resp.text[:200]}")
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
