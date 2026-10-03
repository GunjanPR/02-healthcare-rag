"""LangGraph RAG flow: retrieve -> rerank -> (abstain | generate -> verify).

- retrieve: top-N candidates from the retriever (BM25, dense or hybrid).
- rerank:   reorder with a reranker if one is configured, keep the top K.
- gate:     abstain without calling the LLM when evidence is weak (no hits, or the best
            reranker score is below the threshold).
- generate: one chat call through the LLM Gateway, with numbered passages to cite.
- verify:   the answer must cite at least one passage that was actually provided, and must
            not be the model's "insufficient evidence" sentinel; otherwise abstain.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from healthcare_rag.gateway import ChatResult
from healthcare_rag.types import Filters, Hit

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Answer the question using only the numbered "
    "context passages from PubMed Central articles. Rules:\n"
    "1. Every sentence must cite its supporting passage(s) as [1], [2], etc.\n"
    "2. Do not use outside knowledge. Keep the answer to 1-3 sentences.\n"
    f"3. If the passages do not contain the answer, reply with exactly {INSUFFICIENT}."
)

SUMMARY_PROMPT = (
    "You are a biomedical research assistant. Summarize the article from the numbered "
    "passages in 4-6 sentences covering objective, methods, main findings and conclusion. "
    "Cite passages as [1], [2], etc. after each sentence. Use only the passages."
)

ABSTAIN_MESSAGE = "I can't answer this from the indexed literature with enough confidence."

_CITATION = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]")


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[Hit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Return hits sorted best-first, with `score` set to a 0-1 relevance score."""
        ...


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResult: ...


@dataclass(frozen=True)
class RAGConfig:
    retrieval_top_k: int = 50
    rerank_top_k: int = 5
    # Applied to the reranker's 0-1 score; ignored when there is no reranker.
    abstain_threshold: float = 0.35
    require_citations: bool = True
    max_tokens: int = 256


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0

    def add(self, result: ChatResult) -> None:
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.cost_usd += result.cost_usd
        self.llm_calls += 1


@dataclass
class RAGResult:
    question: str
    answer: str
    abstained: bool
    reason: str  # "answered" | "no_hits" | "low_score" | "model_insufficient" | "uncited"
    passages: list[Hit]  # the passages shown to the model, numbered from 1
    cited: list[int]  # 1-based passage numbers cited in the answer
    candidates: list[Hit]  # retriever output before reranking
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    usage: Usage = field(default_factory=Usage)
    raw_answer: str = ""

    @property
    def cited_passages(self) -> list[Hit]:
        return [self.passages[n - 1] for n in self.cited]


class State(TypedDict, total=False):
    question: str
    filters: Filters | None
    candidates: list[Hit]
    passages: list[Hit]
    raw_answer: str
    answer: str
    cited: list[int]
    abstained: bool
    reason: str
    timings: dict[str, float]
    usage: Usage


def number_passages(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = f" ({h.section})" if h.section else ""
        blocks.append(f"[{i}] {h.title}{where}\n{h.text}")
    return "\n\n".join(blocks)


def parse_citations(answer: str, n_passages: int) -> list[int]:
    """Passage numbers cited in the answer that refer to a provided passage, in order."""
    seen: list[int] = []
    for group in _CITATION.findall(answer):
        for part in re.split(r"\s*[,;]\s*", group):
            bounds = [int(x) for x in re.split(r"\s*[–-]\s*", part)]  # "2-4" is a range
            for n in range(bounds[0], bounds[-1] + 1):
                if 1 <= n <= n_passages and n not in seen:
                    seen.append(n)
    return seen


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        chat: ChatClient,
        reranker: Reranker | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.chat_client = chat
        self.reranker = reranker
        self.config = config or RAGConfig()
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------

    def _retrieve(self, state: State) -> State:
        start = time.perf_counter()
        hits = self.retriever.search(
            state["question"], k=self.config.retrieval_top_k, filters=state.get("filters")
        )
        return {"candidates": hits, "timings": {"retrieve": time.perf_counter() - start}}

    def _rerank(self, state: State) -> State:
        start = time.perf_counter()
        hits = state["candidates"]
        if self.reranker is not None and hits:
            hits = self.reranker.rerank(state["question"], hits)
        timings = {**state["timings"], "rerank": time.perf_counter() - start}
        return {"passages": hits[: self.config.rerank_top_k], "timings": timings}

    def _gate(self, state: State) -> str:
        passages = state["passages"]
        if not passages:
            return "no_hits"
        if self.reranker is not None and passages[0].score < self.config.abstain_threshold:
            return "low_score"
        return "generate"

    def _generate(self, state: State) -> State:
        start = time.perf_counter()
        context = number_passages(state["passages"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"},
        ]
        result = self.chat_client.chat(messages, max_tokens=self.config.max_tokens)
        usage = state.get("usage") or Usage()
        usage.add(result)
        timings = {**state["timings"], "generate": time.perf_counter() - start}
        return {"raw_answer": result.text.strip(), "usage": usage, "timings": timings}

    def _verify(self, state: State) -> State:
        raw = state["raw_answer"]
        cited = parse_citations(raw, len(state["passages"]))
        if INSUFFICIENT in raw:
            return {"abstained": True, "reason": "model_insufficient", "cited": []}
        if self.config.require_citations and not cited:
            return {"abstained": True, "reason": "uncited", "cited": []}
        return {"abstained": False, "reason": "answered", "answer": raw, "cited": cited}

    def _abstain(self, reason: str):
        def node(state: State) -> State:
            return {"abstained": True, "reason": reason, "cited": []}

        return node

    def _build(self):
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("rerank", self._rerank)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)
        g.add_node("abstain_no_hits", self._abstain("no_hits"))
        g.add_node("abstain_low_score", self._abstain("low_score"))
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_conditional_edges(
            "rerank",
            self._gate,
            {
                "no_hits": "abstain_no_hits",
                "low_score": "abstain_low_score",
                "generate": "generate",
            },
        )
        g.add_edge("generate", "verify")
        g.add_edge("verify", END)
        g.add_edge("abstain_no_hits", END)
        g.add_edge("abstain_low_score", END)
        return g.compile()

    # ---- public API --------------------------------------------------------------------

    def ask(self, question: str, filters: Filters | None = None) -> RAGResult:
        start = time.perf_counter()
        out = self.graph.invoke({"question": question, "filters": filters, "usage": Usage()})
        timings = dict(out.get("timings", {}))
        timings["total"] = time.perf_counter() - start
        abstained = out.get("abstained", True)
        return RAGResult(
            question=question,
            answer=ABSTAIN_MESSAGE if abstained else out["answer"],
            abstained=abstained,
            reason=out.get("reason", "no_hits"),
            passages=out.get("passages", []),
            cited=out.get("cited", []),
            candidates=out.get("candidates", []),
            timings=timings,
            usage=out.get("usage") or Usage(),
            raw_answer=out.get("raw_answer", ""),
        )

    def summarize(self, title: str, chunks: list[Hit], max_passages: int = 8) -> RAGResult:
        """Summarize one article from its first `max_passages` chunks (reading order)."""
        start = time.perf_counter()
        passages = chunks[:max_passages]
        usage = Usage()
        if not passages:
            return RAGResult(title, ABSTAIN_MESSAGE, True, "no_hits", [], [], [])
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Article: {title}\n\n{number_passages(passages)}"},
        ]
        result = self.chat_client.chat(messages, max_tokens=400)
        usage.add(result)
        text = result.text.strip()
        return RAGResult(
            question=title,
            answer=text,
            abstained=False,
            reason="answered",
            passages=passages,
            cited=parse_citations(text, len(passages)),
            candidates=passages,
            timings={"generate": result.latency_s, "total": time.perf_counter() - start},
            usage=usage,
            raw_answer=text,
        )
