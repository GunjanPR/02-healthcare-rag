"""LangGraph RAG flow: retrieve -> rerank -> (abstain | generate -> verify).

- retrieve: top-N candidates from the retriever (BM25, dense or hybrid).
- rerank:   reorder with a reranker if one is configured, keep the top K.
- gate:     abstain without calling the LLM when evidence is weak (no hits, or the best
            reranker score is below the threshold).
- generate: one chat call through the LLM Gateway, with numbered passages to cite.
- verify:   the answer must cite at least one passage that was actually provided, and must
            not be the model's "insufficient evidence" sentinel; otherwise abstain.
- repair:   an answer with no citations gets citations attached, either by matching each
            sentence to the passage that supports it (no LLM call, see citations.py) or by
            one follow-up turn asking the model for them (on the dev split, every uncited
            answer dropped by verify was otherwise correct).
- check:    optionally, every sentence is checked against the passage it cites; a wrong
            citation is pointed at the supporting passage ("relabel") and, in "drop" mode,
            sentences no passage supports are removed (abstain if none is left).
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace

from healthcare_rag import citations
from healthcare_rag.gateway import ChatResult
from healthcare_rag.observability import get_tracer
from healthcare_rag.types import Filters, Hit

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Answer the question using only the numbered "
    "context passages from PubMed Central articles. Rules:\n"
    "1. Every sentence must cite its supporting passage(s) as [1], [2], etc.\n"
    "2. Do not use outside knowledge. Keep the answer to 1-3 sentences.\n"
    f"3. If the passages do not contain the answer, reply with exactly {INSUFFICIENT}."
)

# v2: written after reading the v1 failures on dev. It asks for the direct answer first,
# for numbers and names copied exactly (keyword grading and clinicians both need them),
# and shows the citation format, since most uncited answers simply forgot the brackets.
SYSTEM_PROMPT_V2 = (
    "You are a biomedical research assistant. A clinician asks a question; answer it using "
    "only the numbered context passages from PubMed Central articles. Rules:\n"
    "1. Start with the direct answer to the question in one sentence, then add at most two "
    "sentences of supporting detail.\n"
    "2. Copy numbers, units, percentages, drug names, tests and outcomes exactly as the "
    "passages write them. Do not use outside knowledge.\n"
    "3. End every sentence with the passage number(s) that support it, for example: "
    "'Metformin lowered HbA1c by 1.1% [2].'\n"
    f"4. If the passages do not contain the answer, reply with exactly {INSUFFICIENT}."
)

PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2}
ABSTAIN_MODES = ("both", "threshold", "model")


def system_prompt(version: str, abstain_mode: str = "both") -> str:
    """The prompt's last rule lets the model decline; "threshold" mode drops it."""
    prompt = PROMPTS[version]
    return prompt.rsplit("\n", 1)[0] if abstain_mode == "threshold" else prompt


SUMMARY_PROMPT = (
    "You are a biomedical research assistant. Summarize the article from the numbered "
    "passages in 4-6 sentences covering objective, methods, main findings and conclusion. "
    "Cite passages as [1], [2], etc. after each sentence. Use only the passages."
)

REPAIR_PROMPT = (
    "Your answer did not cite the passages. Rewrite the same answer and put the number of "
    "the supporting passage in square brackets, e.g. [1], after each sentence. If the "
    f"passages do not support the answer, reply with exactly {INSUFFICIENT}."
)

ABSTAIN_MESSAGE = "I can't answer this from the indexed literature with enough confidence."

_CITATION = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]")


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[Hit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Return hits sorted best-first, with `score` set to the reranker's relevance score."""
        ...


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResult: ...


@dataclass(frozen=True)
class RAGConfig:
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    # Compared with the best reranker score (MedCPT: logit units); ignored without a reranker.
    abstain_threshold: float = 0.35
    require_citations: bool = True
    # Follow-up turns that ask the model to add citations to an uncited answer
    citation_repairs: int = 1
    max_tokens: int = 256
    prompt: str = "v1"  # key of PROMPTS
    # How an uncited answer gets citations: "llm" (follow-up turn), "attribute" (match
    # sentences to passages, no LLM call) or "attribute_then_llm"
    citation_repair: str = "llm"
    # Sentence-level citation check: "off", "relabel" (fix wrong citations) or "drop"
    # (also remove sentences no passage supports)
    citation_check: str = "off"
    support_threshold: float = 0.6
    # Who may decline: "both" (reranker threshold, then the model's INSUFFICIENT_EVIDENCE),
    # "threshold" (the prompt has no decline rule) or "model" (no reranker threshold)
    abstain_mode: str = "both"

    def __post_init__(self) -> None:
        if self.prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {sorted(PROMPTS)}, got {self.prompt!r}")
        if self.citation_repair not in ("llm", "attribute", "attribute_then_llm"):
            raise ValueError(f"unknown citation_repair {self.citation_repair!r}")
        if self.citation_check not in ("off", "relabel", "drop"):
            raise ValueError(f"unknown citation_check {self.citation_check!r}")
        if self.abstain_mode not in ABSTAIN_MODES:
            raise ValueError(f"abstain_mode must be one of {ABSTAIN_MODES}")


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
    # "answered" | "no_hits" | "low_score" | "model_insufficient" | "uncited" | "unsupported"
    reason: str
    passages: list[Hit]  # the passages shown to the model, numbered from 1
    cited: list[int]  # 1-based passage numbers cited in the answer
    candidates: list[Hit]  # retriever output before reranking
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    usage: Usage = field(default_factory=Usage)
    raw_answer: str = ""
    top_score: float | None = None  # best reranker score, when a reranker ran
    attributed: bool = False  # citations were attached without an LLM call
    relabeled: int = 0  # sentences whose citation was pointed at a better passage
    support_rate: float | None = None  # share of sentences their citations support

    @property
    def cited_passages(self) -> list[Hit]:
        return [self.passages[n - 1] for n in self.cited]


class State(TypedDict, total=False):
    question: str
    instruction: str  # appended to the question in the generation prompt only
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
    messages: list[dict[str, str]]
    repairs: int
    citation: citations.CitationReport
    trace_ctx: Any  # OpenTelemetry context of the request's "rag.ask" span


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
        scorer: citations.SupportScorer | None = None,
    ) -> None:
        self.retriever = retriever
        self.chat_client = chat
        self.reranker = reranker
        self.config = config or RAGConfig()
        self.scorer = scorer or citations.LexicalScorer()
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------

    def _span(self, state: State, name: str):
        return get_tracer().start_as_current_span(f"rag.{name}", context=state.get("trace_ctx"))

    def _retrieve(self, state: State) -> State:
        start = time.perf_counter()
        with self._span(state, "retrieve") as span:
            hits = self.retriever.search(
                state["question"], k=self.config.retrieval_top_k, filters=state.get("filters")
            )
            span.set_attribute("rag.retriever", type(self.retriever).__name__)
            span.set_attribute("rag.candidates", len(hits))
        return {"candidates": hits, "timings": {"retrieve": time.perf_counter() - start}}

    def _rerank(self, state: State) -> State:
        start = time.perf_counter()
        hits = state["candidates"]
        with self._span(state, "rerank") as span:
            if self.reranker is not None and hits:
                hits = self.reranker.rerank(state["question"], hits)
                span.set_attribute("rag.rerank.top_score", hits[0].score)
            span.set_attribute("rag.passages", min(len(hits), self.config.rerank_top_k))
        timings = {**state["timings"], "rerank": time.perf_counter() - start}
        return {"passages": hits[: self.config.rerank_top_k], "timings": timings}

    def _gate(self, state: State) -> str:
        passages = state["passages"]
        if not passages:
            return "no_hits"
        cfg = self.config
        gated = self.reranker is not None and cfg.abstain_mode != "model"
        if gated and passages[0].score < cfg.abstain_threshold:
            return "low_score"
        return "generate"

    def _generate(self, state: State) -> State:
        start = time.perf_counter()
        context = number_passages(state["passages"])
        asked = f"{state['question']} {state.get('instruction', '')}".strip()
        messages = [
            {
                "role": "system",
                "content": system_prompt(self.config.prompt, self.config.abstain_mode),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {asked}"},
        ]
        with self._span(state, "generate") as span:
            result = self.chat_client.chat(messages, max_tokens=self.config.max_tokens)
            span.set_attribute("gen_ai.usage.input_tokens", result.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", result.completion_tokens)
            span.set_attribute("rag.cost_usd", result.cost_usd)
        usage = state.get("usage") or Usage()
        usage.add(result)
        timings = {**state["timings"], "generate": time.perf_counter() - start}
        messages = [*messages, {"role": "assistant", "content": result.text}]
        return {
            "raw_answer": result.text.strip(),
            "usage": usage,
            "timings": timings,
            "messages": messages,
            "repairs": 0,
        }

    def _repair(self, state: State) -> State:
        """One more turn asking for citations; the answer then goes through verify again."""
        start = time.perf_counter()
        messages = [*state["messages"], {"role": "user", "content": REPAIR_PROMPT}]
        with self._span(state, "repair") as span:
            result = self.chat_client.chat(messages, max_tokens=self.config.max_tokens)
            span.set_attribute("gen_ai.usage.input_tokens", result.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", result.completion_tokens)
        usage = state["usage"]
        usage.add(result)
        timings = {**state["timings"], "repair": time.perf_counter() - start}
        return {
            "raw_answer": result.text.strip(),
            "usage": usage,
            "timings": timings,
            "messages": [*messages, {"role": "assistant", "content": result.text}],
            "repairs": state["repairs"] + 1,
        }

    def _after_verify(self, state: State) -> str:
        llm_repair = self.config.citation_repair in ("llm", "attribute_then_llm")
        if (
            llm_repair
            and state["reason"] == "uncited"
            and state["repairs"] < self.config.citation_repairs
        ):
            return "repair"
        return "done"

    def _verify(self, state: State) -> State:
        start = time.perf_counter()
        cfg = self.config
        raw = state["raw_answer"]
        texts = [f"{h.title}. {h.text}" for h in state["passages"]]
        cited = parse_citations(raw, len(texts))
        timings = state["timings"]
        if INSUFFICIENT in raw:
            return {"abstained": True, "reason": "model_insufficient", "cited": []}
        attributed = False
        if not cited and cfg.require_citations and cfg.citation_repair != "llm":
            attempt = citations.attribute(raw, texts, self.scorer, cfg.support_threshold)
            if attempt.attributed:
                raw, attributed = attempt.answer, True
                cited = parse_citations(raw, len(texts))
        if cfg.require_citations and not cited:
            return {"abstained": True, "reason": "uncited", "cited": []}
        report = citations.check(raw, texts, self.scorer, cfg.support_threshold)
        report.attributed = attributed
        if cfg.citation_check in ("relabel", "drop"):
            report = citations.relabel(report, cfg.support_threshold)
        if cfg.citation_check == "drop":
            report = citations.drop_unsupported(report)
        answer = report.answer if cfg.citation_check != "off" else raw
        timings = {**timings, "verify": timings.get("verify", 0.0) + time.perf_counter() - start}
        if not answer:
            return {"abstained": True, "reason": "unsupported", "cited": [], "timings": timings}
        return {
            "abstained": False,
            "reason": "answered",
            "answer": answer,
            "cited": parse_citations(answer, len(texts)),
            "citation": report,
            "timings": timings,
        }

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
        g.add_node("repair", self._repair)
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", self._after_verify, {"repair": "repair", "done": END})
        g.add_edge("repair", "verify")
        g.add_edge("abstain_no_hits", END)
        g.add_edge("abstain_low_score", END)
        return g.compile()

    # ---- public API --------------------------------------------------------------------

    def ask(
        self, question: str, filters: Filters | None = None, instruction: str = ""
    ) -> RAGResult:
        """`instruction` (e.g. an answer format) goes to the LLM only; retrieval and
        reranking see the plain question, so the abstention scores are unaffected."""
        start = time.perf_counter()
        with get_tracer().start_as_current_span("rag.ask") as span:
            ctx = trace.set_span_in_context(span)
            state = {
                "question": question,
                "instruction": instruction,
                "filters": filters,
                "usage": Usage(),
                "trace_ctx": ctx,
            }
            out = self.graph.invoke(state)
            abstained = out.get("abstained", True)
            span.set_attribute("rag.abstained", abstained)
            span.set_attribute("rag.reason", out.get("reason", "no_hits"))
            span.set_attribute("rag.cited", len(out.get("cited", [])))
        timings = dict(out.get("timings", {}))
        timings["total"] = time.perf_counter() - start
        passages = out.get("passages", [])
        report = out.get("citation")
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
            top_score=passages[0].score if self.reranker and passages else None,
            attributed=bool(report and report.attributed),
            relabeled=report.relabeled if report else 0,
            support_rate=report.supported_rate if report and not abstained else None,
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
