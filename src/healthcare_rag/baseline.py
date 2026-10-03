"""M1 baseline: BM25-only retrieval (OpenSearch) + one gateway chat call. No reranking.

This is the control that later pipelines (hybrid retrieval, reranking, LangGraph) are
measured against.
"""

import time
from dataclasses import dataclass
from typing import Protocol

from healthcare_rag.bm25 import Hit
from healthcare_rag.gateway import ChatResult

SYSTEM_PROMPT = (
    "You are a medical research assistant. Answer the question using only the numbered "
    "context passages. Be brief: one or two sentences. Cite the passages you used as [1], "
    "[2]. If the context does not contain the answer, say you don't know."
)


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> ChatResult: ...


@dataclass(frozen=True)
class BaselineAnswer:
    question: str
    answer: str
    hits: list[Hit]
    retrieval_s: float
    chat: ChatResult


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{i}] {h.title}\n{h.text}" for i, h in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


class BM25Baseline:
    def __init__(self, retriever: Retriever, chat: ChatClient, k: int = 3) -> None:
        self.retriever = retriever
        self.chat_client = chat
        self.k = k

    def answer(self, question: str) -> BaselineAnswer:
        start = time.perf_counter()
        hits = self.retriever.search(question, k=self.k)
        retrieval_s = time.perf_counter() - start
        result = self.chat_client.chat(build_messages(question, hits))
        return BaselineAnswer(question, result.text, hits, retrieval_s, result)
