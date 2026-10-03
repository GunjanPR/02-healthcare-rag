"""Test doubles for the retriever, reranker and gateway."""

from healthcare_rag.gateway import ChatResult
from healthcare_rag.types import Filters, Hit


def hit(i: int, text: str = "", score: float = 1.0, doc: str = "") -> Hit:
    return Hit(
        id=f"c{i}",
        title=f"Title {i}",
        text=text or f"passage {i}",
        score=score,
        doc_id=doc or f"PMC{i}",
        section="results",
        year=2020,
    )


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, Filters | None]] = []

    def search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[Hit]:
        self.calls.append((query, k, filters))
        return self.hits[:k]


class FakeReranker:
    """Reverses the order and assigns fixed scores."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        out = list(reversed(hits))
        return [
            Hit(h.id, h.title, h.text, s, h.doc_id, h.section, h.year, h.journal)
            for h, s in zip(out, self.scores, strict=False)
        ]


class FakeChat:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **_: object) -> ChatResult:
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return ChatResult(reply, 100, 10, 0.001, 0.2, "ollama", "BYPASS")
