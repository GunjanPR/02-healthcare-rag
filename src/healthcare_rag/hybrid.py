"""Hybrid retrieval: reciprocal rank fusion (RRF) of BM25 and dense results."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Protocol

from healthcare_rag.types import Filters, Hit


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[Hit]: ...


def rrf(rankings: list[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (rrf_k + rank). Best first."""
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(h.id, h)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered]


class HybridRetriever:
    def __init__(self, sparse: Retriever, dense: Retriever, rrf_k: int = 60) -> None:
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k
        self._pool = ThreadPoolExecutor(max_workers=2)

    def search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[Hit]:
        retrievers = (self.sparse, self.dense)
        futures = [self._pool.submit(r.search, query, k, filters) for r in retrievers]
        return rrf([f.result() for f in futures], k=k, rrf_k=self.rrf_k)
