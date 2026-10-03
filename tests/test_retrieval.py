import pytest
from qdrant_client import models

from fakes import FakeRetriever, hit
from healthcare_rag.bm25 import filter_clauses
from healthcare_rag.dense import DenseIndex, point_id, qdrant_filter
from healthcare_rag.gateway import EmbeddingResult
from healthcare_rag.hybrid import HybridRetriever, rrf
from healthcare_rag.indexing import index_dense
from healthcare_rag.types import Chunk, Filters


def test_rrf_rewards_agreement() -> None:
    a = [hit(1), hit(2), hit(3)]
    b = [hit(3), hit(4), hit(1)]
    fused = rrf([a, b], k=4, rrf_k=60)
    # c1: 1/61 + 1/63, c3: 1/63 + 1/61 (tie, c1 seen first), then c2 and c4 (1/62 each)
    assert [h.id for h in fused] == ["c1", "c3", "c2", "c4"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_hybrid_queries_both_and_truncates() -> None:
    sparse = FakeRetriever([hit(1), hit(2)])
    dense = FakeRetriever([hit(2), hit(3)])
    out = HybridRetriever(sparse, dense).search("q", k=2, filters=Filters(topics=("asthma",)))
    assert [h.id for h in out] == ["c2", "c1"]
    assert sparse.calls[0][2] == dense.calls[0][2] == Filters(topics=("asthma",))


def test_opensearch_filter_clauses() -> None:
    assert filter_clauses(None) == []
    clauses = filter_clauses(Filters(sections=("results",), year_min=2020))
    assert {"terms": {"section": ["results"]}} in clauses
    assert {"range": {"year": {"gte": 2020}}} in clauses


def test_qdrant_filter() -> None:
    assert qdrant_filter(None) is None
    assert qdrant_filter(Filters()) is None
    f = qdrant_filter(Filters(doc_ids=("PMC1",), year_min=2019, year_max=2021))
    assert f is not None and len(f.must) == 2
    year = next(c for c in f.must if c.key == "year")
    assert year.range == models.Range(gte=2019, lte=2021)


def test_point_id_is_deterministic() -> None:
    assert point_id("PMC1:0") == point_id("PMC1:0") != point_id("PMC1:1")


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self.batches.append(texts)
        return EmbeddingResult([[1.0, 0.0]] * len(texts), 5 * len(texts), 0.0, 0.01)


class FakeQdrantDense(DenseIndex):
    """DenseIndex with an in-memory store instead of a Qdrant server."""

    def __init__(self, embedder: FakeEmbedder) -> None:
        super().__init__(None, "test", embedder, doc_prefix="search_document: ")
        self.points: dict[str, list[float]] = {}
        self.created_dim: int | None = None

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        self.created_dim = dim

    def missing(self, chunk_ids):
        return [c for c in chunk_ids if c not in self.points]

    def upsert(self, chunks, vectors) -> None:
        for c, v in zip(chunks, vectors, strict=True):
            self.points[c.chunk_id] = v

    def count(self) -> int:
        return len(self.points)


def test_index_dense_is_idempotent() -> None:
    chunks = [Chunk(f"PMC1:{i}", "PMC1", "T", f"text {i}") for i in range(5)]
    embedder = FakeEmbedder()
    dense = FakeQdrantDense(embedder)

    first = index_dense(dense, chunks, batch_size=2)
    assert first["embedded"] == 5 and first["skipped_existing"] == 0
    assert dense.created_dim == 2 and dense.count() == 5
    assert embedder.batches[0][0] == "search_document: T\ntext 0"

    second = index_dense(dense, chunks + [Chunk("PMC2:0", "PMC2", "U", "new")], batch_size=2)
    assert second["embedded"] == 1 and second["skipped_existing"] == 5
    assert dense.count() == 6
