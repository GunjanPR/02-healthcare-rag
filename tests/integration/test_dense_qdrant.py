"""Needs Qdrant on QDRANT_URL (make up). Run with: make test-integration"""

import uuid
from collections.abc import Iterator

import pytest

from healthcare_rag.config import Settings
from healthcare_rag.dense import DenseIndex, client_from_settings
from healthcare_rag.gateway import EmbeddingResult
from healthcare_rag.indexing import index_dense
from healthcare_rag.types import Chunk, Filters

pytestmark = pytest.mark.integration

VECTORS = {"metformin": [1.0, 0.0, 0.0], "asthma": [0.0, 1.0, 0.0], "sepsis": [0.0, 0.0, 1.0]}


class KeywordEmbedder:
    """Deterministic 3-d embeddings, so the test needs Qdrant but not the gateway."""

    def embed(self, texts: list[str]) -> EmbeddingResult:
        vecs = [next(v for k, v in VECTORS.items() if k in t.lower()) for t in texts]
        return EmbeddingResult(vecs, 0, 0.0, 0.0)


@pytest.fixture
def dense() -> Iterator[DenseIndex]:
    idx = DenseIndex(
        client_from_settings(Settings()), f"test-{uuid.uuid4().hex[:8]}", KeywordEmbedder()
    )
    yield idx
    idx.client.delete_collection(idx.collection)


def test_dense_index_is_idempotent_filterable_and_searchable(dense: DenseIndex) -> None:
    chunks = [
        Chunk("PMC1:0", "PMC1", "Metformin", "metformin text", "results", year=2020),
        Chunk("PMC2:0", "PMC2", "Asthma", "asthma text", "abstract", year=2018),
        Chunk("PMC3:0", "PMC3", "Sepsis", "sepsis text", "results", year=2022),
    ]
    assert index_dense(dense, chunks)["embedded"] == 3
    assert index_dense(dense, chunks)["embedded"] == 0
    assert dense.count() == 3

    assert dense.search("asthma control", k=1)[0].id == "PMC2:0"
    filtered = dense.search("asthma control", k=3, filters=Filters(year_min=2019))
    assert {h.id for h in filtered} == {"PMC1:0", "PMC3:0"}
    by_section = dense.search("metformin", k=3, filters=Filters(sections=("abstract",)))
    assert [h.id for h in by_section] == ["PMC2:0"]
