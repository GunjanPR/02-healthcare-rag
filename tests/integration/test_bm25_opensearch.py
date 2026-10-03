"""Needs OpenSearch on OPENSEARCH_URL (make up). Run with: make test-integration"""

import uuid
from collections.abc import Iterator

import pytest

from healthcare_rag.bm25 import BM25Index, Document, client_from_settings
from healthcare_rag.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def index() -> Iterator[BM25Index]:
    settings = Settings()
    idx = BM25Index(client_from_settings(settings), f"test-{uuid.uuid4().hex[:8]}")
    idx.ensure_index()
    yield idx
    idx.client.indices.delete(index=idx.index)


def test_index_is_idempotent_and_searchable(index: BM25Index) -> None:
    docs = [
        Document("d1", "Metformin", "Metformin is first-line therapy for type 2 diabetes."),
        Document("d2", "Asthma", "Inhaled corticosteroids are the mainstay of asthma control."),
    ]
    index.index_documents(docs)
    index.index_documents(docs)
    assert index.count() == 2
    hits = index.search("which drug treats diabetes", k=2)
    assert hits[0].id == "d1"
