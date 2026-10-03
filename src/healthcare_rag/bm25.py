"""BM25 retrieval over OpenSearch."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from opensearchpy import OpenSearch, helpers

from healthcare_rag.config import Settings

# English analyzer (stemming + stopwords) on both fields; scoring is OpenSearch's default BM25.
INDEX_BODY: dict[str, Any] = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "english"},
            "text": {"type": "text", "analyzer": "english"},
        }
    },
}


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    id: str
    title: str
    text: str
    score: float


def client_from_settings(settings: Settings) -> OpenSearch:
    auth = (
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
    )
    return OpenSearch(hosts=[settings.opensearch_url], http_auth=auth, timeout=30)


class BM25Index:
    def __init__(self, client: OpenSearch, index: str) -> None:
        self.client = client
        self.index = index

    def ensure_index(self, recreate: bool = False) -> None:
        exists = self.client.indices.exists(index=self.index)
        if exists and recreate:
            self.client.indices.delete(index=self.index)
            exists = False
        if not exists:
            self.client.indices.create(index=self.index, body=INDEX_BODY)

    def index_documents(self, docs: Iterable[Document]) -> int:
        """Upsert documents keyed by their id, so re-running does not create duplicates."""
        actions = (
            {
                "_op_type": "index",
                "_index": self.index,
                "_id": doc.id,
                "_source": {"doc_id": doc.id, "title": doc.title, "text": doc.text},
            }
            for doc in docs
        )
        indexed, _ = helpers.bulk(self.client, actions, refresh="wait_for")
        return indexed

    def count(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        body = {
            "size": k,
            "query": {"multi_match": {"query": query, "fields": ["title", "text"]}},
        }
        resp = self.client.search(index=self.index, body=body)
        return [
            Hit(
                id=h["_source"]["doc_id"],
                title=h["_source"]["title"],
                text=h["_source"]["text"],
                score=float(h["_score"]),
            )
            for h in resp["hits"]["hits"]
        ]
