"""BM25 retrieval over OpenSearch."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from opensearchpy import OpenSearch, helpers

from healthcare_rag.config import Settings
from healthcare_rag.types import Chunk, Filters, Hit

# English analyzer (stemming + stopwords) on the text fields; scoring is OpenSearch's
# default BM25. Metadata fields are keywords/integers so they can be used as filters.
INDEX_BODY: dict[str, Any] = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "english"},
            "section": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "english"},
            "position": {"type": "integer"},
            "year": {"type": "integer"},
            "journal": {"type": "keyword"},
            "topic": {"type": "keyword"},
        }
    },
}


@dataclass(frozen=True)
class Document:
    """Toy-corpus document: one chunk per document."""

    id: str
    title: str
    text: str

    def to_chunk(self) -> Chunk:
        return Chunk(chunk_id=self.id, doc_id=self.id, title=self.title, text=self.text)


def client_from_settings(settings: Settings) -> OpenSearch:
    auth = (
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
    )
    return OpenSearch(hosts=[settings.opensearch_url], http_auth=auth, timeout=60)


def filter_clauses(filters: Filters | None) -> list[dict[str, Any]]:
    if filters is None:
        return []
    clauses: list[dict[str, Any]] = []
    if filters.sections:
        clauses.append({"terms": {"section": list(filters.sections)}})
    if filters.topics:
        clauses.append({"terms": {"topic": list(filters.topics)}})
    if filters.doc_ids:
        clauses.append({"terms": {"doc_id": list(filters.doc_ids)}})
    if filters.year_min is not None or filters.year_max is not None:
        rng = {"gte": filters.year_min, "lte": filters.year_max}
        clauses.append({"range": {"year": {k: v for k, v in rng.items() if v is not None}}})
    return clauses


def _hit(source: dict[str, Any], score: float) -> Hit:
    return Hit(
        id=source["chunk_id"],
        title=source.get("title", ""),
        text=source["text"],
        score=score,
        doc_id=source.get("doc_id", ""),
        section=source.get("section", ""),
        year=source.get("year"),
        journal=source.get("journal", ""),
    )


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
        return self.index_chunks(d.to_chunk() for d in docs)

    def index_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Upsert chunks keyed by chunk_id, so re-running does not create duplicates."""
        actions = (
            {"_op_type": "index", "_index": self.index, "_id": c.chunk_id, "_source": asdict(c)}
            for c in chunks
        )
        indexed, _ = helpers.bulk(self.client, actions, refresh="wait_for", chunk_size=500)
        return indexed

    def count(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    def search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[Hit]:
        body = {
            "size": k,
            "query": {
                "bool": {
                    "must": {"multi_match": {"query": query, "fields": ["title", "text"]}},
                    "filter": filter_clauses(filters),
                }
            },
        }
        resp = self.client.search(index=self.index, body=body)
        return [_hit(h["_source"], float(h["_score"])) for h in resp["hits"]["hits"]]

    def chunks_for_doc(self, doc_id: str, limit: int = 200) -> list[Hit]:
        """All chunks of one article, in reading order."""
        body = {
            "size": limit,
            "query": {"term": {"doc_id": doc_id}},
            "sort": [{"position": "asc"}],
        }
        resp = self.client.search(index=self.index, body=body)
        return [_hit(h["_source"], 0.0) for h in resp["hits"]["hits"]]
