"""Dense retrieval over Qdrant. Embeddings come from the LLM Gateway (nomic-embed-text)."""

import uuid
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from healthcare_rag.config import Settings
from healthcare_rag.gateway import EmbeddingResult
from healthcare_rag.types import Chunk, Filters, Hit

_NAMESPACE = uuid.UUID("6f1c3d1e-6c2b-4a53-9a39-6b8f1f0e2d11")
PAYLOAD_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "topic": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
}


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingResult: ...


def point_id(chunk_id: str) -> str:
    """Deterministic UUID per chunk, so re-ingesting overwrites instead of duplicating."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def client_from_settings(settings: Settings) -> QdrantClient:
    api_key = settings.qdrant_api_key or None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=60)


def qdrant_filter(filters: Filters | None) -> models.Filter | None:
    if filters is None:
        return None
    must: list[Any] = []
    for key, values in (
        ("section", filters.sections),
        ("topic", filters.topics),
        ("doc_id", filters.doc_ids),
    ):
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    if filters.year_min is not None or filters.year_max is not None:
        rng = models.Range(gte=filters.year_min, lte=filters.year_max)
        must.append(models.FieldCondition(key="year", range=rng))
    return models.Filter(must=must) if must else None


class DenseIndex:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: Embedder,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            for field, schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(self.collection, field, field_schema=schema)

    def missing(self, chunk_ids: Sequence[str]) -> list[str]:
        """Chunk ids that are not in the collection yet."""
        if not self.client.collection_exists(self.collection):
            return list(chunk_ids)
        found = self.client.retrieve(
            self.collection, ids=[point_id(c) for c in chunk_ids], with_payload=False
        )
        have = {str(p.id) for p in found}
        return [c for c in chunk_ids if point_id(c) not in have]

    def embed_documents(self, chunks: Sequence[Chunk]) -> EmbeddingResult:
        return self.embedder.embed([f"{self.doc_prefix}{c.title}\n{c.text}" for c in chunks])

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        points = [
            models.PointStruct(id=point_id(c.chunk_id), vector=v, payload=asdict(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[Hit]:
        vector = self.embedder.embed([f"{self.query_prefix}{query}"]).vectors[0]
        resp = self.client.query_points(
            self.collection,
            query=vector,
            limit=k,
            query_filter=qdrant_filter(filters),
            with_payload=True,
        )
        return [_hit(p.payload or {}, float(p.score)) for p in resp.points]


def _hit(payload: dict[str, Any], score: float) -> Hit:
    return Hit(
        id=payload["chunk_id"],
        title=payload.get("title", ""),
        text=payload.get("text", ""),
        score=score,
        doc_id=payload.get("doc_id", ""),
        section=payload.get("section", ""),
        year=payload.get("year"),
        journal=payload.get("journal", ""),
    )
