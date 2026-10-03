"""Write chunks to OpenSearch (BM25) and Qdrant (dense). Both writes are idempotent."""

import time
from collections.abc import Sequence

from healthcare_rag.bm25 import BM25Index
from healthcare_rag.bm25 import client_from_settings as opensearch_client
from healthcare_rag.config import Settings
from healthcare_rag.dense import DenseIndex
from healthcare_rag.dense import client_from_settings as qdrant_client
from healthcare_rag.gateway import GatewayClient
from healthcare_rag.types import Chunk


def build_dense(settings: Settings, embedder: GatewayClient | None = None) -> DenseIndex:
    return DenseIndex(
        qdrant_client(settings),
        settings.qdrant_collection,
        embedder or GatewayClient.from_settings(settings),
        query_prefix=settings.embedding_query_prefix,
        doc_prefix=settings.embedding_doc_prefix,
    )


def index_dense(dense: DenseIndex, chunks: Sequence[Chunk], batch_size: int = 64) -> dict:
    """Embed and upsert only the chunks Qdrant doesn't have yet."""
    by_id = {c.chunk_id: c for c in chunks}
    todo: list[str] = []
    for i in range(0, len(chunks), 1000):
        todo += dense.missing([c.chunk_id for c in chunks[i : i + 1000]])
    start = time.perf_counter()
    tokens = 0
    cost = 0.0
    for i in range(0, len(todo), batch_size):
        batch = [by_id[cid] for cid in todo[i : i + batch_size]]
        result = dense.embed_documents(batch)
        if i == 0:
            dense.ensure_collection(dim=len(result.vectors[0]))
        dense.upsert(batch, result.vectors)
        tokens += result.prompt_tokens
        cost += result.cost_usd
        done = min(i + batch_size, len(todo))
        if done % (batch_size * 20) == 0 or done == len(todo):
            rate = done / (time.perf_counter() - start)
            print(f"  embedded {done}/{len(todo)} ({rate:.1f} chunks/s)", flush=True)
    return {
        "embedded": len(todo),
        "skipped_existing": len(chunks) - len(todo),
        "embed_seconds": round(time.perf_counter() - start, 1),
        "embed_tokens": tokens,
        "embed_cost_usd": cost,
    }


def index_all(settings: Settings, chunks: Sequence[Chunk], batch_size: int = 64) -> dict:
    bm25 = BM25Index(opensearch_client(settings), settings.opensearch_index)
    bm25.ensure_index()
    start = time.perf_counter()
    bm25.index_chunks(chunks)
    report = {"bm25_upserted": len(chunks), "bm25_seconds": round(time.perf_counter() - start, 1)}
    dense = build_dense(settings, GatewayClient.from_settings(settings, bypass_cache=True))
    report |= index_dense(dense, chunks, batch_size=batch_size)
    report |= {"opensearch_count": bm25.count(), "qdrant_count": dense.count()}
    return report
