"""Builds the retriever, reranker and RAG pipeline from settings."""

from functools import lru_cache

from healthcare_rag.bm25 import BM25Index, client_from_settings
from healthcare_rag.config import Settings, get_settings
from healthcare_rag.gateway import GatewayClient
from healthcare_rag.rag import RAGConfig, RAGPipeline


def rag_config(settings: Settings) -> RAGConfig:
    return RAGConfig(
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
        abstain_threshold=settings.abstain_threshold,
        require_citations=settings.require_citations,
    )


def build_bm25(settings: Settings) -> BM25Index:
    return BM25Index(client_from_settings(settings), settings.opensearch_index)


def build_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(
        retriever=build_bm25(settings),
        chat=GatewayClient.from_settings(settings),
        config=rag_config(settings),
    )


@lru_cache
def get_pipeline() -> RAGPipeline:
    return build_pipeline(get_settings())


@lru_cache
def get_index() -> BM25Index:
    return build_bm25(get_settings())
