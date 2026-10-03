"""Builds the retriever, reranker and RAG pipeline from settings."""

from functools import lru_cache

from healthcare_rag.bm25 import BM25Index, client_from_settings
from healthcare_rag.config import Settings, get_settings
from healthcare_rag.gateway import GatewayClient
from healthcare_rag.hybrid import HybridRetriever
from healthcare_rag.indexing import build_dense
from healthcare_rag.rag import RAGConfig, RAGPipeline, Reranker, Retriever

RETRIEVAL_MODES = ("bm25", "dense", "hybrid")


def rag_config(settings: Settings) -> RAGConfig:
    return RAGConfig(
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
        abstain_threshold=settings.abstain_threshold,
        require_citations=settings.require_citations,
        citation_repairs=settings.citation_repairs,
        prompt=settings.prompt_version,
        citation_repair=settings.citation_repair,
        citation_check=settings.citation_check,
        support_threshold=settings.support_threshold,
        abstain_mode=settings.abstain_mode,
    )


def build_bm25(settings: Settings) -> BM25Index:
    return BM25Index(client_from_settings(settings), settings.opensearch_index)


def build_retriever(settings: Settings, mode: str | None = None) -> Retriever:
    mode = mode or settings.retrieval_mode
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
    if mode == "bm25":
        return build_bm25(settings)
    dense = build_dense(settings)
    if mode == "dense":
        return dense
    return HybridRetriever(build_bm25(settings), dense, rrf_k=settings.rrf_k)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_model:
        return None
    from healthcare_rag.rerank import CrossEncoderReranker, EnsembleReranker  # [rerank] extra

    primary = CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)
    if not settings.reranker_model_2 and not settings.reranker_retrieval_weight:
        return primary
    members = [primary]
    if settings.reranker_model_2:
        members.append(
            CrossEncoderReranker(settings.reranker_model_2, device=settings.reranker_device)
        )
    return EnsembleReranker(members, retrieval_weight=settings.reranker_retrieval_weight)


def build_scorer(settings: Settings):
    from healthcare_rag.citations import LexicalScorer

    if not settings.nli_model:
        return LexicalScorer()
    from healthcare_rag.nli import NLIScorer  # needs the [rerank] extra

    return NLIScorer(settings.nli_model, device=settings.reranker_device)


def build_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(
        retriever=build_retriever(settings),
        chat=GatewayClient.from_settings(settings),
        reranker=build_reranker(settings),
        config=rag_config(settings),
        scorer=build_scorer(settings),
    )


@lru_cache
def get_pipeline() -> RAGPipeline:
    return build_pipeline(get_settings())


@lru_cache
def get_index() -> BM25Index:
    return build_bm25(get_settings())
