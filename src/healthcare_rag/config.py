"""Service settings, loaded from environment variables and `.env`."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8010
    log_level: str = "info"

    # LLM Gateway (01-llm-gateway): all chat and embedding calls go through it
    gateway_base_url: str = "http://127.0.0.1:8000/v1"
    gateway_api_key: str = ""
    chat_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    gateway_timeout_seconds: float = 120.0
    # True sends Cache-Control: no-cache (used for evaluation so every answer is generated)
    gateway_bypass_cache: bool = False

    # OpenSearch (BM25)
    opensearch_url: str = "http://127.0.0.1:9200"
    opensearch_index: str = "pmc_chunks"
    opensearch_user: str = ""
    opensearch_password: str = ""

    # Qdrant (dense vectors)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "pmc_chunks"
    qdrant_api_key: str = ""

    # nomic-embed-text expects task prefixes; set both blank for other embedding models
    embedding_query_prefix: str = "search_query: "
    embedding_doc_prefix: str = "search_document: "

    # Retrieval / RAG flow
    # bm25 | dense | hybrid
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    # Hugging Face cross-encoder id; blank = no reranking
    reranker_model: str = ""
    reranker_device: str = "cpu"
    # Optional second cross-encoder fused with the first by reciprocal rank fusion
    # (e.g. BAAI/bge-reranker-v2-m3); the first model's score still drives abstention
    reranker_model_2: str = ""
    # Weight of the retriever's own order in that fusion (0 = ignore it)
    reranker_retrieval_weight: float = 0.0
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    abstain_threshold: float = 6.43  # MedCPT logit; tuned on the dev split (make eval)
    require_citations: bool = True
    citation_repairs: int = 1
    # System prompt version (rag.PROMPTS)
    prompt_version: str = "v1"
    # llm | attribute | attribute_then_llm (see rag.RAGConfig)
    citation_repair: str = "llm"
    # off | relabel | drop
    citation_check: str = "off"
    support_threshold: float = 0.6
    # Blank = lexical support scoring; an NLI cross-encoder id switches to model scoring
    nli_model: str = ""

    # Observability: blank endpoint = tracing off. 01's Jaeger listens on :4318.
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "healthcare-rag"

    # Data / NCBI
    data_dir: str = "./data"
    ncbi_email: str = ""
    ncbi_api_key: str = ""

    # Timeout for readiness probes against backing services
    probe_timeout_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
