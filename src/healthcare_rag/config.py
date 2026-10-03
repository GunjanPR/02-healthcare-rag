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
    gateway_base_url: str = "http://localhost:8000/v1"
    gateway_api_key: str = ""
    chat_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"

    # OpenSearch (BM25)
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "pmc_chunks"
    opensearch_user: str = ""
    opensearch_password: str = ""

    # Qdrant (dense vectors)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pmc_chunks"
    qdrant_api_key: str = ""

    # Timeout for readiness probes against backing services
    probe_timeout_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
