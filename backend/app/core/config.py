"""Typed environment-based application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_name: str = "Adaptive Agentic RAG"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    # Keep this scalar so `ALLOWED_ORIGINS=http://localhost:3000,...` works in `.env`.
    allowed_origins: str = "http://localhost:3000"
    max_upload_size_mb: int = Field(default=50, ge=1, le=1024)
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)

    # Local provider and persistence settings are initialized by the application container.
    ollama_base_url: AnyHttpUrl = "http://host.docker.internal:11434"
    ollama_model: str = "qwen2.5:7b"
    chroma_persist_directory: str = "database/chroma"
    chroma_collection_name: str = "rag_chunks"
    metadata_database_url: str = "sqlite:///database/metadata/app.db"
    upload_directory: str = "database/uploads"
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(default=16, ge=1, le=128)
    reranker_top_k: int = Field(default=5, ge=1, le=50)
    agent_confidence_threshold: float = Field(default=0.65, ge=0, le=1)
    agent_max_retries: int = Field(default=2, ge=0, le=10)
    generation_prompt_version: str = "v1"
    generation_context_max_chars: int = Field(default=12000, ge=1000, le=100000)
    generation_max_output_tokens: int = Field(default=512, ge=64, le=4096)
    ollama_timeout_seconds: float = Field(default=120, ge=1, le=600)
    verification_support_threshold: float = Field(default=0.65, ge=0, le=1)
    verification_min_coverage: float = Field(default=1.0, ge=0, le=1)
    verification_max_retries: int = Field(default=1, ge=0, le=5)
    verification_batch_size: int = Field(default=16, ge=1, le=128)

    @computed_field
    @property
    def allowed_origin_list(self) -> list[str]:
        """Return normalized CORS origins from a comma-separated environment value."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def metadata_database_path(self) -> Path:
        """Resolve the supported SQLite URL form to a filesystem path."""
        prefix = "sqlite:///"
        if not self.metadata_database_url.startswith(prefix):
            raise ValueError("METADATA_DATABASE_URL must use the sqlite:/// URL scheme.")
        return Path(self.metadata_database_url.removeprefix(prefix))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton validated settings instance."""
    return Settings()
