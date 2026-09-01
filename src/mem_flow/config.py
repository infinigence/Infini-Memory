"""Validated configuration for mem_flow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from .models import SearchStrategy


class S3Config(BaseModel):
    endpoint_url: str = Field(min_length=1)
    bucket: str = Field(default="infini-memory", min_length=1)
    access_key: SecretStr = SecretStr("")
    secret_key: SecretStr = SecretStr("")
    region: str = "us-east-1"
    fixed_prefix: str = "inf_mem"
    ensure_bucket: bool = False

    @field_validator("fixed_prefix")
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        value = value.strip(" /")
        if not value or any(part in {".", ".."} for part in value.split("/")):
            raise ValueError("fixed_prefix must be a safe, non-empty S3 prefix")
        return value


class StorageConfig(BaseModel):
    """Select the physical store used by mem_flow."""

    type: Literal["s3", "local"] = "s3"
    path: Path = Path("data/mem_flow")

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("local storage path must not be empty")
        return value


class LLMConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = Field(default="deepseek-v4-flash-0731", min_length=1)
    temperature: float = Field(default=0.1, ge=0, le=2)
    retry_attempts: int = Field(default=5, ge=1, le=20)
    # Structured-output validation has deterministic fallbacks in the
    # maintainer.  Keep its budget separate from transport retries so a valid
    # HTTP response with malformed JSON cannot multiply a slow MaaS call five
    # times before that fallback is allowed to run.
    structured_retry_attempts: int = Field(default=2, ge=1, le=20)
    retry_initial_seconds: float = Field(default=1.0, ge=0)
    retry_max_seconds: float = Field(default=8.0, ge=0)
    retry_jitter: float = Field(default=0.2, ge=0, le=1)
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    # Bound concurrent MaaS requests made by one MemFlow composition. The
    # evaluator supplies one shared adaptive limiter across all sample flows.
    max_concurrency: int = Field(default=50, ge=1, le=64)


class ExtractionConfig(BaseModel):
    current_max_tokens: int = Field(default=5000, ge=1)


class MaintenanceConfig(BaseModel):
    rewrite_batch_max_tokens: int = Field(default=12000, ge=1)
    document_summary_tokens: int = Field(default=100, ge=20)
    # Lossless high-throughput ingestion can compact and route code-owned facts
    # deterministically. The ordinary LLM-assisted maintenance path remains the
    # default for production callers.
    deterministic: bool = False


class IndexConfig(BaseModel):
    """Rebuildable structured sidecar settings."""

    enabled: bool = True
    schema_version: int = Field(default=1, ge=1)
    embedding_enabled: bool = True
    relation_confidence_threshold: float = Field(default=0.85, ge=0, le=1)


class RetrievalConfig(BaseModel):
    strategy: SearchStrategy = SearchStrategy.HIERARCHICAL
    directory_limit: int = Field(default=5, ge=1, le=50)
    working_document_preview_chars: int = Field(default=500, ge=50)
    working_candidate_limit: int = Field(default=20, ge=1, le=200)
    directory_topic_preview_chars: int = Field(default=1000, ge=100)
    search_folder: str = "doc"
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    fact_candidate_limit: int = Field(default=100, ge=1, le=1000)
    rerank_candidate_limit: int = Field(default=40, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1)
    subgoal_min_candidates: int = Field(default=1, ge=1, le=20)
    coverage_expand_rounds: int = Field(default=2, ge=0, le=10)
    evidence_sufficiency_threshold: float = Field(default=0.7, ge=0, le=1)
    agentic_max_iterations: int = Field(default=7, ge=1, le=30)
    agentic_min_relevant_docs: int = Field(default=1, ge=1)
    agentic_grep_limit: int = Field(default=30, ge=1)
    agentic_read_lines_max_range: int = Field(default=100, ge=1)
    agentic_list_docs_page_size: int = Field(default=20, ge=1)
    agentic_catalog_summary_length: int = Field(default=200, ge=1)
    agentic_min_result_tokens: int = Field(default=500, ge=0)
    # Bound the evidence passed to the answer model after Agentic has selected
    # documents. Ranking individual fact lines retains exact evidence while
    # avoiding abstentions caused by thousands of unrelated facts in a leaf.
    agentic_answer_max_lines_per_document: int = Field(default=16, ge=1, le=200)
    # Aggregation queries need a wider view because synonymous events such as
    # attended, visited, and volunteered can rank far apart inside one leaf.
    agentic_aggregation_max_lines_per_document: int = Field(
        default=12, ge=1, le=200
    )


class MetricsConfig(BaseModel):
    enabled: bool = True
    namespace: str = "mem_flow"


class MemFlowConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    s3: S3Config | None = None
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    maintenance: MaintenanceConfig = Field(default_factory=MaintenanceConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @model_validator(mode="after")
    def require_selected_backend_config(self) -> "MemFlowConfig":
        if self.storage.type == "s3" and self.s3 is None:
            raise ValueError("s3 configuration is required when storage.type is 's3'")
        return self

    @property
    def storage_prefix(self) -> str:
        """Return the physical key prefix used outside the bound user scope."""
        if self.storage.type == "local":
            return ""
        assert self.s3 is not None
        return self.s3.fixed_prefix


__all__ = [
    "IndexConfig",
    "LLMConfig",
    "ExtractionConfig",
    "MaintenanceConfig",
    "MemFlowConfig",
    "MetricsConfig",
    "RetrievalConfig",
    "S3Config",
    "StorageConfig",
]
