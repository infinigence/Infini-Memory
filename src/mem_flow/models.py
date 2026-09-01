"""Pydantic domain models shared by all mem_flow components."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def as_beijing_time(value: datetime) -> datetime:
    """Normalize a recorded datetime to Beijing time.

    Historical metadata can contain UTC offsets or no offset at all. A naive value
    is interpreted as an already-local Beijing wall-clock time; aware values retain
    their instant and are converted to Asia/Shanghai.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=BEIJING_TIMEZONE)
    return value.astimezone(BEIJING_TIMEZONE)


def beijing_now() -> datetime:
    return datetime.now(BEIJING_TIMEZONE)


def compact_beijing_timestamp(*, microseconds: bool = False) -> str:
    """Return a key-safe Beijing timestamp with an explicit ``+0800`` offset."""

    pattern = "%Y%m%dT%H%M%S%f%z" if microseconds else "%Y%m%dT%H%M%S%z"
    return beijing_now().strftime(pattern)


class MemoryScope(BaseModel):
    """Logical S3 namespace for one store and one user."""

    store_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)

    @field_validator("store_id", "user_id")
    @classmethod
    def validate_path_segment(cls, value: str) -> str:
        value = value.strip()
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError(
                "store_id and user_id must be safe, non-empty path segments"
            )
        return value


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)
    # Optional source coordinates let batch evaluators retain exact session/turn
    # provenance without changing ordinary chat callers.
    source_id: str | None = None
    observed_at: datetime | date | None = None
    turn_index: int | None = Field(default=None, ge=0)


class LLMRequest(BaseModel):
    operation: str
    messages: list[ChatMessage]
    temperature: float | None = None


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str
    kind: Literal[
        "current",
        "raw",
        "rewrite",
        "evidence",
        "memory",
        "directory_topic",
        "directory_summary",
        "topic",
    ]
    summary: str = ""
    title: str = ""
    directory_id: str = ""
    document_count: int = Field(default=0, ge=0)
    content_digest: str = ""
    store_id: str = ""
    user_id: str = ""
    created_at: datetime = Field(default_factory=beijing_now)
    updated_at: datetime = Field(default_factory=beijing_now)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_recorded_time(cls, value: datetime) -> datetime:
        return as_beijing_time(value)


class MemoryDocument(BaseModel):
    metadata: DocumentMetadata
    content: str
    key: str = ""


class ExtractionRequest(BaseModel):
    messages: list[ChatMessage]
    infer: bool = True
    # Callers that need a deterministic logical order (for example offline
    # evaluation) may provide it explicitly. Online mem_flow callers omit this
    # value and retain the extraction-time Unix timestamp behavior.
    sequence: int | None = Field(default=None, ge=0)


class ExtractionResult(BaseModel):
    instance_id: str
    current_key: str
    sequence_timestamp: int = Field(ge=0)
    rotated_key: str | None = None
    extracted_content: str
    evidence_key: str = ""
    appended: bool
    bytes_written: int = Field(ge=0)
    tokens: int = Field(default=0, ge=0)


class MaintenanceRequest(BaseModel):
    # Deprecated compatibility input. Hierarchical mem_flow never merges leaf files.
    merge_topics: bool | None = None
    # Rewrite batches and immutable leaf summaries are independent. Keep the
    # public mem_flow default sequential while allowing batch callers to opt in.
    workers: int = Field(default=1, ge=1, le=64)


class DocReadRequest(BaseModel):
    """Locate one maintained memory leaf by its document id."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        value = value.strip()
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("document_id must be a safe, non-empty id")
        return value


class DocDocument(BaseModel):
    """Body-only view of a document stored below ``doc/``."""

    document_id: str
    content: str
    line_count: int = Field(ge=0)


class DocLineUpdateRequest(DocReadRequest):
    """Replace an inclusive, one-based range in a doc memory body."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    replacement: str

    @model_validator(mode="after")
    def validate_line_range(self) -> "DocLineUpdateRequest":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class DocLineUpdateResult(BaseModel):
    document: DocDocument
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    replacement_line_count: int = Field(ge=0)
    changed: bool


class MaintenanceResult(BaseModel):
    current_documents: int = 0
    raw_documents: int = 0
    rewrite_documents: int = 0
    directories_created: int = 0
    memory_documents_created: int = 0
    directory_topics_updated: int = 0
    deleted_current_keys: list[str] = Field(default_factory=list)
    deleted_raw_keys: list[str] = Field(default_factory=list)
    deleted_rewrite_keys: list[str] = Field(default_factory=list)
    deleted_route_keys: list[str] = Field(default_factory=list)

    @property
    def topics_created(self) -> int:
        """Compatibility alias for callers migrating from the flat layout."""
        return self.memory_documents_created

    @property
    def topics_updated(self) -> int:
        return 0

    @property
    def topics_merged(self) -> int:
        return 0

    @property
    def directory_summaries_updated(self) -> int:
        """Compatibility alias for callers migrating from SUMMARY.md."""
        return self.directory_topics_updated


class LegacyMigrationItem(BaseModel):
    document_id: str
    source_key: str
    directory_id: str
    target_key: str
    action: Literal["planned", "copied", "existing"]


class LegacyMigrationResult(BaseModel):
    scanned: int = 0
    copied: int = 0
    deleted_sources: int = 0
    directory_topics_updated: int = 0
    items: list[LegacyMigrationItem] = Field(default_factory=list)

    @property
    def directory_summaries_updated(self) -> int:
        """Compatibility alias for callers migrating from SUMMARY.md."""
        return self.directory_topics_updated


class SearchSource(StrEnum):
    EVIDENCE = "evidence"
    CURRENT = "current"
    RAW = "raw"
    REWRITE = "rewrite"
    DOC = "doc"


class SearchStrategy(StrEnum):
    """Retrieval algorithms exposed by mem_flow."""

    AUTO = "AUTO"
    HIERARCHICAL = "HIERARCHICAL"
    LLM = "LLM"
    BM25 = "BM25"
    BM25_PARTITION = "BM25_partition"
    LLM_AND_BM25 = "LLM_and_BM25"
    LLM_AND_BM25_PARTITION = "LLM_and_BM25_partition"
    LLM_OR_BM25 = "LLM_or_BM25"
    FOLDER_BM25_PARTITION = "FOLDER_BM25_partition"
    AGENTIC = "AGENTIC"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    directory_limit: int = Field(default=5, ge=1, le=50)
    sources: set[SearchSource] = Field(default_factory=lambda: set(SearchSource))
    strategy: SearchStrategy | None = None
    answer: bool = True
    as_of: datetime | date | None = None
    debug: bool = False


class SearchScopeSelection(BaseModel):
    working_document_ids: list[str] = Field(default_factory=list)
    directory_ids: list[str] = Field(default_factory=list)


class DocumentSelection(BaseModel):
    document_ids: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    id: str
    kind: Literal["evidence", "current", "raw", "rewrite", "memory"]
    directory_id: str | None = None
    title: str = ""
    summary: str = ""
    content: str
    key: str
    source_ids: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    query: str
    strategy: SearchStrategy = SearchStrategy.HIERARCHICAL
    hits: list[SearchHit] = Field(default_factory=list)
    answer: str | None = None
    route: str = ""
    coverage: dict[str, object] | None = None
    execution: dict[str, object] | None = None
    abstention_reason: str | None = None


__all__ = [
    "ChatMessage",
    "DocDocument",
    "DocLineUpdateRequest",
    "DocLineUpdateResult",
    "DocReadRequest",
    "DocumentMetadata",
    "ExtractionRequest",
    "ExtractionResult",
    "LLMRequest",
    "LegacyMigrationItem",
    "LegacyMigrationResult",
    "MaintenanceRequest",
    "MaintenanceResult",
    "MemoryDocument",
    "MemoryScope",
    "DocumentSelection",
    "SearchHit",
    "SearchRequest",
    "SearchResult",
    "SearchScopeSelection",
    "SearchSource",
    "SearchStrategy",
]
