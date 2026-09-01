"""Typed, rebuildable projections of immutable Markdown memory facts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TemporalPrecision(StrEnum):
    EXACT = "exact"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    SEASON = "season"
    YEAR = "year"
    FUZZY = "fuzzy"


class TemporalValue(BaseModel):
    start: date | datetime
    end: date | datetime
    precision: TemporalPrecision = TemporalPrecision.DAY
    raw_expression: str = ""
    anchor: Literal["event", "valid", "observed", "question"] = "event"
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_range(self) -> "TemporalValue":
        if self.end < self.start:
            raise ValueError("temporal end must not precede start")
        return self


class FactStatus(StrEnum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    HYPOTHETICAL = "hypothetical"
    RECOMMENDED = "recommended"
    UNKNOWN = "unknown"


class FactModality(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    ASSISTANT_SUGGESTION = "assistant_suggestion"
    SYSTEM_GENERATED = "system_generated"


class FactValue(BaseModel):
    kind: Literal["text", "entity", "number", "money", "duration", "date", "bool"]
    raw: str
    normalized: str = ""
    number: float | None = None
    unit: str = ""
    currency: str = ""


class EvidenceSpan(BaseModel):
    evidence_id: str
    source_session_id: str = ""
    message_index: int | None = Field(default=None, ge=0)
    role: Literal["user", "assistant", "system", "tool", "unknown"] = "unknown"
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    quote_digest: str = ""


class StructuredFact(BaseModel):
    fact_id: str
    subject_entity_id: str
    predicate: str
    object: FactValue
    qualifiers: dict[str, FactValue] = Field(default_factory=dict)
    status: FactStatus = FactStatus.UNKNOWN
    modality: FactModality = FactModality.EXPLICIT
    valid_from: TemporalValue | None = None
    valid_to: TemporalValue | None = None
    event_time: TemporalValue | None = None
    observed_at: TemporalValue | None = None
    source_role: str = "user"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_document_id: str
    source_document_key: str = ""
    source_document_digest: str
    source_kind: str = "memory"
    line_number: int = Field(ge=1)
    heading: str = ""
    text: str
    markdown: str
    sequence: int | None = None

    @property
    def search_text(self) -> str:
        return " ".join(
            item
            for item in (
                self.heading,
                self.subject_entity_id,
                self.predicate,
                self.object.normalized,
                self.text,
            )
            if item
        )


class EntityAlias(BaseModel):
    value: str
    normalized: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_fact_ids: list[str] = Field(default_factory=list)


class EntityRecord(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str = "unknown"
    aliases: list[EntityAlias] = Field(default_factory=list)
    evidence_fact_ids: list[str] = Field(default_factory=list)


class FactRelation(BaseModel):
    relation_id: str
    source_fact_id: str
    target_fact_id: str
    kind: Literal[
        "same_event", "supersedes", "contradicts", "extends", "derives", "supports"
    ]
    confidence: float = Field(default=1.0, ge=0, le=1)
    evidence_fact_ids: list[str] = Field(default_factory=list)


class IndexSource(BaseModel):
    document_id: str
    document_key: str
    document_digest: str
    artifact_key: str
    artifact_digest: str
    fact_count: int = Field(ge=0)


class IndexManifest(BaseModel):
    schema_version: int = Field(ge=1)
    digest: str
    sources: list[IndexSource] = Field(default_factory=list)
    relation_artifact_key: str = ""
    relation_artifact_digest: str = ""


class ActiveIndexPointer(BaseModel):
    schema_version: int = Field(ge=1)
    manifest_key: str
    manifest_digest: str


class FactIndexSnapshot(BaseModel):
    facts: list[StructuredFact] = Field(default_factory=list)
    entities: list[EntityRecord] = Field(default_factory=list)
    relations: list[FactRelation] = Field(default_factory=list)
    manifest: IndexManifest | None = None
    persisted: bool = False


__all__ = [
    "ActiveIndexPointer",
    "EntityAlias",
    "EntityRecord",
    "EvidenceSpan",
    "FactIndexSnapshot",
    "FactModality",
    "FactRelation",
    "FactStatus",
    "FactValue",
    "IndexManifest",
    "IndexSource",
    "StructuredFact",
    "TemporalPrecision",
    "TemporalValue",
]
