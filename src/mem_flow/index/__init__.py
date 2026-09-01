"""Rebuildable structured sidecars derived from Markdown memory facts."""

from .models import (
    ActiveIndexPointer,
    EntityAlias,
    EntityRecord,
    EvidenceSpan,
    FactIndexSnapshot,
    FactModality,
    FactRelation,
    FactStatus,
    FactValue,
    IndexManifest,
    IndexSource,
    StructuredFact,
    TemporalPrecision,
    TemporalValue,
)
from .projector import assign_fact_ids, project_documents
from .snapshot import FactIndexStore

__all__ = [
    "ActiveIndexPointer",
    "EntityAlias",
    "EntityRecord",
    "EvidenceSpan",
    "FactIndexSnapshot",
    "FactIndexStore",
    "FactModality",
    "FactRelation",
    "FactStatus",
    "FactValue",
    "IndexManifest",
    "IndexSource",
    "StructuredFact",
    "TemporalPrecision",
    "TemporalValue",
    "assign_fact_ids",
    "project_documents",
]
