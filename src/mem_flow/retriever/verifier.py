"""Origin rehydration and evidence sufficiency checks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..index.models import FactRelation, StructuredFact
from ..models import MemoryDocument
from .coverage import CoverageReport


class EvidenceBundle(BaseModel):
    documents: list[MemoryDocument] = Field(default_factory=list)
    hydrated_origins: list[str] = Field(default_factory=list)
    missing_origins: list[str] = Field(default_factory=list)
    sufficiency_score: float = Field(default=0, ge=0, le=1)
    sufficient: bool = False
    reason: str = ""


def verify_evidence(
    facts: list[StructuredFact],
    documents: list[MemoryDocument],
    coverage: CoverageReport,
    *,
    threshold: float,
    allow_assistant_only: bool = False,
    relations: list[FactRelation] | None = None,
    block_on_conflicts: bool = True,
) -> EvidenceBundle:
    evidence_by_id = {
        item.metadata.id: item for item in documents if item.metadata.kind == "evidence"
    }
    origins = list(
        dict.fromkeys(
            span.evidence_id
            for fact in facts
            for span in fact.evidence
            if span.evidence_id
        )
    )
    hydrated = [origin for origin in origins if origin in evidence_by_id]
    missing = [origin for origin in origins if origin not in evidence_by_id]
    if origins:
        provenance_score = len(hydrated) / len(origins)
    else:
        # Legacy facts have no origin; keep them usable but never award full
        # provenance confidence.
        provenance_score = 0.5
    assistant_only = bool(facts) and all(
        fact.modality.value == "assistant_suggestion" for fact in facts
    )
    source_score = 1.0
    if assistant_only:
        source_score = 0.25
    score = 0.6 * coverage.score + 0.3 * provenance_score + 0.1 * source_score
    selected_fact_ids = {fact.fact_id for fact in facts}
    conflicts = [
        relation.relation_id
        for relation in relations or []
        if relation.kind == "contradicts"
        and relation.confidence >= 0.85
        and relation.source_fact_id in selected_fact_ids
        and relation.target_fact_id in selected_fact_ids
    ]
    coverage.unresolved_conflicts = conflicts
    sufficient = coverage.complete and score >= threshold and not missing
    if assistant_only and not allow_assistant_only:
        sufficient = False
    if conflicts and block_on_conflicts:
        sufficient = False
    reason = ""
    if conflicts and block_on_conflicts:
        reason = "unresolved_conflicts:" + ",".join(conflicts)
    elif assistant_only and not allow_assistant_only:
        reason = "assistant_suggestion_only"
    elif coverage.missing_subgoals:
        reason = "missing_subgoals:" + ",".join(coverage.missing_subgoals)
    elif missing:
        reason = "missing_origins:" + ",".join(missing)
    elif not sufficient:
        reason = "evidence_below_threshold"
    return EvidenceBundle(
        documents=[evidence_by_id[item] for item in hydrated],
        hydrated_origins=hydrated,
        missing_origins=missing,
        sufficiency_score=score,
        sufficient=sufficient,
        reason=reason,
    )


__all__ = ["EvidenceBundle", "verify_evidence"]
