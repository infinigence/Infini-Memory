"""Coverage reports that stop retrieval only after every required subgoal is met."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..index.models import StructuredFact
from .planner import QueryPlan
from .ranking import tokenize


def normalized_concepts(value: str) -> set[str]:
    """Return lightweight morphological forms for deterministic coverage checks."""

    normalized: set[str] = set()
    for token in tokenize(value):
        normalized.add(token)
        if len(token) > 4 and token.endswith("ies"):
            normalized.add(token[:-3] + "y")
        if len(token) > 4 and token.endswith("ing"):
            normalized.add(token[:-3])
        if len(token) > 3 and token.endswith("ed"):
            normalized.add(token[:-2])
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            normalized.add(token[:-1])
    return normalized


class SubgoalCoverage(BaseModel):
    subgoal_id: str
    matched_fact_ids: list[str] = Field(default_factory=list)
    matched_concepts: list[str] = Field(default_factory=list)
    required_concepts: list[str] = Field(default_factory=list)
    complete: bool = False


class CoverageReport(BaseModel):
    subgoals: list[SubgoalCoverage] = Field(default_factory=list)
    complete: bool = False
    score: float = Field(default=0, ge=0, le=1)
    missing_subgoals: list[str] = Field(default_factory=list)
    unresolved_entities: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)


def assess_coverage(plan: QueryPlan, facts: list[StructuredFact]) -> CoverageReport:
    results: list[SubgoalCoverage] = []
    for goal in plan.subgoals:
        required = normalized_concepts(" ".join(goal.concepts))
        expanded = normalized_concepts(" ".join(goal.expanded_concepts))
        matched_facts: list[str] = []
        matched_concepts: set[str] = set()
        for fact in facts:
            tokens = normalized_concepts(fact.search_text)
            matches = expanded & tokens
            if matches:
                matched_facts.append(fact.fact_id)
                matched_concepts.update(required & tokens)
        concept_threshold = 1 if len(required) <= 2 else max(1, (len(required) + 1) // 2)
        complete = (
            len(set(matched_facts)) >= goal.minimum_candidates
            and len(matched_concepts) >= concept_threshold
        )
        results.append(
            SubgoalCoverage(
                subgoal_id=goal.id,
                matched_fact_ids=list(dict.fromkeys(matched_facts)),
                matched_concepts=sorted(matched_concepts),
                required_concepts=sorted(required),
                complete=complete,
            )
        )
    required_results = [
        result
        for goal, result in zip(plan.subgoals, results, strict=True)
        if goal.required
    ]
    complete_count = sum(item.complete for item in required_results)
    score = complete_count / len(required_results) if required_results else 0.0
    return CoverageReport(
        subgoals=results,
        complete=bool(required_results) and complete_count == len(required_results),
        score=score,
        missing_subgoals=[item.subgoal_id for item in required_results if not item.complete],
    )


__all__ = [
    "CoverageReport",
    "SubgoalCoverage",
    "assess_coverage",
    "normalized_concepts",
]
