"""Parallel lexical, semantic, entity, temporal and state-aware fact recall."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime

from pydantic import BaseModel, Field

from ..config import RetrievalConfig
from ..index.models import FactIndexSnapshot, StructuredFact
from ..models import MemoryDocument
from .coverage import CoverageReport, assess_coverage, normalized_concepts
from .planner import QueryIntent, QueryPlan
from .ranking import bm25_rank, tokenize


class HybridCandidate(BaseModel):
    fact_id: str
    score: float
    channels: list[str] = Field(default_factory=list)


class HybridSearchResult(BaseModel):
    facts: list[StructuredFact] = Field(default_factory=list)
    documents: list[MemoryDocument] = Field(default_factory=list)
    coverage: CoverageReport
    candidates: list[HybridCandidate] = Field(default_factory=list)


_SEMANTIC_FAMILIES = (
    {"buy", "bought", "purchase", "purchased", "acquire", "acquired", "order", "ordered"},
    {"attend", "attended", "visit", "visited", "participate", "participated", "joined", "went"},
    {"finish", "finished", "complete", "completed", "done"},
    {"like", "likes", "love", "loves", "prefer", "prefers", "enjoy", "enjoys"},
    {"cost", "spent", "paid", "price", "amount"},
    {"cancelled", "canceled", "stopped", "ended"},
)
_CANONICAL = {
    token: sorted(family)[0]
    for family in _SEMANTIC_FAMILIES
    for token in family
}


def _semantic_counter(value: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for token in tokenize(value):
        canonical = _CANONICAL.get(token, token[:5] if len(token) >= 6 else token)
        result[canonical] += 1
    return result


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right[key] for key, value in left.items())
    denominator = math.sqrt(sum(value * value for value in left.values())) * math.sqrt(
        sum(value * value for value in right.values())
    )
    return numerator / denominator if denominator else 0.0


def _fact_date(fact: StructuredFact) -> date | None:
    value = fact.event_time or fact.observed_at
    if value is None:
        return None
    return value.start.date() if isinstance(value.start, datetime) else value.start


def _rank_by_score(scores: dict[int, float]) -> list[int]:
    return [index for index, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0]


def hybrid_recall(
    plan: QueryPlan,
    snapshot: FactIndexSnapshot,
    documents: list[MemoryDocument],
    *,
    limit: int,
    config: RetrievalConfig,
    semantic_enabled: bool = True,
) -> HybridSearchResult:
    facts = snapshot.facts
    if not facts:
        return HybridSearchResult(coverage=assess_coverage(plan, []))
    query_text = " ".join(
        dict.fromkeys(
            [plan.body]
            + [concept for goal in plan.subgoals for concept in goal.expanded_concepts]
        )
    )
    channel_ranks: dict[str, list[int]] = {}
    lexical = bm25_rank(
        query_text,
        [fact.search_text for fact in facts],
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    channel_ranks["bm25"] = [index for index, _score in lexical]

    if semantic_enabled:
        query_vector = _semantic_counter(query_text)
        semantic_scores = {
            index: _cosine(query_vector, _semantic_counter(fact.search_text))
            for index, fact in enumerate(facts)
        }
        channel_ranks["semantic"] = _rank_by_score(semantic_scores)

    concepts = normalized_concepts(
        " ".join(concept for goal in plan.subgoals for concept in goal.concepts)
    )
    entity_scores = {
        index: float(len(concepts & normalized_concepts(fact.search_text)))
        for index, fact in enumerate(facts)
    }
    channel_ranks["entity"] = _rank_by_score(entity_scores)

    if plan.target_date is not None:
        temporal_scores = {
            index: 1.0 / (1.0 + abs((fact_date - plan.target_date).days) / 7.0)
            for index, fact in enumerate(facts)
            if (fact_date := _fact_date(fact)) is not None
        }
        channel_ranks["temporal"] = _rank_by_score(temporal_scores)

    if plan.intent in {QueryIntent.STATE, QueryIntent.PREFERENCE}:
        state_scores = {
            index: float(fact.sequence or 0)
            for index, fact in enumerate(facts)
            if fact.status.value not in {"cancelled", "failed"}
        }
        channel_ranks["state"] = _rank_by_score(state_scores)

    fused: dict[int, float] = defaultdict(float)
    channels: dict[int, set[str]] = defaultdict(set)
    for channel, ranking in channel_ranks.items():
        for rank, index in enumerate(ranking[: config.fact_candidate_limit], start=1):
            fused[index] += 1.0 / (config.rrf_k + rank)
            channels[index].add(channel)
    ranked = sorted(fused, key=lambda index: (-fused[index], index))

    # Reserve one source per independently requested subgoal before global fill.
    selected: list[int] = []
    for goal in plan.subgoals:
        goal_terms = normalized_concepts(" ".join(goal.expanded_concepts))
        candidates = [
            index
            for index in ranked
            if goal_terms & normalized_concepts(facts[index].search_text)
        ]
        for index in candidates[: goal.minimum_candidates]:
            if index not in selected:
                selected.append(index)
    budget = max(limit, len(selected))
    for index in ranked:
        if len(selected) >= budget:
            break
        if index not in selected:
            selected.append(index)

    selected_facts = [facts[index] for index in selected]
    coverage = assess_coverage(plan, selected_facts)
    for _round in range(config.coverage_expand_rounds):
        if coverage.complete or len(selected) >= config.rerank_candidate_limit:
            break
        previous_count = len(selected)
        expanded_budget = min(
            config.rerank_candidate_limit,
            max(previous_count + limit, previous_count + 1),
        )
        for index in ranked:
            if len(selected) >= expanded_budget:
                break
            if index not in selected:
                selected.append(index)
        if len(selected) == previous_count:
            break
        selected_facts = [facts[index] for index in selected]
        coverage = assess_coverage(plan, selected_facts)
    document_by_id = {item.metadata.id: item for item in documents}
    selected_documents: list[MemoryDocument] = []
    seen_documents: set[str] = set()
    for fact in selected_facts:
        document = document_by_id.get(fact.source_document_id)
        if document and document.metadata.id not in seen_documents:
            selected_documents.append(document)
            seen_documents.add(document.metadata.id)
        if len(selected_documents) >= limit:
            break
    return HybridSearchResult(
        facts=selected_facts,
        documents=selected_documents,
        coverage=coverage,
        candidates=[
            HybridCandidate(
                fact_id=facts[index].fact_id,
                score=fused[index],
                channels=sorted(channels[index]),
            )
            for index in selected
        ],
    )


__all__ = ["HybridCandidate", "HybridSearchResult", "hybrid_recall"]
