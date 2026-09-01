"""Generic, validated query plans for deterministic memory operations."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from .fact_index import build_fact_query_plan
from .ranking import tokenize


class QueryIntent(StrEnum):
    DIRECT = "direct"
    AGGREGATE = "aggregate"
    TEMPORAL = "temporal"
    STATE = "state"
    PREFERENCE = "preference"
    COMPARISON = "comparison"
    ABSTENTION_CHECK = "abstention_check"
    AMBIGUOUS = "ambiguous"


class QueryOperator(StrEnum):
    FILTER = "FILTER"
    TEMPORAL_WINDOW = "TEMPORAL_WINDOW"
    GROUP = "GROUP"
    DEDUP = "DEDUP"
    SORT = "SORT"
    COUNT = "COUNT"
    SUM = "SUM"
    DIFFERENCE = "DIFFERENCE"
    LATEST = "LATEST"
    FIRST = "FIRST"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    ELAPSED = "ELAPSED"
    CHRONOLOGICAL_ORDER = "CHRONOLOGICAL_ORDER"
    ARGMAX = "ARGMAX"
    ARGMIN = "ARGMIN"
    VERIFY = "VERIFY"


class QuerySubgoal(BaseModel):
    id: str
    concepts: list[str] = Field(default_factory=list)
    expanded_concepts: list[str] = Field(default_factory=list)
    required: bool = True
    minimum_candidates: int = Field(default=1, ge=1)
    source_role: str = ""
    statuses: list[str] = Field(default_factory=list)


class QueryStep(BaseModel):
    operator: QueryOperator
    params: dict[str, object] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    normalized_query: str
    body: str
    as_of: date | datetime | None = None
    target_date: date | None = None
    intent: QueryIntent
    subgoals: list[QuerySubgoal]
    steps: list[QueryStep]
    answer_type: str = "text"
    planner_confidence: float = Field(default=1.0, ge=0, le=1)


_STOPWORDS = {
    "about", "after", "again", "ago", "all", "and", "before", "between",
    "could", "current", "currently", "date", "did", "does", "from", "have",
    "how", "into", "many", "most", "much", "question", "should", "since",
    "that", "their", "then", "there", "these", "they", "this", "total",
    "what", "when", "where", "which", "with", "would", "your", "mine",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
}
_EXPANSIONS = {
    "bought": ("purchased", "acquired", "ordered"),
    "buy": ("purchase", "acquire", "order"),
    "attended": ("participated", "visited", "joined", "went"),
    "finished": ("completed", "done"),
    "likes": ("prefers", "loves", "enjoys"),
    "preference": ("prefers", "likes", "avoids", "requires"),
    "spent": ("paid", "cost", "price", "amount"),
    "latest": ("current", "now", "recent"),
    "cancelled": ("canceled", "called", "off"),
}


def _concepts(body: str) -> list[str]:
    return list(
        dict.fromkeys(
            token
            for token in tokenize(body)
            if len(token) >= 3 and token not in _STOPWORDS and not token.isdigit()
        )
    )


def _subgoals(body: str, concepts: list[str], minimum: int) -> list[QuerySubgoal]:
    # Conjunctions are the common source of incomplete cross-session recall.
    clauses = re.split(
        r"\s+(?:and|before|after|versus|vs\.?|compared with)\s+",
        body,
        flags=re.I,
    )
    clause_concepts = [_concepts(clause) for clause in clauses]
    clause_concepts = [items for items in clause_concepts if items]
    if len(clause_concepts) < 2:
        clause_concepts = [concepts]
    goals: list[QuerySubgoal] = []
    requested_statuses: list[str] = []
    if re.search(r"\b(?:plan|planned|planning|upcoming|intend|will)\b", body, re.I):
        requested_statuses.append("planned")
    if re.search(r"\b(?:recommend|recommended|suggest|suggested)\b", body, re.I):
        requested_statuses.append("recommended")
    if re.search(r"\b(?:cancel|cancelled|canceled|called off)\b", body, re.I):
        requested_statuses.append("cancelled")
    for index, items in enumerate(clause_concepts, start=1):
        expanded = list(items)
        for concept in items:
            expanded.extend(_EXPANSIONS.get(concept, ()))
        goals.append(
            QuerySubgoal(
                id=f"g{index}",
                concepts=items,
                expanded_concepts=list(dict.fromkeys(expanded)),
                minimum_candidates=minimum,
                source_role=(
                    "assistant"
                    if re.search(
                        r"\b(?:assistant|you (?:said|suggested|recommended|wrote))\b",
                        body,
                        re.I,
                    )
                    else "user"
                ),
                statuses=requested_statuses,
            )
        )
    return goals


def compile_query_plan(
    query: str,
    *,
    as_of: date | datetime | None = None,
    subgoal_min_candidates: int = 1,
) -> QueryPlan:
    legacy = build_fact_query_plan(query)
    body = legacy.body
    normalized = " ".join(query.split())
    effective_as_of = as_of or legacy.question_date
    lowered = body.casefold()
    concepts = _concepts(body)

    steps = [QueryStep(operator=QueryOperator.FILTER)]
    answer_type = "text"
    if legacy.temporal:
        steps.append(QueryStep(operator=QueryOperator.TEMPORAL_WINDOW))
    steps.extend(
        [
            QueryStep(operator=QueryOperator.GROUP, params={"by": "same_event"}),
            QueryStep(operator=QueryOperator.DEDUP, params={"by": "same_event"}),
        ]
    )

    if re.search(r"\bhow many (?:days|weeks|months|years)\b", lowered):
        intent = QueryIntent.TEMPORAL
        operator = QueryOperator.ELAPSED
        answer_type = "duration"
        unit_match = re.search(r"\b(days|weeks|months|years)\b", lowered)
        params = {"unit": unit_match.group(1) if unit_match else "days"}
    elif re.search(r"\b(?:chronological|in order|what order|which came first)\b", lowered):
        intent = QueryIntent.TEMPORAL
        operator = QueryOperator.CHRONOLOGICAL_ORDER
        answer_type = "ordered_list"
        params = {}
    elif re.search(r"\b(?:how many|count|number of)\b", lowered):
        intent = QueryIntent.AGGREGATE
        operator = QueryOperator.COUNT
        answer_type = "number"
        params = {}
    elif re.search(r"\b(?:difference|how much more|how much less|increase|decrease)\b", lowered):
        intent = QueryIntent.COMPARISON
        operator = QueryOperator.DIFFERENCE
        answer_type = "number"
        params = {}
    elif re.search(r"\b(?:total|sum|altogether|combined)\b", lowered):
        intent = QueryIntent.AGGREGATE
        operator = QueryOperator.SUM
        answer_type = "number"
        params = {}
    elif re.search(r"\b(?:most|highest|maximum|largest)\b", lowered):
        intent = QueryIntent.COMPARISON
        operator = QueryOperator.ARGMAX
        params = {}
    elif re.search(r"\b(?:least|lowest|minimum|smallest)\b", lowered):
        intent = QueryIntent.COMPARISON
        operator = QueryOperator.ARGMIN
        params = {}
    elif transition := re.search(r"\b(?P<relation>before|after)\b(?P<anchor>.+)", lowered):
        intent = QueryIntent.TEMPORAL
        operator = (
            QueryOperator.BEFORE
            if transition.group("relation") == "before"
            else QueryOperator.AFTER
        )
        params = {"anchor_concepts": _concepts(transition.group("anchor"))}
    elif re.search(r"\b(?:latest|current|currently|now|most recent)\b", lowered):
        intent = QueryIntent.STATE
        operator = QueryOperator.LATEST
        params = {}
    elif re.search(r"\b(?:first|earliest)\b", lowered):
        intent = QueryIntent.TEMPORAL
        operator = QueryOperator.FIRST
        params = {}
    elif legacy.profile:
        intent = QueryIntent.PREFERENCE
        operator = QueryOperator.LATEST
        params = {"prefer_explicit": True}
    elif re.search(r"\b(?:do not know|unknown|not enough|any evidence)\b", lowered):
        intent = QueryIntent.ABSTENTION_CHECK
        operator = QueryOperator.VERIFY
        params = {}
    else:
        intent = QueryIntent.DIRECT if concepts else QueryIntent.AMBIGUOUS
        operator = QueryOperator.VERIFY
        params = {}

    if operator not in {QueryOperator.VERIFY}:
        if operator in {
            QueryOperator.LATEST,
            QueryOperator.FIRST,
            QueryOperator.CHRONOLOGICAL_ORDER,
            QueryOperator.ARGMAX,
            QueryOperator.ARGMIN,
            QueryOperator.BEFORE,
            QueryOperator.AFTER,
        }:
            steps.append(QueryStep(operator=QueryOperator.SORT))
        steps.append(QueryStep(operator=operator, params=params))
    steps.append(QueryStep(operator=QueryOperator.VERIFY))

    return QueryPlan(
        normalized_query=normalized,
        body=body,
        as_of=effective_as_of,
        target_date=legacy.target_date,
        intent=intent,
        subgoals=_subgoals(body, concepts, subgoal_min_candidates),
        steps=steps,
        answer_type=answer_type,
        planner_confidence=0.95 if concepts else 0.35,
    )


__all__ = [
    "QueryIntent",
    "QueryOperator",
    "QueryPlan",
    "QueryStep",
    "QuerySubgoal",
    "compile_query_plan",
]
