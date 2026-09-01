"""Typed query execution over verified structured facts."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from ..index.models import FactRelation, FactStatus, StructuredFact
from .coverage import normalized_concepts
from .planner import QueryOperator, QueryPlan


class ExecutionTrace(BaseModel):
    operator: QueryOperator
    input_fact_ids: list[str] = Field(default_factory=list)
    output: object | None = None
    reason: str = ""


class ExecutionResult(BaseModel):
    value: object | None = None
    answer_type: str = "text"
    unit: str = ""
    fact_ids: list[str] = Field(default_factory=list)
    traces: list[ExecutionTrace] = Field(default_factory=list)
    complete: bool = False

    def render(self) -> str | None:
        if not self.complete or self.value is None:
            return None
        if isinstance(self.value, list):
            return "\n".join(str(item) for item in self.value)
        value = f"{self.value:g}" if isinstance(self.value, float) else str(self.value)
        return f"{value} {self.unit}".strip()


def _canonical_ids(
    facts: list[StructuredFact],
    relations: list[FactRelation],
    *,
    relation_confidence_threshold: float,
) -> dict[str, str]:
    parent = {fact.fact_id: fact.fact_id for fact in facts}

    def find(value: str) -> str:
        while parent.get(value, value) != value:
            parent[value] = parent.get(parent[value], parent[value])
            value = parent[value]
        return value

    for relation in relations:
        if (
            relation.kind != "same_event"
            or relation.confidence < relation_confidence_threshold
        ):
            continue
        if relation.source_fact_id in parent and relation.target_fact_id in parent:
            left = find(relation.source_fact_id)
            right = find(relation.target_fact_id)
            if left != right:
                parent[left] = right
    return {fact_id: find(fact_id) for fact_id in parent}


def _date(fact: StructuredFact) -> date | None:
    temporal = fact.event_time or fact.observed_at
    if temporal is None:
        return None
    return temporal.start.date() if isinstance(temporal.start, datetime) else temporal.start


def _eligible(fact: StructuredFact) -> bool:
    return fact.status not in {
        FactStatus.PLANNED,
        FactStatus.RECOMMENDED,
        FactStatus.HYPOTHETICAL,
        FactStatus.CANCELLED,
        FactStatus.FAILED,
    }


def execute_query_plan(
    plan: QueryPlan,
    facts: list[StructuredFact],
    relations: list[FactRelation],
    *,
    relation_confidence_threshold: float = 0.85,
) -> ExecutionResult:
    result = ExecutionResult(answer_type=plan.answer_type)
    canonical = _canonical_ids(
        facts,
        relations,
        relation_confidence_threshold=relation_confidence_threshold,
    )
    unique: list[StructuredFact] = []
    seen: set[str] = set()
    for fact in facts:
        identity = canonical.get(fact.fact_id, fact.fact_id)
        if identity not in seen:
            unique.append(fact)
            seen.add(identity)
    result.traces.append(
        ExecutionTrace(
            operator=QueryOperator.DEDUP,
            input_fact_ids=[fact.fact_id for fact in facts],
            output=[fact.fact_id for fact in unique],
        )
    )
    operation = next(
        (
            step
            for step in plan.steps
            if step.operator
            in {
                QueryOperator.COUNT,
                QueryOperator.SUM,
                QueryOperator.DIFFERENCE,
                QueryOperator.LATEST,
                QueryOperator.FIRST,
                QueryOperator.ELAPSED,
                QueryOperator.CHRONOLOGICAL_ORDER,
                QueryOperator.ARGMAX,
                QueryOperator.ARGMIN,
                QueryOperator.BEFORE,
                QueryOperator.AFTER,
            }
        ),
        None,
    )
    if operation is None:
        return result

    requested_statuses = {
        status for subgoal in plan.subgoals for status in subgoal.statuses
    }
    eligible = (
        [fact for fact in unique if fact.status.value in requested_statuses]
        if requested_statuses
        else [fact for fact in unique if _eligible(fact)]
    )
    fact_ids = [fact.fact_id for fact in eligible]
    if operation.operator == QueryOperator.COUNT:
        numeric = [
            fact.object.number
            for fact in eligible
            if fact.object.kind == "number" and fact.object.number is not None
        ]
        if len(eligible) == 1 and numeric:
            value: object = numeric[0]
        elif eligible:
            value = len(eligible)
        else:
            return result
    elif operation.operator in {QueryOperator.SUM, QueryOperator.DIFFERENCE}:
        values = [
            fact.object
            for fact in eligible
            if fact.object.number is not None
        ]
        if len(values) < 2:
            return result
        units = {(item.kind, item.currency, item.unit) for item in values}
        if len(units) != 1:
            return result
        numbers = [item.number for item in values if item.number is not None]
        value = (
            sum(numbers)
            if operation.operator == QueryOperator.SUM
            else max(numbers) - min(numbers)
        )
        kind, currency, unit = next(iter(units))
        result.unit = currency or unit
        if kind == "money" and not result.unit:
            return result
    elif operation.operator == QueryOperator.ELAPSED:
        dated = [(item, _date(item)) for item in eligible]
        dated = [(item, value) for item, value in dated if value is not None]
        as_of = plan.as_of.date() if isinstance(plan.as_of, datetime) else plan.as_of
        if len(dated) >= 2:
            endpoints = sorted(value for _item, value in dated)
            first, second = endpoints[0], endpoints[-1]
        elif len(dated) == 1 and as_of is not None:
            first, second = sorted((dated[0][1], as_of))
        else:
            return result
        days = (second - first).days
        unit = str(operation.params.get("unit", "days"))
        if unit == "weeks":
            value = days // 7
            result.unit = "weeks"
        elif unit == "months":
            value = (second.year - first.year) * 12 + second.month - first.month
            if second.day < first.day:
                value -= 1
            result.unit = "months"
        elif unit == "years":
            value = second.year - first.year - ((second.month, second.day) < (first.month, first.day))
            result.unit = "years"
        else:
            value = days
            result.unit = "days"
    elif operation.operator in {QueryOperator.LATEST, QueryOperator.FIRST}:
        ordered = sorted(
            eligible,
            key=lambda item: (_date(item) or date.min, item.sequence or 0, item.fact_id),
        )
        if not ordered:
            return result
        selected = ordered[-1] if operation.operator == QueryOperator.LATEST else ordered[0]
        value = selected.text
        fact_ids = [selected.fact_id]
    elif operation.operator == QueryOperator.CHRONOLOGICAL_ORDER:
        ordered = sorted(
            (item for item in eligible if _date(item) is not None),
            key=lambda item: (_date(item), item.sequence or 0, item.fact_id),
        )
        if len(ordered) < 2:
            return result
        value = [item.text for item in ordered]
        fact_ids = [item.fact_id for item in ordered]
    elif operation.operator in {QueryOperator.BEFORE, QueryOperator.AFTER}:
        anchor_terms = normalized_concepts(
            " ".join(str(item) for item in operation.params.get("anchor_concepts", []))
        )
        dated = [item for item in eligible if _date(item) is not None]
        anchor_candidates = [
            item
            for item in dated
            if anchor_terms & normalized_concepts(item.search_text)
        ]
        if not anchor_candidates:
            return result
        anchor = max(
            anchor_candidates,
            key=lambda item: len(anchor_terms & normalized_concepts(item.search_text)),
        )
        anchor_date = _date(anchor)
        assert anchor_date is not None
        candidates = [
            item
            for item in dated
            if item.fact_id != anchor.fact_id
            and (
                _date(item) < anchor_date
                if operation.operator == QueryOperator.BEFORE
                else _date(item) > anchor_date
            )
        ]
        if not candidates:
            return result
        selected = (
            max(candidates, key=lambda item: (_date(item), item.sequence or 0))
            if operation.operator == QueryOperator.BEFORE
            else min(candidates, key=lambda item: (_date(item), item.sequence or 0))
        )
        value = selected.text
        fact_ids = [selected.fact_id, anchor.fact_id]
    else:
        numbered = [item for item in eligible if item.object.number is not None]
        if not numbered:
            return result
        selected = (max if operation.operator == QueryOperator.ARGMAX else min)(
            numbered, key=lambda item: item.object.number or 0
        )
        value = selected.text
        fact_ids = [selected.fact_id]

    result.value = value
    result.fact_ids = fact_ids
    result.complete = True
    result.traces.append(
        ExecutionTrace(
            operator=operation.operator,
            input_fact_ids=fact_ids,
            output=value,
        )
    )
    return result


__all__ = ["ExecutionResult", "ExecutionTrace", "execute_query_plan"]
