"""Structured fact projection and multi-signal recall for long-term memory.

The Markdown leaf remains the source of truth.  This module projects its atomic
bullets into database-like rows at query time, so retrieval can combine lexical,
entity, temporal, provenance, and recency signals without introducing another
storage system or weakening the scoped object-store boundary.
"""

from __future__ import annotations

import math
import hashlib
import re
from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from ..models import MemoryDocument
from ..utils.codec import strip_yaml_front_matter
from .ranking import bm25_rank, tokenize


_TAG_RE = re.compile(r"^[-*+]\s+<(?P<metadata>[^>]*\bseq\s*=\s*[^>]*)>\s*(?P<text>.+)$")
_QUESTION_DATE_RE = re.compile(
    r"\bquestion date:\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b",
    re.IGNORECASE,
)
_GENERIC_TERMS = {
    "about",
    "after",
    "again",
    "ago",
    "and",
    "before",
    "between",
    "could",
    "current",
    "currently",
    "days",
    "does",
    "from",
    "have",
    "how",
    "into",
    "many",
    "months",
    "most",
    "much",
    "question",
    "should",
    "since",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "weeks",
    "years",
}
_TEMPORAL_RE = re.compile(
    r"\b(?:ago|before|after|between|date|day|week|month|year|when|"
    r"first|last|latest|earliest|previous|current|now|passed|since)\b",
    re.IGNORECASE,
)
_AGGREGATE_RE = re.compile(
    r"\b(?:how many|how much|total|altogether|average|most|least|"
    r"higher|lower|maximum|minimum|different|compare)\b",
    re.IGNORECASE,
)
_PROFILE_RE = re.compile(
    r"\b(?:recommend|suggest|preference|prefer|interest|like|dislike|"
    r"personaliz|should i|would suit)\b",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"\b(?:current|currently|now|latest|previous|previously|used to|"
    r"changed|switched|updated|before)\b",
    re.IGNORECASE,
)
_ASSISTANT_RE = re.compile(
    r"\b(?:assistant|you (?:said|suggested|recommended|created|wrote|named))\b",
    re.IGNORECASE,
)
_RELATIVE_AGO_RE = re.compile(
    r"\b(?P<count>a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_RELATIVE_WEEKDAY_RE = re.compile(
    r"\b(?:last|previous|this\s+past)\s+"
    r"(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_WEEKDAY_NUMBERS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True, slots=True)
class FactRecord:
    """One immutable atomic row projected from a Markdown memory bullet."""

    fact_id: str
    document_id: str
    document_index: int
    line_number: int
    heading: str
    text: str
    markdown: str
    sequence: int | None
    event_date: date | None
    observed_date: date | None
    source: str
    origin_id: str

    @property
    def search_text(self) -> str:
        return " ".join(part for part in (self.heading, self.text) if part)


@dataclass(frozen=True, slots=True)
class FactQueryPlan:
    """Deterministic query plan used before optional LLM reasoning."""

    body: str
    question_date: date | None
    target_date: date | None
    concepts: tuple[str, ...]
    temporal: bool
    aggregate: bool
    profile: bool
    update: bool
    assistant_source: bool


def _metadata_fields(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_field in value.split(","):
        name, separator, field_value = raw_field.partition("=")
        if separator and name.strip():
            fields[name.strip().casefold()] = field_value.strip()
    return fields


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _relative_target_date(body: str, question_date: date | None) -> date | None:
    """Resolve an explicit ``N units ago`` phrase to its calendar target.

    Month and year offsets use calendar arithmetic rather than fixed-day
    approximations so ordinary memory queries retain calendar semantics.
    """

    if question_date is None:
        return None
    if match := _RELATIVE_AGO_RE.search(body):
        raw_count = match.group("count").casefold()
        try:
            count = int(raw_count)
        except ValueError:
            count = _NUMBER_WORDS[raw_count]
        unit = match.group("unit").casefold()
        if unit == "day":
            return question_date - timedelta(days=count)
        if unit == "week":
            return question_date - timedelta(weeks=count)
        if unit == "year":
            target_year = question_date.year - count
            target_day = min(
                question_date.day,
                monthrange(target_year, question_date.month)[1],
            )
            return date(target_year, question_date.month, target_day)
        month_index = question_date.year * 12 + question_date.month - 1 - count
        target_year, zero_based_month = divmod(month_index, 12)
        target_month = zero_based_month + 1
        target_day = min(question_date.day, monthrange(target_year, target_month)[1])
        return date(target_year, target_month, target_day)
    if weekday_match := _RELATIVE_WEEKDAY_RE.search(body):
        target_weekday = _WEEKDAY_NUMBERS[weekday_match.group("weekday").casefold()]
        days_back = (question_date.weekday() - target_weekday) % 7 or 7
        return question_date - timedelta(days=days_back)
    return None


def parse_fact_records(documents: list[MemoryDocument]) -> list[FactRecord]:
    """Project all atomic bullets while retaining exact source coordinates."""

    records: list[FactRecord] = []
    for document_index, document in enumerate(documents):
        heading = ""
        for line_number, raw_line in enumerate(
            strip_yaml_front_matter(document.content).splitlines(), start=1
        ):
            line = raw_line.strip()
            if line.startswith("# "):
                heading = line[2:].strip()
                continue
            match = _TAG_RE.match(line)
            if not match:
                continue
            fields = _metadata_fields(match.group("metadata"))
            try:
                sequence = int(fields["seq"])
            except (KeyError, ValueError):
                sequence = None
            records.append(
                FactRecord(
                    fact_id=fields.get("fid")
                    or "legacy_f_"
                    + hashlib.sha256(
                        f"{document.metadata.id}\0{line_number}\0{match.group('text').strip()}".encode()
                    ).hexdigest()[:24],
                    document_id=document.metadata.id,
                    document_index=document_index,
                    line_number=line_number,
                    heading=heading,
                    text=match.group("text").strip(),
                    markdown=line,
                    sequence=sequence,
                    event_date=_parse_date(fields.get("time")),
                    observed_date=_parse_date(fields.get("observed")),
                    source=fields.get("source", "user").casefold(),
                    origin_id=fields.get("origin", ""),
                )
            )
    return records


def build_fact_query_plan(query: str) -> FactQueryPlan:
    """Compile the user question into stable retrieval constraints."""

    normalized = " ".join(query.split())
    question_date = None
    if match := _QUESTION_DATE_RE.search(normalized):
        try:
            question_date = date(*(int(part) for part in match.groups()))
        except ValueError:
            question_date = None
    body = normalized.rsplit("Question:", 1)[-1].strip()
    counts = Counter(
        token
        for token in tokenize(body)
        if len(token) >= 4 and token not in _GENERIC_TERMS
    )
    concepts = tuple(sorted(counts, key=lambda token: (-len(token), token)))
    return FactQueryPlan(
        body=body,
        question_date=question_date,
        target_date=_relative_target_date(body, question_date),
        concepts=concepts,
        temporal=bool(_TEMPORAL_RE.search(body)),
        aggregate=bool(_AGGREGATE_RE.search(body)),
        profile=bool(_PROFILE_RE.search(body)),
        update=bool(_UPDATE_RE.search(body)),
        assistant_source=bool(_ASSISTANT_RE.search(body)),
    )


def rank_fact_documents(
    query: str,
    documents: list[MemoryDocument],
    limit: int,
    *,
    expanded_query: str | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[MemoryDocument]:
    """Fuse fact BM25, exact entities, time, provenance, and state recency.

    Results are complete source leaves rather than detached snippets.  That
    preserves the atomic-memory/high-recall benefit while rehydrating the local
    context needed for exact values and multi-record reasoning.
    """

    if not documents or limit <= 0:
        return []
    records = parse_fact_records(documents)
    if not records:
        return []
    plan = build_fact_query_plan(query)
    search_query = expanded_query or plan.body
    ranks = bm25_rank(
        search_query,
        [record.search_text for record in records],
        k1=k1,
        b=b,
    )
    bm25_scores = {index: score for index, score in ranks}
    if not bm25_scores:
        return []

    concept_document_frequency = {
        concept: sum(
            concept in set(tokenize(record.search_text)) for record in records
        )
        for concept in plan.concepts
    }
    sequences = [record.sequence for record in records if record.sequence is not None]
    minimum_sequence = min(sequences, default=0)
    maximum_sequence = max(sequences, default=0)
    scores_by_document: dict[int, list[float]] = defaultdict(list)
    concepts_by_document: dict[int, set[str]] = defaultdict(set)
    for record_index, lexical_score in bm25_scores.items():
        record = records[record_index]
        record_tokens = set(tokenize(record.search_text))
        matched_concepts = {
            concept for concept in plan.concepts if concept in record_tokens
        }
        entity_score = sum(
            math.log1p(len(records) / max(1, concept_document_frequency[concept]))
            for concept in matched_concepts
        )
        temporal_score = 0.0
        if plan.temporal:
            temporal_score += 1.5 if record.event_date else 0.0
            temporal_score += 0.5 if record.observed_date else 0.0
            if plan.question_date is not None and record.event_date is not None:
                age_days = (plan.question_date - record.event_date).days
                if age_days >= 0:
                    temporal_score += 0.75 / (1.0 + age_days / 30.0)
            if plan.target_date is not None:
                record_dates = {
                    candidate
                    for candidate in (record.event_date, record.observed_date)
                    if candidate is not None
                }
                if record_dates:
                    distance_days = min(
                        abs((candidate - plan.target_date).days)
                        for candidate in record_dates
                    )
                    # Exact/near-exact relative-time matches should outrank a
                    # lexically similar event from the wrong month. Keep a
                    # smooth weekly decay for conversationally approximate dates.
                    temporal_score += 6.0 / (1.0 + distance_days / 7.0)
        provenance_score = 0.0
        if plan.assistant_source:
            provenance_score = 1.5 if record.source == "ai" else 0.0
        elif record.source != "ai":
            provenance_score = 0.25
        profile_score = 0.0
        if plan.profile and re.search(
            r"\b(?:prefer|like|love|enjoy|interest|goal|need|want|expert|beginner)\b",
            record.text,
            re.IGNORECASE,
        ):
            profile_score = 1.25
        recency_score = 0.0
        if plan.update and record.sequence is not None:
            sequence_span = maximum_sequence - minimum_sequence
            recency_score = (
                (record.sequence - minimum_sequence) / sequence_span
                if sequence_span
                else 0.5
            )
        total = (
            lexical_score
            + entity_score
            + temporal_score
            + provenance_score
            + profile_score
            + recency_score
        )
        scores_by_document[record.document_index].append(total)
        concepts_by_document[record.document_index].update(matched_concepts)

    document_scores = {
        document_index: max(scores) + 0.2 * sum(sorted(scores, reverse=True)[1:4])
        for document_index, scores in scores_by_document.items()
    }
    ranked_documents = sorted(
        document_scores,
        key=lambda index: (-document_scores[index], index),
    )

    # Rare, independently requested entities get one reserved source before
    # global score filling. This avoids a frequently discussed entity crowding
    # a complementary endpoint or aggregation member out of the result set.
    selected: list[int] = []
    rare_concepts = sorted(
        plan.concepts,
        key=lambda concept: (concept_document_frequency.get(concept, 0), concept),
    )[: min(6, limit)]
    for concept in rare_concepts:
        candidates = [
            index
            for index in ranked_documents
            if concept in concepts_by_document[index]
        ]
        if candidates and candidates[0] not in selected:
            selected.append(candidates[0])
        if len(selected) >= limit:
            break
    for document_index in ranked_documents:
        if len(selected) >= limit:
            break
        if document_index not in selected:
            selected.append(document_index)
    return [documents[index] for index in selected]


__all__ = [
    "FactQueryPlan",
    "FactRecord",
    "build_fact_query_plan",
    "parse_fact_records",
    "rank_fact_documents",
]
