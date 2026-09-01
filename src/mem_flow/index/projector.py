"""Deterministic Markdown-to-fact projection and conservative relation building."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import date

from ..models import MemoryDocument
from ..utils.codec import strip_yaml_front_matter
from .models import (
    EntityAlias,
    EntityRecord,
    EvidenceSpan,
    FactIndexSnapshot,
    FactModality,
    FactRelation,
    FactStatus,
    FactValue,
    StructuredFact,
    TemporalPrecision,
    TemporalValue,
)


_FACT_LINE_RE = re.compile(
    r"^(?P<prefix>\s*[-*+]\s+)<(?P<metadata>[^>]*\bseq\s*=\s*[^>]*)>\s*(?P<text>.+)$"
)
_DATE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_MONEY_RE = re.compile(r"(?P<symbol>[$€£])\s*(?P<number>\d[\d,]*(?:\.\d+)?)")
_DURATION_RE = re.compile(
    r"\b(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9'&.-]*)(?:\s+[A-Z][A-Za-z0-9'&.-]*)*\b")
_CURRENCIES = {"$": "USD", "€": "EUR", "£": "GBP"}
_SOURCE_ENVELOPE_RE = re.compile(
    r"^(?:The\s+(?:user|assistant)\s+said:\s*)?"
    r"Session ID:\s*\S+\s+Session date:\s*"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"(?:\s+\([^)]+\))?(?:\s+\d{1,2}:\d{2})?\s*",
    re.IGNORECASE,
)


def metadata_fields(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_field in value.split(","):
        name, separator, field_value = raw_field.partition("=")
        if separator and name.strip():
            fields[name.strip().casefold()] = field_value.strip()
    return fields


def assign_fact_ids(content: str, *, fallback_origin: str = "") -> str:
    """Insert code-owned stable fact ids without changing semantic fact text."""

    output: list[str] = []
    fact_index = 0
    fence: str | None = None
    for line in content.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            character = marker.group(1)[0]
            fence = character if fence is None else None if fence == character else fence
            output.append(line)
            continue
        match = None if fence else _FACT_LINE_RE.match(line)
        if not match:
            output.append(line)
            continue
        fact_index += 1
        raw_fields = [item.strip() for item in match.group("metadata").split(",")]
        parsed = metadata_fields(match.group("metadata"))
        if parsed.get("fid"):
            output.append(line)
            continue
        origin = parsed.get("origin", fallback_origin)
        sequence = parsed.get("seq", "")
        normalized_text = " ".join(match.group("text").casefold().split())
        digest = hashlib.sha256(
            f"{origin}\0{sequence}\0{fact_index}\0{normalized_text}".encode("utf-8")
        ).hexdigest()[:24]
        insert_at = next(
            (
                index + 1
                for index, field in enumerate(raw_fields)
                if field.partition("=")[0].strip().casefold() == "origin"
            ),
            1,
        )
        raw_fields.insert(insert_at, f"fid=f_{digest}")
        output.append(
            f"{match.group('prefix')}<{','.join(raw_fields)}> {match.group('text')}"
        )
    return "\n".join(output)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_date(value: str | None, *, anchor: str) -> TemporalValue | None:
    if not value or not (match := _DATE_RE.match(value.strip())):
        return None
    try:
        parsed = date(*(int(part) for part in match.groups()))
    except ValueError:
        return None
    return TemporalValue(
        start=parsed,
        end=parsed,
        precision=TemporalPrecision.DAY,
        raw_expression=value,
        anchor=anchor,
    )


def _status(text: str, source: str) -> FactStatus:
    lowered = text.casefold()
    if re.search(r"\b(?:cancelled|canceled|called off|no longer plans?)\b", lowered):
        return FactStatus.CANCELLED
    if re.search(r"\b(?:failed|could not|couldn't|was unable)\b", lowered):
        return FactStatus.FAILED
    if re.search(r"\b(?:recommend(?:ed)?|suggest(?:ed)?|should|could try)\b", lowered):
        return FactStatus.RECOMMENDED
    if re.search(r"\b(?:plan(?:s|ned)? to|will|going to|intend(?:s|ed)? to|booked)\b", lowered):
        return FactStatus.PLANNED
    if re.search(r"\b(?:currently|in progress|working on|still doing)\b", lowered):
        return FactStatus.IN_PROGRESS
    if re.search(
        r"\b(?:completed|finished|attended|visited|bought|purchased|paid|made|"
        r"started|joined|returned|received|won|went|did|has|had|is|was)\b",
        lowered,
    ):
        return FactStatus.COMPLETED
    if re.search(r"\b(?:might|maybe|if |would like|considering)\b", lowered):
        return FactStatus.HYPOTHETICAL
    return FactStatus.RECOMMENDED if source in {"ai", "assistant"} else FactStatus.UNKNOWN


def _predicate(text: str, status: FactStatus) -> str:
    lowered = text.casefold()
    patterns = (
        (r"\b(?:dislikes?|hates?|avoids?|allergic)\b", "avoids"),
        (r"\b(?:prefers?|likes?|loves?|enjoys?)\b", "prefers"),
        (r"\b(?:cancelled|canceled|called off)\b", "cancelled"),
        (r"\b(?:recommend(?:ed)?|suggest(?:ed)?)\b", "recommends"),
        (r"\b(?:bought|purchased|acquired|ordered)\b", "purchased"),
        (r"\b(?:paid|spent|cost)\b", "spent"),
        (r"\b(?:attended|visited|went to|participated)\b", "attended"),
        (r"\b(?:completed|finished)\b", "completed"),
        (r"\b(?:started|began|joined)\b", "started"),
        (r"\b(?:works?|worked|employed)\b", "works_at"),
        (r"\b(?:lives?|moved|stays?)\b", "located_at"),
        (r"\b(?:has|owns?|had)\b", "has"),
    )
    for pattern, predicate in patterns:
        if re.search(pattern, lowered):
            return predicate
    if status == FactStatus.PLANNED:
        return "plans"
    return "mentions"


def _fact_value(text: str) -> tuple[FactValue, dict[str, FactValue]]:
    qualifiers: dict[str, FactValue] = {}
    if match := _MONEY_RE.search(text):
        number = float(match.group("number").replace(",", ""))
        value = FactValue(
            kind="money",
            raw=match.group(0),
            normalized=f"{number:g} {_CURRENCIES[match.group('symbol')]}",
            number=number,
            currency=_CURRENCIES[match.group("symbol")],
        )
        qualifiers["amount"] = value
    elif match := _DURATION_RE.search(text):
        number = float(match.group("number"))
        unit = match.group("unit").casefold().rstrip("s")
        value = FactValue(
            kind="duration",
            raw=match.group(0),
            normalized=f"{number:g} {unit}",
            number=number,
            unit=unit,
        )
        qualifiers["duration"] = value
    elif match := _NUMBER_RE.search(text):
        number = float(match.group(0))
        value = FactValue(
            kind="number",
            raw=match.group(0),
            normalized=f"{number:g}",
            number=number,
        )
        qualifiers["quantity"] = value
    else:
        value = FactValue(kind="text", raw=text, normalized=" ".join(text.split()))
    return value, qualifiers


def _semantic_fact_text(text: str) -> str:
    """Remove evaluator/source coordinates before semantic projection.

    Lossless memories intentionally retain a human-readable session envelope.
    Its date is provenance, not the fact's quantity, predicate, or status.
    Temporal projection already reads the trusted ``observed`` metadata.
    """

    stripped = _SOURCE_ENVELOPE_RE.sub("", text).strip()
    return stripped or text


def project_document(document: MemoryDocument) -> list[StructuredFact]:
    body = strip_yaml_front_matter(document.content)
    digest = _digest(body)
    facts: list[StructuredFact] = []
    heading = ""
    for line_number, raw_line in enumerate(body.splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith("# "):
            heading = line[2:].strip()
            continue
        match = _FACT_LINE_RE.match(line)
        if not match:
            continue
        fields = metadata_fields(match.group("metadata"))
        text = match.group("text").strip()
        source = fields.get("source", "user").casefold()
        origin = fields.get("origin", "")
        fact_id = fields.get("fid") or "legacy_f_" + _digest(
            f"{document.metadata.id}\0{line_number}\0{text}"
        )[:24]
        try:
            sequence = int(fields["seq"])
        except (KeyError, ValueError):
            sequence = None
        semantic_text = _semantic_fact_text(text)
        status = _status(semantic_text, source)
        value, qualifiers = _fact_value(semantic_text)
        evidence = (
            [EvidenceSpan(evidence_id=origin, role="assistant" if source in {"ai", "assistant"} else "user")]
            if origin
            else []
        )
        facts.append(
            StructuredFact(
                fact_id=fact_id,
                subject_entity_id="assistant" if source in {"ai", "assistant"} else "user",
                predicate=_predicate(semantic_text, status),
                object=value,
                qualifiers=qualifiers,
                status=status,
                modality=(
                    FactModality.ASSISTANT_SUGGESTION
                    if source in {"ai", "assistant"} and status == FactStatus.RECOMMENDED
                    else FactModality.EXPLICIT
                ),
                event_time=_parse_date(fields.get("time"), anchor="event"),
                observed_at=_parse_date(fields.get("observed"), anchor="observed"),
                source_role=source,
                evidence=evidence,
                source_document_id=document.metadata.id,
                source_document_key=document.key,
                source_document_digest=digest,
                source_kind=document.metadata.kind,
                line_number=line_number,
                heading=heading,
                text=text,
                markdown=line,
                sequence=sequence,
            )
        )
    return facts


def _event_signature(fact: StructuredFact) -> str:
    text = re.sub(
        r"\b(?:the|a|an|user|assistant|currently|previously|now|just)\b",
        " ",
        fact.text.casefold(),
    )
    text = re.sub(r"\s+", " ", text).strip()
    # Observation time is not event identity: a later session may retell the
    # same event. Only an explicit event time participates in the signature.
    date_value = fact.event_time
    date_key = str(date_value.start) if date_value else ""
    return f"{fact.subject_entity_id}|{fact.predicate}|{date_key}|{text}"


def build_relations(facts: list[StructuredFact]) -> list[FactRelation]:
    by_signature: dict[str, list[StructuredFact]] = defaultdict(list)
    for fact in facts:
        by_signature[_event_signature(fact)].append(fact)
    relations: list[FactRelation] = []
    for matches in by_signature.values():
        if len(matches) < 2:
            continue
        retained = min(matches, key=lambda item: (item.sequence or 0, item.fact_id))
        for duplicate in matches:
            if duplicate.fact_id == retained.fact_id:
                continue
            relation_id = "rel_" + _digest(
                f"same_event\0{duplicate.fact_id}\0{retained.fact_id}"
            )[:24]
            relations.append(
                FactRelation(
                    relation_id=relation_id,
                    source_fact_id=duplicate.fact_id,
                    target_fact_id=retained.fact_id,
                    kind="same_event",
                    confidence=1.0,
                    evidence_fact_ids=[duplicate.fact_id, retained.fact_id],
                )
            )
    update_marker = re.compile(
        r"\b(?:now|currently|no longer|changed|switched|instead|cancelled|canceled)\b",
        re.IGNORECASE,
    )
    relation_pairs = {
        (item.source_fact_id, item.target_fact_id, item.kind) for item in relations
    }
    ordered = sorted(facts, key=lambda item: (item.sequence or 0, item.fact_id))
    opposite_preferences = {
        ("prefers", "avoids"),
        ("avoids", "prefers"),
    }

    def relation_terms(fact: StructuredFact) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", fact.text.casefold())
            if len(token) >= 4
            and token
            not in {
                "assistant",
                "currently",
                "instead",
                "likes",
                "prefers",
                "avoids",
                "user",
            }
        }

    for index, newer in enumerate(ordered):
        if not update_marker.search(newer.text):
            continue
        newer_terms = relation_terms(newer)
        for older in reversed(ordered[:index]):
            if (
                older.subject_entity_id != newer.subject_entity_id
                or (
                    older.predicate != newer.predicate
                    and (older.predicate, newer.predicate)
                    not in opposite_preferences
                )
            ):
                continue
            older_terms = relation_terms(older)
            if not newer_terms & older_terms:
                continue
            pair = (newer.fact_id, older.fact_id, "supersedes")
            if pair not in relation_pairs:
                relations.append(
                    FactRelation(
                        relation_id="rel_" + _digest("\0".join(pair))[:24],
                        source_fact_id=newer.fact_id,
                        target_fact_id=older.fact_id,
                        kind="supersedes",
                        confidence=0.9,
                        evidence_fact_ids=[newer.fact_id, older.fact_id],
                    )
                )
            break

    superseded = {
        (item.source_fact_id, item.target_fact_id)
        for item in relations
        if item.kind == "supersedes"
    }
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.subject_entity_id != right.subject_entity_id:
                continue
            conflict = False
            if (left.predicate, right.predicate) in opposite_preferences:
                conflict = bool(relation_terms(left) & relation_terms(right))
            elif (
                left.predicate == right.predicate
                and left.object.number is not None
                and right.object.number is not None
                and left.object.number != right.object.number
                and (left.object.kind, left.object.currency, left.object.unit)
                == (right.object.kind, right.object.currency, right.object.unit)
                and left.event_time is not None
                and right.event_time is not None
                and left.event_time.start == right.event_time.start
            ):
                conflict = True
            if not conflict or (
                (right.fact_id, left.fact_id) in superseded
                or (left.fact_id, right.fact_id) in superseded
            ):
                continue
            pair = (right.fact_id, left.fact_id, "contradicts")
            if pair in relation_pairs:
                continue
            relations.append(
                FactRelation(
                    relation_id="rel_" + _digest("\0".join(pair))[:24],
                    source_fact_id=right.fact_id,
                    target_fact_id=left.fact_id,
                    kind="contradicts",
                    confidence=0.9,
                    evidence_fact_ids=[right.fact_id, left.fact_id],
                )
            )
    return relations


def build_entities(facts: list[StructuredFact]) -> list[EntityRecord]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        for match in _ENTITY_RE.findall(fact.text):
            if match.casefold() in {"the", "user", "assistant", "session"}:
                continue
            aliases[match.casefold()].add(fact.fact_id)
    entities = [
        EntityRecord(
            entity_id="entity_" + _digest(name)[:20],
            canonical_name=name,
            aliases=[
                EntityAlias(
                    value=name,
                    normalized=name,
                    source_fact_ids=sorted(fact_ids),
                )
            ],
            evidence_fact_ids=sorted(fact_ids),
        )
        for name, fact_ids in aliases.items()
    ]
    return sorted(entities, key=lambda item: item.entity_id)


def project_documents(documents: list[MemoryDocument]) -> FactIndexSnapshot:
    maturity = {"memory": 5, "topic": 5, "rewrite": 4, "raw": 3, "current": 2}
    selected: dict[str, StructuredFact] = {}
    for document in documents:
        for fact in project_document(document):
            current = selected.get(fact.fact_id)
            if current is None or maturity.get(fact.source_kind, 0) > maturity.get(
                current.source_kind, 0
            ):
                selected[fact.fact_id] = fact
    facts = sorted(selected.values(), key=lambda item: (item.sequence or 0, item.fact_id))
    return FactIndexSnapshot(
        facts=facts,
        entities=build_entities(facts),
        relations=build_relations(facts),
    )


__all__ = [
    "assign_fact_ids",
    "build_entities",
    "build_relations",
    "metadata_fields",
    "project_document",
    "project_documents",
]
