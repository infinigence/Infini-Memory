"""Combined working-layer and hierarchical long-term memory retrieval."""

from __future__ import annotations

import json
import logging
import re
from calendar import monthrange
from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict

from ..config import RetrievalConfig
from ..index import FactIndexStore
from ..hierarchy import DirectoryTopicBuilder
from ..llm import FlowLLM
from ..models import (
    ChatMessage,
    DocumentMetadata,
    DocumentSelection,
    LLMRequest,
    MemoryDocument,
    SearchHit,
    SearchRequest,
    SearchResult,
    SearchScopeSelection,
    SearchSource,
    SearchStrategy,
)
from ..observability import FlowMetrics, observe_flow
from ..prompts import (
    AGENTIC_ANSWER_PROMPT,
    ANSWER_PROMPT,
    ANSWER_REVIEW_PROMPT,
    SEARCH_DOCUMENT_PROMPT,
    SEARCH_SCOPE_PROMPT,
)
from ..repository import MemoryRepository
from ..utils.codec import strip_yaml_front_matter
from ..utils.parsing import parse_json_model
from .agentic import (
    AgenticRetriever,
    _expand_focus_query,
    _needs_high_recall_merge,
    _needs_personalization_context,
)
from .fact_index import build_fact_query_plan
from .guidance import answer_task_guidance
from .ranking import bm25_documents, bm25_partitions, bm25_rank
from .executor import execute_query_plan
from .hybrid import hybrid_recall
from .planner import QueryIntent, compile_query_plan
from .verifier import verify_evidence


_WINDOW_COUNT_WORDS = {
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
    "couple": 2,
    "few": 3,
    "several": 4,
}


def _needs_answer_audit(query: str, draft: str) -> bool:
    """Use an independent evidence pass for questions prone to scope errors."""

    text = query.casefold()
    uncertain = draft.casefold()
    return bool(
        re.search(
            r"\b(?:how (?:many|much|long|old|early|late|often)|when|where|which|"
            r"what (?:did|was|were)|at which|recommend|suggest|tips?)\b|"
            r"\b(?:total|difference|percent(?:age)?|compared|comparison|order|"
            r"earlier|later|before|after|ago|last|past|this year|do you think|"
            r"should i)\b",
            text,
        )
        or re.search(
            r"\b(?:insufficient|not enough|cannot determine|can't determine|"
            r"does not contain|doesn't contain|not mentioned|not recorded)\b",
            uncertain,
        )
    )


def _json_answer(raw: str) -> str:
    """Extract an answer field from a compact JSON response when available."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return raw
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return raw
    answer = payload.get("answer") if isinstance(payload, dict) else None
    return answer.strip() if isinstance(answer, str) and answer.strip() else raw


def _is_insufficient_answer(answer: str) -> bool:
    return bool(
        re.search(
            r"\b(?:insufficient|not enough|cannot determine|can't determine|"
            r"does not contain|doesn't contain|do not have|don't have|"
            r"not mentioned|not recorded|no stored)\b",
            answer,
            re.I,
        )
    )


def _answer_review_context(
    query: str,
    documents: list[MemoryDocument],
    *,
    document_limit: int = 12,
    chars_per_document: int = 6000,
) -> str:
    """Keep bounded authoritative source turns for the independent audit pass."""

    evidence = [item for item in documents if item.metadata.kind == "evidence"]
    selected = evidence[:document_limit] or documents[:document_limit]
    blocks: list[str] = []
    for document in selected:
        body = AgenticRetriever._focus_document_for_answer(
            query,
            document,
            max_lines=64,
        ).content.strip()
        if not body:
            continue
        if len(body) > chars_per_document:
            body = body[:chars_per_document].rstrip() + "\n[truncated]"
        blocks.append(f"SOURCE {document.metadata.id}:\n{body}")
    return "\n\n".join(blocks) or "No additional source excerpt available."


def _primary_source_context(
    query: str,
    documents: list[MemoryDocument],
    *,
    document_limit: int | None = None,
    chars_per_document: int = 3500,
) -> str:
    """Preserve local turn adjacency from the best original conversations.

    The cross-document ledger deliberately ranks individual lines.  That is
    useful for broad recall but can separate a value from the preceding store,
    subject, list heading, or user request that gives it meaning.  A very small
    source-ordered companion view restores that relation without sending every
    selected conversation to the answer model.
    """

    if document_limit is None:
        # Direct lookups usually need one conversation plus a nearby fallback.
        # Aggregation, comparison, chronology, and state questions often join
        # several independently selected source conversations; keeping four
        # preserves those authoritative turns without copying every hit into
        # the answer prompt.
        document_limit = 4 if _needs_high_recall_merge(query) else 2
    evidence = [item for item in documents if item.metadata.kind == "evidence"]
    selected = evidence[:document_limit]
    blocks: list[str] = []
    for document in selected:
        body = AgenticRetriever._focus_document_for_answer(
            query,
            document,
            max_lines=24,
        ).content.strip()
        if not body:
            continue
        if len(body) > chars_per_document:
            body = body[:chars_per_document].rstrip() + "\n[truncated]"
        blocks.append(f"SOURCE {document.metadata.id}:\n{body}")
    return "\n\n".join(blocks) or "No original source excerpt available."


_MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
_EXPLICIT_WINDOW_RE = re.compile(
    r"\b(?:in|during|over|within)?\s*(?:the\s+)?(?:past|last|previous)\s+"
    r"(?:(?P<count>a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"couple|few|several|\d+)\s+(?:of\s+)?)?"
    r"(?P<unit>day|week|month|year)s?\b",
    re.I,
)


def _shift_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    return date(year, month, min(value.day, monthrange(year, month)[1]))


def _explicit_temporal_window(
    query_body: str, question_date: date
) -> tuple[date, date] | None:
    """Return inclusive bounds for an explicit relative time window."""

    if re.search(r"\b(?:last|past|previous)\s+weekend\b", query_body, re.I):
        days_since_saturday = (question_date.weekday() - 5) % 7 or 7
        saturday = question_date - timedelta(days=days_since_saturday)
        return saturday, saturday + timedelta(days=1)
    named_month = re.search(
        r"\b(?:in|during)\s+(?:the\s+month\s+of\s+)?"
        r"(?P<month>january|february|march|april|may|june|july|august|"
        r"september|october|november|december)"
        r"(?:\s+(?P<year>\d{4}))?\b",
        query_body,
        re.I,
    )
    if named_month is not None:
        month = _MONTH_NUMBERS[named_month.group("month").casefold()]
        raw_year = named_month.group("year")
        year = int(raw_year) if raw_year else question_date.year
        if raw_year is None and month > question_date.month:
            year -= 1
        return date(year, month, 1), date(year, month, monthrange(year, month)[1])
    since_period_start = re.search(
        r"\bsince\s+(?:the\s+)?(?:start|beginning)\s+of\s+"
        r"(?:the\s+)?(?P<unit>year|month)\b",
        query_body,
        re.I,
    )
    if since_period_start is not None:
        start = (
            date(question_date.year, 1, 1)
            if since_period_start.group("unit").casefold() == "year"
            else date(question_date.year, question_date.month, 1)
        )
        return start, question_date
    if re.search(r"\b(?:this|current)\s+year\b", query_body, re.I):
        return date(question_date.year, 1, 1), question_date
    match = _EXPLICIT_WINDOW_RE.search(query_body)
    if match is None:
        return None
    raw_count = (match.group("count") or "one").casefold()
    count = int(raw_count) if raw_count.isdigit() else _WINDOW_COUNT_WORDS[raw_count]
    unit = match.group("unit").casefold()
    if unit == "day":
        start = question_date - timedelta(days=count)
    elif unit == "week":
        start = question_date - timedelta(weeks=count)
    elif unit == "month":
        start = _shift_months(question_date, count)
    else:
        start = _shift_months(question_date, count * 12)
    return start, question_date


def _answer_evidence_ledger(
    query: str,
    documents: list[MemoryDocument],
    *,
    limit: int = 24,
) -> str:
    """Build a compact, cross-document fact ledger beside the answer instruction.

    Agentic retrieval already focuses every selected document, but aggregation
    evidence can still be hundreds of lines apart. Re-ranking fact lines across
    all selected documents keeps complementary values and events close to the
    model's answer position without discarding the full verification context.
    """

    candidates: list[str] = []
    candidate_lines: list[str] = []
    candidate_roles: list[str] = []
    expanded_query = _expand_focus_query(query)
    focus_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", expanded_query)
        if len(token) >= 4
    }
    query_plan = build_fact_query_plan(query)
    assistant_query = query_plan.assistant_source or _asks_about_assistant_memory(query)
    selected_evidence_ids = {
        document.metadata.id
        for document in documents
        if document.metadata.kind == "evidence"
    }
    indices_by_document: list[list[int]] = []
    seen: set[str] = set()
    for document in documents:
        heading = ""
        role = ""
        list_context = ""
        last_user_context = ""
        document_indices: list[int] = []
        for raw_line in strip_yaml_front_matter(document.content).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if document.metadata.kind != "evidence" and selected_evidence_ids:
                origin = re.search(r"(?:^|[<,\s])origin=(EVIDENCE_[^,>\s]+)", line)
                if origin is not None and origin.group(1) in selected_evidence_ids:
                    # Prefer the selected lossless source conversation over
                    # its derived memory copy. This provenance-level collapse
                    # prevents aggregates from seeing one event twice and
                    # frees ledger space for additional distinct sources.
                    continue
            if line.startswith("# "):
                heading = line[2:].strip()
                continue
            if line in {"[USER]", "[ASSISTANT]", "[SYSTEM]", "[TOOL]"}:
                role = line[1:-1].lower()
                list_context = last_user_context if role == "assistant" else ""
                continue
            normalized = re.sub(r"\s+", " ", line).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            location = f" [{heading}]" if heading else ""
            effective_role = role
            if re.search(r"(?:^|[<,\s])source=AI(?:[>,\s]|$)", line, re.I):
                effective_role = "assistant"
            fact_body = re.sub(r"^[-*+]\s+(?:<[^>]+>\s*)?", "", line).strip()
            if re.match(r"^(?:the\s+)?assistant(?:'s|\s)", fact_body, re.I):
                effective_role = "assistant"
            elif re.match(r"^(?:the\s+)?user(?:'s|\s)", fact_body, re.I):
                effective_role = "user"
            if (
                effective_role == "user"
                and not re.match(r"^Session (?:ID|date):", line, re.I)
                and len(line) <= 500
            ):
                last_user_context = line
            list_item = bool(
                re.match(r"^(?:[-*+]\s+|\d{1,3}(?:st|nd|rd|th)?[.):]\s+)", line, re.I)
            )
            if (
                effective_role == "assistant"
                and not list_item
                and not re.match(r"^Session (?:ID|date):", line, re.I)
                and len(line) <= 500
            ):
                list_context = line
            if (
                effective_role == "assistant"
                and document.metadata.kind != "evidence"
                and not assistant_query
            ):
                continue
            speaker = f" [{effective_role}]" if effective_role else ""
            document_indices.append(len(candidates))
            target_annotation = ""
            if query_plan.target_date is not None:
                date_match = re.search(
                    r"\b(?:time|observed)=(\d{4}[-/]\d{1,2}[-/]\d{1,2})(?:[T ,>])",
                    line,
                )
                if date_match:
                    try:
                        fact_date = date.fromisoformat(
                            date_match.group(1).replace("/", "-")
                        )
                    except ValueError:
                        fact_date = None
                    if fact_date is not None:
                        target_distance = abs((fact_date - query_plan.target_date).days)
                        target_annotation = (
                            f" [target_date={query_plan.target_date.isoformat()},"
                            f"distance_days={target_distance}]"
                        )
            compact_line = _compact_ledger_line(line, focus_terms=focus_terms)
            list_annotation = ""
            if list_item and effective_role == "assistant" and list_context:
                context = re.sub(r"\s+", " ", list_context).strip()
                list_annotation = f" [list_context={context[:300]}]"
            candidates.append(
                f"[{document.metadata.id}]{location}{speaker}{target_annotation}"
                f"{list_annotation} "
                f"{compact_line}"
            )
            candidate_lines.append(line)
            candidate_roles.append(effective_role)
            if effective_role == "assistant" and re.match(
                r"^\d{1,3}(?:st|nd|rd|th)?[.):].*:\s*$", line, re.I
            ):
                # A numbered line ending in ':' is a subsection header rather
                # than an ordinary list value. Bind following bullets to that
                # header so similarly named bullets in sibling sections remain
                # distinguishable (for example processes at three facilities).
                list_context = line
        indices_by_document.append(document_indices)
    ranks = bm25_rank(expanded_query, candidates)
    if not ranks:
        return "- No focused fact line matched; inspect MEMORY_CONTEXT directly."

    diverse: set[int] = set()
    if _needs_high_recall_merge(query) or _needs_personalization_context(query):
        # Aggregation and chronology questions need complementary events from
        # multiple sessions. Ordinary fact lookup instead uses the global rank
        # directly so an unrelated selected document cannot claim a slot.
        for document_indices in indices_by_document:
            document_ranks = bm25_rank(
                expanded_query,
                [candidates[index] for index in document_indices],
            )
            # Aggregate questions often need several values from the same
            # consolidated leaf (for example multiple completed activities).
            # Keep enough per-document candidates to avoid a long leaf's
            # topical sentences displacing its later numeric facts.
            per_document = 8 if _needs_high_recall_merge(query) else 2
            diverse.update(
                document_indices[local_index]
                for local_index, _score in document_ranks[:per_document]
            )
            if query_plan.aggregate:
                requested_units = {
                    unit
                    for unit in (
                        "hour",
                        "minute",
                        "day",
                        "week",
                        "month",
                        "year",
                        "dollar",
                        "percent",
                        "point",
                        "page",
                    )
                    if re.search(rf"\b{unit}s?\b", query, re.I)
                }
                for index in document_indices:
                    candidate = candidates[index]
                    if "[assistant]" in candidate.casefold():
                        continue
                    if not re.search(r"\b\d+(?:\.\d+)?\b", candidate):
                        continue
                    if (
                        not requested_units
                        or any(
                            re.search(rf"\b{unit}s?\b", candidate, re.I)
                            for unit in requested_units
                        )
                        or (
                            "percent" in requested_units
                            and re.search(r"\d+(?:\.\d+)?\s*%", candidate)
                        )
                    ):
                        diverse.add(index)
    ranked_indices = [index for index, _score in ranks]
    asks_latest_state = query_plan.update or bool(
        not query_plan.aggregate
        and re.search(
            r"\b(?:what|which|where|how)\b.{0,45}\b(?:do i|am i|is my|are my|have i)\b",
            query,
            re.I,
        )
    )
    if asks_latest_state and ranks:
        # BM25 identifies the semantic state first; sequence then resolves
        # competing versions of that same state.  Restrict the recency sort to
        # strongly matching candidates so an unrelated high-seq fact cannot
        # displace the requested attribute.
        top_score = ranks[0][1]
        relevant = [
            (index, score)
            for index, score in ranks[: max(24, limit)]
            if score >= top_score * 0.55
        ]

        def sequence(index: int) -> int:
            match = re.search(r"(?:^|[<,\s])seq\s*=\s*(\d+)", candidates[index], re.I)
            return int(match.group(1)) if match else -1

        relevant.sort(key=lambda item: (-sequence(item[0]), -item[1]))
        promoted = [index for index, _score in relevant]
        promoted_set = set(promoted)
        ranked_indices = promoted + [
            index for index in ranked_indices if index not in promoted_set
        ]
    if assistant_query:
        # Assistant-memory questions frequently point at an introductory user
        # request while the answer is several physical lines later (recipes,
        # numbered lists, or a subsection followed by bullets).  Pure line-level
        # BM25 keeps the request but can discard the answer line because it no
        # longer repeats the topic words.  Promote a small conversation-local
        # neighbourhood from the most relevant documents, and honor an explicit
        # ordinal independently of the item text.
        ordinal = _requested_list_ordinal(query)
        document_rankings: list[tuple[float, list[int], list[int]]] = []
        for document_indices in indices_by_document:
            if not document_indices:
                continue
            local_ranks = bm25_rank(
                expanded_query,
                [candidates[index] for index in document_indices],
            )
            if not local_ranks:
                continue
            document_rankings.append(
                (
                    local_ranks[0][1],
                    document_indices,
                    [document_indices[index] for index, _score in local_ranks],
                )
            )
        document_rankings.sort(key=lambda item: item[0], reverse=True)
        top_document_score = document_rankings[0][0] if document_rankings else 0.0
        relevant_documents = [
            item
            for item in document_rankings[:8]
            if item[0] >= top_document_score * 0.35
        ]
        ordinal_promoted: list[int] = []
        if ordinal is not None:
            # Exact list positions are rare and high-confidence. Search every
            # selected document so a semantically vague item ("Sound effects")
            # is not lost merely because its line has none of the list's topic
            # words.
            for _score, document_indices, _local_ranks in relevant_documents:
                ordinal_promoted.extend(
                    index
                    for index in document_indices
                    if candidate_roles[index] == "assistant"
                    and _line_starts_with_ordinal(candidate_lines[index], ordinal)
                )
        query_value_terms = {
            token
            for token in re.findall(r"[a-z0-9]+", query.casefold())
            if len(token) >= 4
            and token
            not in {
                "about",
                "again",
                "could",
                "from",
                "have",
                "looking",
                "previous",
                "remind",
                "that",
                "what",
                "when",
                "which",
                "with",
                "would",
                "your",
            }
        }
        value_promoted: list[int] = []
        for index, candidate in enumerate(candidates):
            if candidate_roles[index] != "assistant" or not re.search(r"\d", candidate):
                continue
            overlap = sum(term in candidate.casefold() for term in query_value_terms)
            if overlap >= min(2, max(1, len(query_value_terms))):
                value_promoted.append(index)
        assistant_context: list[int] = []
        for _score, document_indices, local_ranks in relevant_documents[:4]:
            positions = {
                index: position for position, index in enumerate(document_indices)
            }
            # Two anchors cover a matching user request and a matching answer
            # heading without copying an entire long conversation into context.
            for anchor in local_ranks[:2]:
                position = positions[anchor]
                start = max(0, position - 2)
                end = min(len(document_indices), position + 9)
                assistant_context.extend(document_indices[start:end])
        promoted_seen: set[int] = set()
        # Direct lexical matches must remain ahead of broad local windows. A
        # previous ordering allowed several windows to consume the whole
        # ledger and hide exact values such as an ingredient count or budget.
        direct_count = min(24, max(8, limit // 3))
        combined = ordinal_promoted + value_promoted + ranked_indices[:direct_count]
        combined.extend(assistant_context)
        combined.extend(ranked_indices[direct_count:])
        ranked_indices = [
            index
            for index in combined
            if not (index in promoted_seen or promoted_seen.add(index))
        ]
    else:
        # Questions about the user's own history should not let long assistant
        # recommendations crowd direct user statements out of the compact
        # ledger.  Keep source-local assistant context for ordinary lookups,
        # because a nearby response may resolve an omitted location or object,
        # but put user facts first for aggregates where recommendations would
        # otherwise look like additional completed events.
        user_ranked = [
            index for index in ranked_indices if candidate_roles[index] != "assistant"
        ]
        assistant_ranked = [
            index for index in ranked_indices if candidate_roles[index] == "assistant"
        ]
        if _needs_high_recall_merge(query):
            ranked_indices = user_ranked + assistant_ranked
        else:
            nearby_context: list[int] = []
            # Follow global anchor rank rather than selected-document order so
            # the exact source conversation's neighbourhood precedes a long
            # derived-memory copy of the same statement.
            for anchor in user_ranked[:4]:
                for document_indices in indices_by_document:
                    if anchor not in document_indices:
                        continue
                    positions = {
                        index: position
                        for position, index in enumerate(document_indices)
                    }
                    position = positions[anchor]
                    start = max(0, position - 3)
                    end = min(len(document_indices), position + 13)
                    nearby_context.extend(document_indices[start:end])
                    break
            promoted_seen: set[int] = set()
            combined = (
                user_ranked[:2] + nearby_context + user_ranked[2:] + assistant_ranked
            )
            ranked_indices = [
                index
                for index in combined
                if not (index in promoted_seen or promoted_seen.add(index))
            ]

    if re.search(
        r"\bhow (?:old|many years) (?:will|would) i be\b.*\bwhen\b",
        query,
        re.I,
    ):
        # A projected-age answer joins a weak profile fact (current age) to a
        # future offset attached to the named event. Promote age statements so
        # the event-heavy lexical query cannot push that endpoint out.
        age_lines = [
            index
            for index, line in enumerate(candidate_lines)
            if candidate_roles[index] != "assistant"
            and re.search(
                r"\b(?:i(?:'m| am)|my (?:current )?age|user.{0,20}age|"
                r"turn(?:ed|ing)?)\D{0,10}\d{1,3}\b",
                line,
                re.I,
            )
        ]
        age_set = set(age_lines)
        diverse.update(age_lines)
        ranked_indices = age_lines + [
            index for index in ranked_indices if index not in age_set
        ]
    selected = [index for index in ranked_indices if index in diverse][:limit]
    selected_set = set(selected)
    for index in ranked_indices:
        if len(selected) >= limit:
            break
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
    return "\n".join(f"- {candidates[index]}" for index in selected)


_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}


def _requested_ordinal(query: str) -> int | None:
    match = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)\b", query, re.I)
    if match:
        return int(match.group(1))
    folded = query.casefold()
    for word, value in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", folded):
            return value
    return None


def _requested_list_ordinal(query: str) -> int | None:
    """Return an ordinal only when the question asks for a list position.

    Ordinals are also common inside proper names and scope qualifiers (for
    example ``Fifth Album``, ``second part`` and ``second song``).  Treating
    every such token as "item N" made assistant recall short-circuit on an
    unrelated numbered list before the answer model saw the matching source
    conversation.
    """

    ordinal = _requested_ordinal(query)
    if ordinal is None:
        return None
    folded = " ".join(query.casefold().split())
    ordinal_tokens = [rf"{ordinal}(?:st|nd|rd|th)"]
    ordinal_tokens.extend(
        re.escape(word) for word, value in _ORDINAL_WORDS.items() if value == ordinal
    )
    marker = rf"(?:{'|'.join(ordinal_tokens)})"
    list_nouns = (
        r"(?:answer|bottle|bullet|choice|entry|example|idea|item|job|option|point|"
        r"recommendation|step|suggestion|tip|title)"
    )
    return (
        ordinal
        if (
            re.search(
                rf"\b{marker}\s+(?:[a-z0-9-]+\s+){{0,2}}{list_nouns}s?\b",
                folded,
                re.I,
            )
            or re.search(
                rf"\b{marker}\b.{{0,35}}\b(?:in|on|from)\s+(?:the\s+)?list\b",
                folded,
                re.I,
            )
            or re.search(rf"\blist(?:ed)?\b.{{0,35}}\b{marker}\b", folded, re.I)
        )
        else None
    )


def _line_starts_with_ordinal(line: str, ordinal: int) -> bool:
    fact_body = re.sub(r"^[-*+]\s+(?:<[^>]+>\s*)?", "", line).strip()
    fact_body = re.sub(
        r"^(?:the\s+assistant\s+said:\s*)?(?:session\s+id:\s*\S+\s+)?"
        r"(?:session\s+date:\s*\S+(?:\s+\([^)]*\))?(?:\s+\S+)?\s+)?",
        "",
        fact_body,
        flags=re.I,
    )
    return bool(
        re.match(
            rf"^(?:[*_`]{{0,2}}){ordinal}(?:st|nd|rd|th)?(?:[*_`]{{0,2}})"
            r"\s*[.):\-]\s+",
            fact_body,
            re.I,
        )
    )


def _compact_ledger_line(
    line: str,
    *,
    focus_terms: set[str],
    limit: int = 900,
) -> str:
    """Bound long conversation turns while retaining the best query-local window."""

    if len(line) <= limit:
        return line
    folded = line.casefold()
    positions = [
        match.start()
        for term in focus_terms
        for match in re.finditer(re.escape(term), folded)
    ]
    if not positions:
        return line[: limit - 1].rstrip() + "…"
    half = limit // 2

    def window_score(position: int) -> tuple[int, int]:
        start = max(0, position - half)
        end = min(len(line), start + limit)
        window = folded[start:end]
        return (sum(term in window for term in focus_terms), -position)

    center = max(positions, key=window_score)
    start = max(0, min(center - half, len(line) - limit))
    end = min(len(line), start + limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(line) else ""
    return prefix + line[start:end].strip() + suffix


def _asks_about_assistant_memory(query: str) -> bool:
    """Recognize questions whose answer was authored by the assistant.

    Long-term-memory questions commonly refer to the assistant indirectly as
    "the example you gave" or "that name you mentioned".  Treating the first
    person in "I was reviewing our conversation" as the subject used to remove
    every ``source=AI`` fact before the answer pass.
    """

    return bool(
        re.search(
            r"\b(?:assistant|your (?:answer|example|recommendation|suggestion)|"
            r"you (?:said|mentioned|gave|suggested|recommended|told|wrote|"
            r"created|provided|listed|named|called|described|explained))\b",
            query,
            re.I,
        )
        or (
            re.search(
                r"\b(?:previous|prior|earlier)\s+(?:chat|conversation|discussion)\b",
                query,
                re.I,
            )
            and re.search(
                r"\b(?:remind me|recall|confirm|what (?:was|did)|which was|how many)\b",
                query,
                re.I,
            )
        )
        or (
            re.search(
                r"\b(?:we|you)\s+(?:discussed|talked about|covered)\b", query, re.I
            )
            and re.search(
                r"\b(?:remind me|recall|confirm|what|which|how)\b", query, re.I
            )
        )
    )


def _execution_is_high_confidence(
    *,
    plan_subgoal_count: int,
    executable_fact_count: int,
    execution_complete: bool,
    execution_fact_count: int,
) -> bool:
    """Keep deterministic execution inside a narrow, auditable evidence set.

    Hybrid recall deliberately favors recall and can attach many weak lexical
    matches to a subgoal. Counting, sorting, or subtracting that broad set
    converts retrieval noise into a confidently wrong answer. A small bound
    retains exact two-event calculations and tiny multi-goal plans while larger
    sets fall back to evidence-grounded synthesis.
    """

    if not execution_complete or execution_fact_count < 1:
        return False
    return executable_fact_count <= max(2, plan_subgoal_count * 2)


class MemoryRetriever(BaseModel):
    """Search current/raw/rewrite plus hierarchical long-term memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repository: MemoryRepository
    llm: FlowLLM
    metrics: FlowMetrics
    config: RetrievalConfig
    index_store: FactIndexStore

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.retriever")

    def search(self, request: SearchRequest) -> SearchResult:
        strategy = request.strategy or self.config.strategy
        if strategy == SearchStrategy.AUTO:
            return self._search_auto(request, strategy)
        if strategy == SearchStrategy.HIERARCHICAL:
            return self._search_hierarchical(request, strategy)
        return self._search_flat(request, strategy)

    def _search_auto(
        self, request: SearchRequest, strategy: SearchStrategy
    ) -> SearchResult:
        """Use structured facts first and reserve Agentic for explicit callers."""

        documents = self._flat_documents(request, strategy)
        context = {
            "store_id": self.repository.scope.store_id,
            "user_id": self.repository.scope.user_id,
            "strategy": strategy.value,
            "candidate_count": len(documents),
            "limit": request.limit,
        }
        with observe_flow(
            self.metrics,
            self.logger,
            module="retriever",
            flow="auto_search",
            context=context,
        ):
            if not documents:
                return SearchResult(
                    query=request.query,
                    strategy=strategy,
                    route="abstain",
                    abstention_reason="empty_catalog",
                )
            indexable = [
                item
                for item in documents
                if item.metadata.kind
                in {"current", "raw", "rewrite", "memory", "topic"}
            ]
            snapshot = self.index_store.load_with_overlay(indexable)
            plan = compile_query_plan(
                request.query,
                as_of=request.as_of,
                subgoal_min_candidates=self.config.subgoal_min_candidates,
            )
            recalled = hybrid_recall(
                plan,
                snapshot,
                documents,
                limit=request.limit,
                config=self.config,
                semantic_enabled=self.index_store.config.embedding_enabled,
            )
            matched_fact_ids = {
                fact_id
                for item in recalled.coverage.subgoals
                for fact_id in item.matched_fact_ids
            }
            executable_facts = [
                fact
                for fact in recalled.facts
                if not matched_fact_ids or fact.fact_id in matched_fact_ids
            ]
            # Untagged legacy/evidence documents remain searchable. Structured
            # recall is the primary order; document BM25 fills uncovered slots.
            selected = _merge_unique(
                recalled.documents,
                self._bm25(
                    request.query,
                    documents,
                    request.limit,
                ),
                request.limit,
            )
            evidence = verify_evidence(
                executable_facts,
                documents,
                recalled.coverage,
                threshold=self.config.evidence_sufficiency_threshold,
                allow_assistant_only=any(
                    goal.source_role == "assistant" for goal in plan.subgoals
                ),
                relations=snapshot.relations,
                block_on_conflicts=plan.intent
                in {
                    QueryIntent.STATE,
                    QueryIntent.PREFERENCE,
                    QueryIntent.ABSTENTION_CHECK,
                    QueryIntent.AMBIGUOUS,
                },
            )
            selected = _merge_unique(
                evidence.documents,
                selected,
                request.limit,
            )
            execution = execute_query_plan(
                plan,
                executable_facts,
                snapshot.relations,
                relation_confidence_threshold=(
                    self.index_store.config.relation_confidence_threshold
                ),
            )
            trust_execution = _execution_is_high_confidence(
                plan_subgoal_count=len(plan.subgoals),
                executable_fact_count=len(executable_facts),
                execution_complete=execution.complete,
                execution_fact_count=len(execution.fact_ids),
            )
            route = (
                "structured"
                if trust_execution
                else "direct"
                if plan.intent == QueryIntent.DIRECT
                else "llm_synthesis"
            )
            answer: str | None = None
            if request.answer:
                if trust_execution and evidence.sufficient:
                    answer = execution.render()
                if answer is None and selected and evidence.sufficient:
                    answer = self._answer(request, selected)
            hits = [_to_hit(item) for item in selected[: request.limit]]
            abstention_reason = None
            if request.answer and not answer:
                route = "abstain"
                abstention_reason = evidence.reason or "no_verified_answer"
            self.logger.info(
                "auto_search_completed facts=%d hits=%d coverage=%.3f "
                "sufficiency=%.3f route=%s persisted_index=%s",
                len(recalled.facts),
                len(hits),
                recalled.coverage.score,
                evidence.sufficiency_score,
                route,
                snapshot.persisted,
            )
            return SearchResult(
                query=request.query,
                strategy=strategy,
                hits=hits,
                answer=answer,
                route=route,
                coverage=(
                    recalled.coverage.model_dump(mode="json") if request.debug else None
                ),
                execution=(
                    execution.model_dump(mode="json") if request.debug else None
                ),
                abstention_reason=abstention_reason,
            )

    def _search_flat(
        self, request: SearchRequest, strategy: SearchStrategy
    ) -> SearchResult:
        documents = self._flat_documents(request, strategy)
        context = {
            "store_id": self.repository.scope.store_id,
            "user_id": self.repository.scope.user_id,
            "strategy": strategy.value,
            "candidate_count": len(documents),
            "limit": request.limit,
        }
        with observe_flow(
            self.metrics,
            self.logger,
            module="retriever",
            flow="search",
            context=context,
        ):
            if not documents:
                return SearchResult(query=request.query, strategy=strategy)
            if strategy == SearchStrategy.LLM:
                selected = self._select_documents(request, documents)
            elif strategy == SearchStrategy.BM25:
                selected = self._bm25(request.query, documents, request.limit)
            elif strategy in {
                SearchStrategy.BM25_PARTITION,
                SearchStrategy.FOLDER_BM25_PARTITION,
            }:
                selected = self._bm25_partition(request.query, documents, request.limit)
            elif strategy == SearchStrategy.LLM_AND_BM25:
                selected = _merge_unique(
                    self._select_documents(request, documents),
                    self._bm25(request.query, documents, request.limit),
                    request.limit,
                )
            elif strategy == SearchStrategy.LLM_AND_BM25_PARTITION:
                llm_selected = self._select_documents(request, documents)
                selected_ids = {item.metadata.id for item in llm_selected}
                remaining = [
                    item for item in documents if item.metadata.id not in selected_ids
                ]
                selected = _merge_unique(
                    llm_selected,
                    self._bm25_partition(request.query, remaining, request.limit),
                    request.limit,
                )
            elif strategy == SearchStrategy.LLM_OR_BM25:
                selected = self._select_documents(request, documents)
                if not selected:
                    selected = self._bm25(request.query, documents, request.limit)
            elif strategy == SearchStrategy.AGENTIC:
                with observe_flow(
                    self.metrics,
                    self.logger,
                    module="retriever",
                    flow="agentic_search",
                    context={"candidate_count": len(documents)},
                ):
                    agent_selected = AgenticRetriever(
                        documents=documents,
                        llm=self.llm,
                        config=self.config,
                        limit=request.limit,
                    ).search(request.query)
                    # Preserve a deterministic lossless-evidence lane beside
                    # the stochastic tool-using agent. Aggregates especially
                    # need several independent source sessions; broad derived
                    # leaves must not consume the whole hit budget before
                    # those source conversations reach answer synthesis.
                    high_recall = _needs_high_recall_merge(request.query)
                    evidence_limit = 14 if high_recall else 6
                    lexical_limit = 8 if high_recall else 4
                    evidence_documents = [
                        item for item in documents if item.metadata.kind == "evidence"
                    ]
                    selected = _merge_unique(
                        self._target_date_evidence(
                            request.query,
                            documents,
                            min(6, request.limit),
                        ),
                        self._temporal_window_evidence(
                            request.query,
                            documents,
                            min(10, request.limit),
                        ),
                        request.limit,
                    )
                    selected = _merge_unique(
                        selected,
                        self._bm25_partition(
                            _expand_focus_query(request.query),
                            evidence_documents,
                            min(evidence_limit, request.limit),
                        ),
                        request.limit,
                    )
                    selected = _merge_unique(
                        selected,
                        agent_selected,
                        request.limit,
                    )
                    selected = _merge_unique(
                        selected,
                        self._bm25_partition(
                            _expand_focus_query(request.query),
                            documents,
                            min(lexical_limit, request.limit),
                        ),
                        request.limit,
                    )
            else:  # pragma: no cover - exhaustive StrEnum guard
                raise ValueError(f"unsupported search strategy: {strategy}")

            hits = [_to_hit(item) for item in selected[: request.limit]]
            answer = self._answer(request, selected[: request.limit])
            self.metrics.documents_total.labels(
                module="retriever", flow="search", kind="hit"
            ).inc(len(hits))
            self.logger.info(
                "search_completed strategy=%s candidates=%d hits=%d",
                strategy.value,
                len(documents),
                len(hits),
            )
            return SearchResult(
                query=request.query,
                strategy=strategy,
                hits=hits,
                answer=answer,
            )

    def _flat_documents(
        self, request: SearchRequest, strategy: SearchStrategy
    ) -> list[MemoryDocument]:
        sources = request.sources
        if strategy == SearchStrategy.FOLDER_BM25_PARTITION:
            try:
                sources = {SearchSource(self.config.search_folder.strip("/"))}
            except ValueError as exc:
                raise ValueError(
                    "retrieval.search_folder must be evidence, current, raw, rewrite, or doc"
                ) from exc
        documents = self.repository.list_searchable_working(sources)
        if SearchSource.DOC in sources:
            documents.extend(self.repository.list_legacy_topics())
            documents.extend(self.repository.list_memory_documents())
        return _merge_unique([], documents, len(documents))

    def _bm25(
        self, query: str, documents: list[MemoryDocument], limit: int
    ) -> list[MemoryDocument]:
        return bm25_documents(
            query,
            documents,
            limit,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

    def _bm25_partition(
        self, query: str, documents: list[MemoryDocument], limit: int
    ) -> list[MemoryDocument]:
        return bm25_partitions(
            query,
            documents,
            limit,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

    def _target_date_evidence(
        self,
        query: str,
        documents: list[MemoryDocument],
        limit: int,
    ) -> list[MemoryDocument]:
        """Recall source sessions at the date resolved from a relative query.

        Agentic tools search text, so a query such as ``last Friday`` cannot
        lexically match a source session that only records ``2026-07-24``.
        The query planner already resolves relative expressions against the
        question date.  Use that resolved date as an independent recall lane,
        restricted to lossless evidence documents, and use BM25 only to rank
        ties on the same date.  This is intentionally a retrieval operation:
        answer generation still has to identify the relevant event inside the
        selected source conversation.
        """

        target = build_fact_query_plan(query).target_date
        if target is None or limit <= 0:
            return []

        dated: list[tuple[int, MemoryDocument]] = []
        for document in documents:
            if document.metadata.kind != "evidence":
                continue
            distances: list[int] = []
            body = strip_yaml_front_matter(document.content)
            for match in re.finditer(
                r"\b(?:time|observed)=(\d{4}[-/]\d{1,2}[-/]\d{1,2})"
                r"|\bSession date:\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                body,
                re.I,
            ):
                value = next(item for item in match.groups() if item)
                try:
                    candidate = date.fromisoformat(value.replace("/", "-"))
                except ValueError:
                    continue
                distances.append(abs((candidate - target).days))
            if distances:
                dated.append((min(distances), document))
        if not dated:
            return []

        nearest_distance = min(distance for distance, _document in dated)
        # Exact matches are preferred.  A one-day tolerance covers datasets
        # that normalize a conversational timezone across midnight without
        # admitting unrelated sessions from the same week.
        nearest = [
            document
            for distance, document in dated
            if distance <= max(1, nearest_distance)
        ]
        return self._bm25_partition(query, nearest, min(limit, len(nearest)))

    def _temporal_window_evidence(
        self,
        query: str,
        documents: list[MemoryDocument],
        limit: int,
    ) -> list[MemoryDocument]:
        """Recall source sessions inside an explicit query time window.

        Category-level chronology questions often use a hypernym such as
        ``events`` while the source naturally names only a race, concert, or
        trip.  Restricting a lexical lane to the requested calendar window
        raises recall without flooding the answer with the user's full history.
        """

        if limit <= 0:
            return []
        plan = build_fact_query_plan(query)
        question_date = plan.question_date
        if question_date is None:
            return []
        window = _explicit_temporal_window(plan.body, question_date)
        if window is None:
            return []
        start, end = window
        in_window: list[MemoryDocument] = []
        for document in documents:
            if document.metadata.kind != "evidence":
                continue
            body = strip_yaml_front_matter(document.content)
            dates: list[date] = []
            for match in re.finditer(
                r"\bSession date:\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                body,
                re.I,
            ):
                try:
                    dates.append(date.fromisoformat(match.group(1).replace("/", "-")))
                except ValueError:
                    continue
            if any(start <= value <= end for value in dates):
                in_window.append(document)
        if not in_window:
            return []
        return self._bm25_partition(
            _expand_focus_query(query),
            in_window,
            min(limit, len(in_window)),
        )

    def _answer(
        self, request: SearchRequest, documents: list[MemoryDocument]
    ) -> str | None:
        if not request.answer or not documents:
            return None
        collapsed = _collapse_lineage(documents)
        # The selected set may omit intermediate RAW/REWRITE nodes, making it
        # impossible for lineage collapse to recognize that a hidden ancestor
        # is ultimately an authoritative evidence conversation.  Reinsert all
        # selected evidence explicitly and put it first so exact source turns
        # win ties over compact derived memory copies.
        answer_documents = _merge_unique(
            [item for item in documents if item.metadata.kind == "evidence"],
            collapsed,
            len(documents),
        )
        if _needs_high_recall_merge(request.query):
            ledger_limit = 160
        elif _needs_personalization_context(request.query):
            ledger_limit = 120
        elif _asks_about_assistant_memory(request.query):
            ledger_limit = 72
        else:
            ledger_limit = 48
        evidence_ledger = _answer_evidence_ledger(
            request.query,
            answer_documents,
            limit=ledger_limit,
        )
        strategy = request.strategy or self.config.strategy
        use_compact_evidence_context = strategy in {
            SearchStrategy.AUTO,
            SearchStrategy.AGENTIC,
        }
        if use_compact_evidence_context:
            prompt = AGENTIC_ANSWER_PROMPT.format(
                query=request.query,
                evidence_ledger=evidence_ledger,
                primary_source_context=_primary_source_context(
                    request.query,
                    answer_documents,
                ),
                task_guidance=answer_task_guidance(request.query),
            )
        else:
            prompt = ANSWER_PROMPT.format(
                query=request.query,
                context="\n\n".join(
                    strip_yaml_front_matter(item.content) for item in answer_documents
                ),
                evidence_ledger=evidence_ledger,
                task_guidance=answer_task_guidance(request.query),
            )
        raw = self.llm.complete(
            LLMRequest(
                operation="search_answer",
                messages=[
                    ChatMessage(
                        role="user",
                        content=prompt,
                    )
                ],
            )
        )
        draft_answer = _json_answer(raw).strip()
        if not _is_insufficient_answer(draft_answer) or not _needs_answer_audit(
            request.query, draft_answer
        ):
            return draft_answer
        reviewed = self.llm.complete(
            LLMRequest(
                operation="search_answer",
                messages=[
                    ChatMessage(
                        role="user",
                        content=ANSWER_REVIEW_PROMPT.format(
                            query=request.query,
                            evidence_ledger=evidence_ledger,
                            source_context=_answer_review_context(
                                request.query,
                                answer_documents,
                            ),
                            draft_result=raw,
                            task_guidance=answer_task_guidance(request.query),
                        ),
                    )
                ],
            )
        )
        reviewed_answer = _json_answer(reviewed)
        if _is_insufficient_answer(reviewed_answer) and not _is_insufficient_answer(
            draft_answer
        ):
            return draft_answer
        return reviewed_answer

    def _search_hierarchical(
        self, request: SearchRequest, strategy: SearchStrategy
    ) -> SearchResult:
        directory_limit = (
            request.directory_limit
            if "directory_limit" in request.model_fields_set
            else self.config.directory_limit
        )
        context = {
            "store_id": self.repository.scope.store_id,
            "user_id": self.repository.scope.user_id,
            "limit": request.limit,
            "directory_limit": directory_limit,
            "sources": sorted(item.value for item in request.sources),
            "strategy": strategy.value,
        }
        with observe_flow(
            self.metrics,
            self.logger,
            module="retriever",
            flow="search",
            context=context,
        ):
            working = self.repository.list_searchable_working(request.sources)
            legacy = (
                self.repository.list_legacy_topics()
                if SearchSource.DOC in request.sources
                else []
            )
            direct = [*working, *legacy]
            topics = (
                self.repository.list_directory_topics()
                if SearchSource.DOC in request.sources
                else []
            )
            if SearchSource.DOC in request.sources:
                topics = self._with_orphan_directories(request, topics)
            self.logger.info(
                "directory_catalog_loaded directory_count=%d",
                len(topics),
            )
            if not direct and not topics:
                self.logger.info("search_empty_catalog context=%s", context)
                return SearchResult(query=request.query, strategy=strategy)

            with observe_flow(
                self.metrics,
                self.logger,
                module="retriever",
                flow="search_scope",
                context={
                    "working_count": len(direct),
                    "directory_count": len(topics),
                },
            ):
                selection = self._select_scope(
                    request, direct, topics, directory_limit=directory_limit
                )
            with observe_flow(
                self.metrics,
                self.logger,
                module="retriever",
                flow="search_working_documents",
                context={"candidate_count": len(direct)},
            ):
                direct_by_id = {item.metadata.id: item for item in direct}
                selected_direct = _documents_by_ids(
                    selection.working_document_ids,
                    direct_by_id,
                    self.config.working_candidate_limit,
                )
                if direct:
                    selected_direct = _merge_unique(
                        selected_direct,
                        _keyword_fallback(
                            request.query,
                            direct,
                            self.config.working_candidate_limit,
                        ),
                        self.config.working_candidate_limit,
                    )
                counts = {
                    kind: sum(item.metadata.kind == kind for item in working)
                    for kind in ("current", "raw", "rewrite")
                }
                self.logger.info(
                    "working_search_completed current_count=%d raw_count=%d "
                    "rewrite_count=%d selected_count=%d",
                    counts["current"],
                    counts["raw"],
                    counts["rewrite"],
                    len(selected_direct),
                )

            with observe_flow(
                self.metrics,
                self.logger,
                module="retriever",
                flow="search_directories",
                context={"catalog_count": len(topics)},
            ):
                topic_by_id = {item.metadata.id: item for item in topics}
                selected_topics = _documents_by_ids(
                    selection.directory_ids,
                    topic_by_id,
                    directory_limit,
                )
                if topics:
                    selected_topics = _merge_unique(
                        selected_topics,
                        _keyword_fallback(request.query, topics, directory_limit),
                        directory_limit,
                    )
                self.logger.info(
                    "directory_search_completed catalog_count=%d selected_count=%d",
                    len(topics),
                    len(selected_topics),
                )

            candidate_memories: list[MemoryDocument] = []
            for topic in selected_topics:
                directory_documents = self.repository.list_memory_documents(
                    topic.metadata.id
                )
                if topic.metadata.document_count != len(directory_documents):
                    self.metrics.documents_total.labels(
                        module="retriever",
                        flow="search_directories",
                        kind="stale_topic",
                    ).inc()
                    self.logger.warning(
                        "directory_topic_stale directory_id=%s expected=%d actual=%d",
                        topic.metadata.id,
                        topic.metadata.document_count,
                        len(directory_documents),
                    )
                candidate_memories.extend(directory_documents)

            with observe_flow(
                self.metrics,
                self.logger,
                module="retriever",
                flow="search_documents",
                context={"candidate_count": len(candidate_memories)},
            ):
                if len(candidate_memories) == 1:
                    # The directory stage has already selected the only possible
                    # leaf. A second LLM selection cannot narrow this candidate.
                    selected_memories = list(candidate_memories)
                    self.logger.info(
                        "document_selection_skipped reason=single_candidate"
                    )
                else:
                    selected_memories = self._select_documents(
                        request, candidate_memories
                    )
            if candidate_memories:
                selected_memories = _merge_unique(
                    selected_memories,
                    _keyword_fallback(request.query, candidate_memories, request.limit),
                    request.limit,
                )
            if not selected_memories and SearchSource.DOC in request.sources:
                # A stale or incomplete directory topic must not make a fact
                # permanently unreachable. This is deliberately a fallback only.
                all_memories = self.repository.list_memory_documents()
                selected_memories = _keyword_fallback(
                    request.query, all_memories, request.limit
                )
            self.logger.info(
                "document_search_completed directory_count=%d candidate_count=%d "
                "selected_count=%d",
                len(selected_topics),
                len(candidate_memories),
                len(selected_memories),
            )

            selected = _rank_and_limit(
                selected_memories, selected_direct, request.limit
            )
            hits = [_to_hit(item) for item in selected]
            answer = self._answer(request, selected)
            self.metrics.documents_total.labels(
                module="retriever", flow="search", kind="hit"
            ).inc(len(hits))
            self.logger.info(
                "search_completed working=%d directories=%d candidates=%d hits=%d",
                len(working),
                len(topics),
                len(candidate_memories),
                len(hits),
            )
            return SearchResult(
                query=request.query,
                strategy=strategy,
                hits=hits,
                answer=answer,
            )

    def _with_orphan_directories(
        self, request: SearchRequest, topics: list[MemoryDocument]
    ) -> list[MemoryDocument]:
        by_id = {item.metadata.id: item for item in topics}
        for directory_id in self.repository.list_directory_ids():
            if directory_id in by_id:
                continue
            documents = self.repository.list_memory_documents(directory_id)
            if not documents:
                continue
            title = documents[0].metadata.title or "Memory"
            generated = DirectoryTopicBuilder().build(
                directory_title=title,
                document_contents=[document.content for document in documents],
            )
            synthetic = MemoryDocument(
                metadata=DocumentMetadata(
                    id=directory_id,
                    kind="directory_topic",
                    title=title,
                    summary=title,
                    document_count=len(documents),
                ),
                content=generated.body,
            )
            by_id[directory_id] = synthetic
            self.metrics.documents_total.labels(
                module="retriever", flow="search_directories", kind="orphan_directory"
            ).inc()
            self.logger.warning(
                "orphan_directory_discovered directory_id=%s documents=%d",
                directory_id,
                len(documents),
            )
        return sorted(by_id.values(), key=lambda item: item.metadata.id)

    def _select_scope(
        self,
        request: SearchRequest,
        direct: list[MemoryDocument],
        topics: list[MemoryDocument],
        *,
        directory_limit: int,
    ) -> SearchScopeSelection:
        working_catalog = json.dumps(
            [
                {
                    "id": item.metadata.id,
                    "kind": "memory"
                    if item.metadata.kind in {"memory", "topic"}
                    else item.metadata.kind,
                    "body_preview": strip_yaml_front_matter(item.content)[
                        : self.config.working_document_preview_chars
                    ],
                }
                for item in direct
            ],
            ensure_ascii=False,
        )
        directory_catalog = json.dumps(
            [
                {
                    "id": item.metadata.id,
                    "body_preview": strip_yaml_front_matter(item.content)[
                        : self.config.directory_topic_preview_chars
                    ],
                }
                for item in topics
            ],
            ensure_ascii=False,
        )
        raw = self.llm.complete(
            LLMRequest(
                operation="search_scope",
                messages=[
                    ChatMessage(
                        role="user",
                        content=SEARCH_SCOPE_PROMPT.format(
                            query=request.query,
                            working_limit=self.config.working_candidate_limit,
                            directory_limit=directory_limit,
                            working_catalog=working_catalog,
                            directory_catalog=directory_catalog,
                        ),
                    )
                ],
            )
        )
        try:
            return parse_json_model(raw, SearchScopeSelection)
        except Exception:
            self.logger.warning(
                "search_scope_selection_invalid response_length=%d", len(raw)
            )
            return SearchScopeSelection()

    def _select_documents(
        self, request: SearchRequest, documents: list[MemoryDocument]
    ) -> list[MemoryDocument]:
        if not documents:
            return []
        catalog = json.dumps(
            [
                {
                    "id": item.metadata.id,
                    "directory_id": item.metadata.directory_id,
                    "body_preview": strip_yaml_front_matter(item.content)[
                        : self.config.working_document_preview_chars
                    ],
                }
                for item in documents
            ],
            ensure_ascii=False,
        )
        raw = self.llm.complete(
            LLMRequest(
                operation="search_documents",
                messages=[
                    ChatMessage(
                        role="user",
                        content=SEARCH_DOCUMENT_PROMPT.format(
                            query=request.query,
                            limit=request.limit,
                            catalog=catalog,
                        ),
                    )
                ],
            )
        )
        try:
            selected = parse_json_model(raw, DocumentSelection)
        except Exception:
            self.logger.warning(
                "search_document_selection_invalid response_length=%d", len(raw)
            )
            return []
        return _documents_by_ids(
            selected.document_ids,
            {item.metadata.id: item for item in documents},
            request.limit,
        )


def _documents_by_ids(
    ids: list[str], by_id: dict[str, MemoryDocument], limit: int
) -> list[MemoryDocument]:
    chosen: list[MemoryDocument] = []
    seen: set[str] = set()
    for document_id in ids:
        if document_id in by_id and document_id not in seen:
            chosen.append(by_id[document_id])
            seen.add(document_id)
        if len(chosen) >= limit:
            break
    return chosen


def _keyword_fallback(
    query: str, documents: list[MemoryDocument], limit: int
) -> list[MemoryDocument]:
    terms = {term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", query)}
    scored: list[tuple[int, float, MemoryDocument]] = []
    for document in documents:
        haystack = (
            f"{document.metadata.title} {document.metadata.summary} {document.content}"
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, document.metadata.created_at.timestamp(), document))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def _merge_unique(
    primary: list[MemoryDocument], secondary: list[MemoryDocument], limit: int
) -> list[MemoryDocument]:
    result: list[MemoryDocument] = []
    seen: set[str] = set()
    for document in [*primary, *secondary]:
        if document.metadata.id not in seen:
            result.append(document)
            seen.add(document.metadata.id)
        if len(result) >= limit:
            break
    return result


def _rank_and_limit(
    memories: list[MemoryDocument], direct: list[MemoryDocument], limit: int
) -> list[MemoryDocument]:
    """Interleave relevance ranks; prefer the more mature item at equal rank."""
    result: list[MemoryDocument] = []
    seen: set[str] = set()
    for index in range(max(len(memories), len(direct))):
        same_rank = [
            documents[index]
            for documents in (memories, direct)
            if index < len(documents)
        ]
        same_rank.sort(
            key=lambda item: {
                "memory": 4,
                "topic": 4,
                "rewrite": 3,
                "raw": 2,
                "current": 1,
            }.get(item.metadata.kind, 0),
            reverse=True,
        )
        for document in same_rank:
            if document.metadata.id not in seen:
                result.append(document)
                seen.add(document.metadata.id)
            if len(result) >= limit:
                return result
    return result


def _collapse_lineage(documents: list[MemoryDocument]) -> list[MemoryDocument]:
    by_id = {item.metadata.id: item for item in documents}
    maturity = {"memory": 4, "topic": 4, "rewrite": 3, "raw": 2, "current": 1}
    ordered = sorted(
        documents,
        key=lambda item: maturity.get(item.metadata.kind, 0),
        reverse=True,
    )
    kept: list[MemoryDocument] = []
    hidden_ids: set[str] = set()

    def mark_sources(document: MemoryDocument) -> None:
        for source_id in document.metadata.source_ids:
            if source_id in hidden_ids:
                continue
            # Evidence is an authoritative source conversation, not an
            # intermediate representation.  A derived memory may compact an
            # entire assistant turn into one long line, so hiding the selected
            # evidence loses list tails and local question/answer structure at
            # answer time.  Working-layer ancestors remain safely collapsed.
            source = by_id.get(source_id)
            if source is not None and source.metadata.kind == "evidence":
                continue
            hidden_ids.add(source_id)
            if source_id in by_id:
                mark_sources(by_id[source_id])

    for document in ordered:
        if document.metadata.id in hidden_ids:
            continue
        kept.append(document)
        mark_sources(document)
    return kept


def _to_hit(document: MemoryDocument) -> SearchHit:
    kind = (
        "memory"
        if document.metadata.kind in {"memory", "topic"}
        else document.metadata.kind
    )
    if kind not in {"evidence", "current", "raw", "rewrite", "memory"}:
        raise ValueError(f"unsupported search hit kind: {kind}")
    return SearchHit(
        id=document.metadata.id,
        kind=kind,
        directory_id=document.metadata.directory_id or None,
        title=document.metadata.title,
        summary=document.metadata.summary,
        content=document.content,
        key=document.key,
        source_ids=document.metadata.source_ids,
    )


__all__ = ["MemoryRetriever"]
