"""Independent memory extraction flow."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import ExtractionConfig
from ..index.projector import assign_fact_ids
from ..llm import FlowLLM
from ..models import (
    ChatMessage,
    DocumentMetadata,
    ExtractionRequest,
    ExtractionResult,
    LLMRequest,
    MemoryDocument,
    beijing_now,
)
from ..observability import FlowMetrics, observe_flow
from ..prompts import EXTRACT_PROMPT
from ..repository import MemoryRepository
from ..utils.markdown import normalize_headings_to_h1
from ..utils.paths import KeyLayout
from ..utils.tokens import estimate_tokens


class MemoryExtractor(BaseModel):
    """Extract messages and append them to one instance-owned CURRENT object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repository: MemoryRepository
    layout: KeyLayout
    llm: FlowLLM
    metrics: FlowMetrics
    config: ExtractionConfig
    instance_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)

    @field_validator("instance_id")
    @classmethod
    def validate_instance_id(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ch in value for ch in "/\\") or value in {".", ".."}:
            raise ValueError("instance_id must be a safe path segment")
        return value

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.extractor")

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        key = self.layout.current_key(self.instance_id)
        extracted_at = beijing_now()
        sequence_timestamp = (
            request.sequence
            if request.sequence is not None
            else int(extracted_at.timestamp())
        )
        observed_dates = _session_observed_dates(request.messages)
        observed_date = next(iter(observed_dates)) if len(observed_dates) == 1 else None
        (
            evidence_id,
            evidence_key,
            origin_by_observed,
            origin_by_source,
        ) = self._archive_evidence_groups(
            request.messages,
            sequence_timestamp=sequence_timestamp,
            observed_date=observed_date,
            created_at=extracted_at,
        )
        context = {
            "store_id": self.repository.scope.store_id,
            "user_id": self.repository.scope.user_id,
            "instance_id": str(self.instance_id),
            "sequence_timestamp": sequence_timestamp,
            "message_count": len(request.messages),
            "infer": request.infer,
        }
        with observe_flow(
            self.metrics,
            self.logger,
            module="extractor",
            flow="extract",
            context=context,
        ):
            if request.infer:
                prompt = EXTRACT_PROMPT.format(
                    extraction_timestamp=sequence_timestamp,
                    messages=json.dumps(
                        [
                            message.model_dump(exclude_none=True, mode="json")
                            for message in request.messages
                        ],
                        ensure_ascii=False,
                    ),
                )
                extracted = self.llm.complete(
                    LLMRequest(
                        operation="extract",
                        messages=[ChatMessage(role="user", content=prompt)],
                    )
                ).strip()
            else:
                extracted = _render_direct_content(
                    request.messages,
                    sequence_timestamp=sequence_timestamp,
                    evidence_id=evidence_id,
                    origin_by_source=origin_by_source,
                )
                self.logger.info(
                    "extraction_inference_skipped key=%s sequence_timestamp=%d",
                    key,
                    sequence_timestamp,
                )
            if not extracted or extracted.upper() == "NO_MEMORY":
                self.logger.info("extraction_empty key=%s", key)
                return ExtractionResult(
                    instance_id=self.instance_id,
                    current_key=key,
                    sequence_timestamp=sequence_timestamp,
                    extracted_content="",
                    evidence_key=evidence_key,
                    appended=False,
                    bytes_written=0,
                    tokens=0,
                )

            normalized = (
                _normalize_extraction_metadata(
                    extracted,
                    sequence_timestamp=sequence_timestamp,
                    observed_date=observed_date,
                    allowed_observed_dates=observed_dates,
                    origin_id=evidence_id,
                    origin_by_observed=origin_by_observed,
                )
                if request.infer
                else extracted
            )
            if normalized != extracted:
                self.logger.info(
                    "extraction_metadata_normalized key=%s sequence_timestamp=%d",
                    key,
                    sequence_timestamp,
                )
            extracted = normalized
            extracted = assign_fact_ids(extracted, fallback_origin=evidence_id)

            now = extracted_at
            existed = self.repository.store.exists(key)
            if existed:
                document = self.repository.read(key)
                # CURRENT is a logical append-only log: never rewrite or remove an
                # earlier fact during extraction. S3 replaces the object on PUT, but
                # the previous body remains an exact prefix of the new body.
                document.content = f"{document.content.rstrip()}\n\n{extracted}"
                document.metadata.updated_at = now
            else:
                # Maintenance removes consumed CURRENT objects. The next extraction
                # starts a fresh object at the same instance-owned key.
                document = MemoryDocument(
                    metadata=DocumentMetadata(
                        id=f"CURRENT_{self.instance_id}",
                        kind="current",
                        created_at=now,
                        updated_at=now,
                    ),
                    content=extracted,
                    key=key,
                )
            bytes_written = len(document.content.encode("utf-8"))
            # ``origin`` and ``observed`` are provenance, not semantic payload.
            # Do not let repeated provenance metadata change CURRENT rotation
            # boundaries or evaluation batching behaviour.
            token_count = estimate_tokens(_semantic_memory_content(document.content))
            rotated_key: str | None = None
            if token_count > self.config.current_max_tokens:
                full_number = self._next_full_number()
                rotated_key = self.layout.current_full_key(
                    self.instance_id, full_number
                )
                document.metadata.id = f"CURRENT_{self.instance_id}_full_{full_number}"
                self.repository.write(rotated_key, document)
                if existed:
                    self.repository.store.delete(key)
                self.metrics.documents_total.labels(
                    module="extractor", flow="rotate_current", kind="current_full"
                ).inc()
                self.logger.info(
                    "current_rotated active_key=%s full_key=%s tokens=%d threshold=%d",
                    key,
                    rotated_key,
                    token_count,
                    self.config.current_max_tokens,
                )
            else:
                self.repository.write(key, document)
            self.metrics.documents_total.labels(
                module="extractor", flow="extract", kind="current"
            ).inc()
            self.logger.info(
                "current_written key=%s appended=%s bytes=%d tokens=%d",
                rotated_key or key,
                existed,
                bytes_written,
                token_count,
            )
            return ExtractionResult(
                instance_id=self.instance_id,
                current_key=key,
                sequence_timestamp=sequence_timestamp,
                rotated_key=rotated_key,
                extracted_content=extracted,
                evidence_key=evidence_key,
                appended=existed,
                bytes_written=bytes_written,
                tokens=token_count,
            )

    def _archive_evidence(
        self,
        messages: list[ChatMessage],
        *,
        sequence_timestamp: int,
        observed_date: str | None,
        created_at: datetime,
    ) -> tuple[str, str]:
        """Persist the exact scoped source before lossy memory extraction."""

        payload = json.dumps(
            [message.model_dump(exclude_none=True, mode="json") for message in messages],
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(
            f"{self.instance_id}\0{sequence_timestamp}\0{payload}".encode()
        ).hexdigest()[:24]
        evidence_id = f"EVIDENCE_{digest}"
        key = self.layout.evidence_key(evidence_id)
        if not self.repository.store.exists(key):
            blocks = [
                f"# Source conversation {observed_date or sequence_timestamp}",
                *[
                    f"[{message.role.upper()}]\n{message.content.strip()}"
                    for message in messages
                ],
            ]
            content = "\n\n".join(blocks)
            self.repository.write(
                key,
                MemoryDocument(
                    metadata=DocumentMetadata(
                        id=evidence_id,
                        kind="evidence",
                        title="Source conversation",
                        summary=re.sub(r"\s+", " ", content)[:240],
                        source_ids=sorted(
                            {
                                message.source_id
                                for message in messages
                                if message.source_id
                            }
                        ),
                        created_at=created_at,
                        updated_at=created_at,
                    ),
                    content=content,
                    key=key,
                ),
            )
            self.metrics.documents_total.labels(
                module="extractor", flow="archive_evidence", kind="evidence"
            ).inc()
        return evidence_id, key

    def _archive_evidence_groups(
        self,
        messages: list[ChatMessage],
        *,
        sequence_timestamp: int,
        observed_date: str | None,
        created_at: datetime,
    ) -> tuple[str, str, dict[str, str], dict[str, str]]:
        """Archive source-tagged sessions separately from one batched LLM request."""

        source_ids = list(
            dict.fromkeys(message.source_id for message in messages if message.source_id)
        )
        if len(source_ids) < 2 or any(not message.source_id for message in messages):
            evidence_id, evidence_key = self._archive_evidence(
                messages,
                sequence_timestamp=sequence_timestamp,
                observed_date=observed_date,
                created_at=created_at,
            )
            return (
                evidence_id,
                evidence_key,
                {observed_date: evidence_id} if observed_date else {},
                {
                    message.source_id: evidence_id
                    for message in messages
                    if message.source_id
                },
            )

        archived: list[tuple[str, str]] = []
        origins_by_date: dict[str, list[str]] = {}
        origins_by_source: dict[str, str] = {}
        for source_id in source_ids:
            source_messages = [
                message for message in messages if message.source_id == source_id
            ]
            source_dates = _session_observed_dates(source_messages)
            source_date = next(iter(source_dates)) if len(source_dates) == 1 else None
            evidence_id, evidence_key = self._archive_evidence(
                source_messages,
                sequence_timestamp=sequence_timestamp,
                observed_date=source_date,
                created_at=created_at,
            )
            archived.append((evidence_id, evidence_key))
            origins_by_source[source_id] = evidence_id
            if source_date:
                origins_by_date.setdefault(source_date, []).append(evidence_id)
        return (
            archived[0][0],
            archived[0][1],
            {
                source_date: evidence_ids[0]
                for source_date, evidence_ids in origins_by_date.items()
                if len(evidence_ids) == 1
            },
            origins_by_source,
        )

    def _next_full_number(self) -> int:
        prefix = self.layout.current_prefix()
        pattern = re.compile(
            rf"/CURRENT_{re.escape(str(self.instance_id))}_full_(\d+)\.md$"
        )
        numbers = [
            int(match.group(1))
            for key in self.repository.store.list_keys(prefix)
            if (match := pattern.search(key))
        ]
        return max(numbers, default=0) + 1


def _session_observed_dates(messages: list[ChatMessage]) -> set[str]:
    """Return every authoritative session date carried by the input."""

    embedded = {
        f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        for message in messages
        for match in re.finditer(
            r"\bSession date:\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b",
            message.content,
            re.IGNORECASE,
        )
    }
    provided = {
        message.observed_at.date().isoformat()
        if isinstance(message.observed_at, datetime)
        else message.observed_at.isoformat()
        for message in messages
        if message.observed_at is not None
    }
    return embedded | provided


def _render_direct_content(
    messages: list[ChatMessage],
    *,
    sequence_timestamp: int,
    evidence_id: str,
    origin_by_source: dict[str, str],
) -> str:
    """Render source-tagged batch input as lossless, indexable turn records.

    Ordinary ``infer=False`` add-memory calls retain their historical single
    ``source=add_memory`` record. Dataset/batch callers provide source
    coordinates, so each turn becomes an independent row with exact provenance
    and observation date, avoiding an expensive LLM paraphrase before indexing.
    """

    if not any(message.source_id or message.observed_at for message in messages):
        direct_content = "\n".join(message.content for message in messages).strip()
        return (
            f"- <seq={sequence_timestamp},source=add_memory> {direct_content}"
            if direct_content
            else ""
        )
    rows: list[str] = []
    for message in messages:
        content = " ".join(message.content.split())
        if not content:
            continue
        fields = [f"seq={sequence_timestamp}"]
        observed_dates = _session_observed_dates([message])
        if len(observed_dates) == 1:
            fields.append(f"observed={next(iter(observed_dates))}")
        origin = origin_by_source.get(message.source_id or "", evidence_id)
        fields.append(f"origin={origin}")
        if message.role == "assistant":
            fields.append("source=AI")
            text = f"The assistant said: {content}"
        else:
            text = f"The user said: {content}"
        rows.append(f"- <{','.join(fields)}> {text}")
    return "# Conversation turns\n\n" + "\n".join(rows) if rows else ""


def _semantic_memory_content(content: str) -> str:
    """Strip storage-only provenance fields for semantic token accounting."""

    return re.sub(r",(?:fid|origin|observed)=[^,>]+", "", content)


def _normalize_extraction_metadata(
    content: str,
    *,
    sequence_timestamp: int,
    observed_date: str | None = None,
    allowed_observed_dates: set[str] | None = None,
    origin_id: str | None = None,
    origin_by_observed: dict[str, str] | None = None,
) -> str:
    """Enforce extractor-owned Markdown at the LLM trust boundary.

    Models occasionally copy an example placeholder such as ``seq=0``, emit
    the obsolete ``time=empty`` sentinel, omit required bullet markers, or use
    nested heading levels. Normalize that syntax while leaving fact text untouched.
    """

    def normalize_tag(match: re.Match[str]) -> str:
        fields = [field.strip() for field in match.group(1).split(",")]
        normalized_fields: list[str] = []
        saw_sequence = False
        saw_observed = False
        saw_origin = False
        fact_observed = ""
        for field in fields:
            name, separator, value = field.partition("=")
            normalized_name = name.strip().lower()
            if normalized_name == "seq" and separator:
                normalized_fields.append(f"seq={sequence_timestamp}")
                saw_sequence = True
                continue
            if normalized_name == "observed" and separator:
                if observed_date:
                    normalized_fields.append(f"observed={observed_date}")
                    saw_observed = True
                elif value.strip() in (allowed_observed_dates or set()):
                    fact_observed = value.strip()
                    normalized_fields.append(f"observed={fact_observed}")
                    saw_observed = True
                continue
            if normalized_name == "origin" and separator:
                if origin_id:
                    normalized_fields.append(f"origin={origin_id}")
                    saw_origin = True
                continue
            if (
                normalized_name == "time"
                and separator
                and value.strip().lower() in {"", "empty", "none", "null"}
            ):
                continue
            normalized_fields.append(field)
        if not saw_sequence:
            return match.group(0)
        if observed_date and not saw_observed:
            normalized_fields.insert(1, f"observed={observed_date}")
        selected_origin = (origin_by_observed or {}).get(fact_observed, origin_id)
        if selected_origin and not saw_origin:
            normalized_fields.insert(1, f"origin={selected_origin}")
        return f"<{','.join(normalized_fields)}>"

    normalized = normalize_headings_to_h1(content)
    normalized = re.sub(
        r"<([^<>]*\bseq\s*=\s*[^<>]+)>", normalize_tag, normalized
    )
    return _normalize_fact_bullets(normalized)


def _normalize_fact_bullets(content: str) -> str:
    """Make every top-level metadata-prefixed fact a Markdown bullet.

    Real models occasionally omit the list marker even when the prompt requires
    it. Normalizing only lines that begin with a ``seq`` metadata tag preserves
    headings, continuations, nested lists, and fenced code verbatim.
    """

    normalized_lines: list[str] = []
    fence_marker: str | None = None
    for line in content.splitlines():
        stripped = line.lstrip()
        marker_match = re.match(r"(`{3,}|~{3,})", stripped)
        if marker_match:
            marker = marker_match.group(1)
            if fence_marker is None:
                fence_marker = marker[0]
            elif marker.startswith(fence_marker):
                fence_marker = None
            normalized_lines.append(line)
            continue
        if fence_marker is None and re.match(r"<[^<>]*\bseq\s*=\s*[^<>]+>", stripped):
            indentation = line[: len(line) - len(stripped)]
            line = f"{indentation}- {stripped}"
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


__all__ = ["MemoryExtractor"]
