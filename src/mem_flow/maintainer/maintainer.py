"""CronJob-friendly hierarchical memory maintenance flow."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..config import MaintenanceConfig
from ..hierarchy import (
    DirectoryRouter,
    DirectoryTopicBuilder,
    MarkdownFact,
    PersistedRoutePlan,
    deterministic_document_summary,
    directory_content_digest,
    normalize_document_summary,
)
from ..index import FactIndexStore
from ..llm import FlowLLM, complete_structured
from ..models import (
    ChatMessage,
    DocumentMetadata,
    LLMRequest,
    LegacyMigrationItem,
    LegacyMigrationResult,
    MaintenanceRequest,
    MaintenanceResult,
    MemoryDocument,
    as_beijing_time,
    beijing_now,
    compact_beijing_timestamp,
)
from ..observability import FlowMetrics, observe_flow
from ..prompts import REWRITE_PROMPT
from ..repository import MemoryRepository
from ..utils.codec import strip_yaml_front_matter
from ..utils.markdown import normalize_headings_to_h1
from ..utils.parsing import parse_json_model
from ..utils.paths import KeyLayout
from ..utils.tokens import estimate_tokens

MaintenanceProgress = Callable[[str, dict[str, object]], None]


class _TopicReferences(BaseModel):
    title: str
    fact_ids: list[str] = Field(min_length=1)


class _DuplicateReference(BaseModel):
    discarded_id: str
    retained_id: str


class _ContentReferencePlan(BaseModel):
    topics: list[_TopicReferences] = Field(default_factory=list)
    duplicates: list[_DuplicateReference] = Field(default_factory=list)


class MemoryMaintainer(BaseModel):
    """Consume working objects and append immutable leaves to topic directories."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repository: MemoryRepository
    layout: KeyLayout
    llm: FlowLLM
    metrics: FlowMetrics
    config: MaintenanceConfig
    index_store: FactIndexStore

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.maintainer")

    def maintain(
        self,
        request: MaintenanceRequest,
        *,
        progress: MaintenanceProgress | None = None,
    ) -> MaintenanceResult:
        context = {
            "store_id": self.repository.scope.store_id,
            "user_id": self.repository.scope.user_id,
        }
        if request.merge_topics is not None:
            self.logger.warning(
                "merge_topics_ignored value=%s hierarchical leaf documents are immutable",
                request.merge_topics,
            )
        with observe_flow(
            self.metrics,
            self.logger,
            module="maintainer",
            flow="maintain",
            context=context,
        ):
            _notify(progress, "scan_current", "running")
            current_documents = _order_documents_by_creation(
                self.repository.list_current()
            )
            # RAW is the durable hand-off between online extraction and offline
            # maintenance. A previous run may have archived and removed CURRENT,
            # then failed during an LLM stage; include those RAW objects on retry.
            raw_documents = self.repository.list_raw()
            existing_rewrites = self.repository.list_rewrites()
            result = MaintenanceResult(current_documents=len(current_documents))
            _notify(
                progress,
                "scan_current",
                "completed",
                current_documents=len(current_documents),
                recoverable_raw_documents=len(raw_documents),
                recoverable_rewrite_documents=len(existing_rewrites),
            )

            for index, current in enumerate(current_documents, start=1):
                details = {
                    "current": index,
                    "total": len(current_documents),
                    "current_id": current.metadata.id,
                }
                _notify(progress, "archive_raw", "running", **details)
                raw = self._archive_raw(request, current)
                raw_documents.append(raw)
                self.repository.store.delete(current.key)
                result.deleted_current_keys.append(current.key)
                self.metrics.documents_total.labels(
                    module="maintainer", flow="maintain", kind="current_consumed"
                ).inc()
                self.logger.info(
                    "current_deleted_after_raw current_id=%s raw_id=%s",
                    current.metadata.id,
                    raw.metadata.id,
                )
                _notify(
                    progress,
                    "archive_raw",
                    "completed",
                    **details,
                    raw_id=raw.metadata.id,
                    current_key=current.key,
                    deleted_current=1,
                )

            raw_documents = sorted(raw_documents, key=lambda item: item.key)
            result.raw_documents = len(raw_documents)
            rewrites_by_id = {item.metadata.id: item for item in existing_rewrites}
            rewrite_batches = _batch_documents_by_tokens(
                raw_documents,
                max_tokens=self.config.rewrite_batch_max_tokens,
            )
            rewrite_details: list[dict[str, Any]] = []
            for index, (batch, batch_tokens) in enumerate(rewrite_batches, start=1):
                details = {
                    "current": index,
                    "total": len(rewrite_batches),
                    "source_documents": len(batch),
                    "source_tokens": batch_tokens,
                    "raw_ids": [item.metadata.id for item in batch],
                }
                rewrite_details.append(details)
                _notify(progress, "rewrite_current", "running", **details)

            def rewrite_batch(
                item: tuple[list[MemoryDocument], int],
            ) -> MemoryDocument:
                batch, batch_tokens = item
                return self._rewrite(request, batch, batch_tokens=batch_tokens)

            if request.workers == 1 or len(rewrite_batches) < 2:
                completed_rewrites = [rewrite_batch(item) for item in rewrite_batches]
            else:
                workers = min(request.workers, len(rewrite_batches))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    completed_rewrites = list(
                        executor.map(rewrite_batch, rewrite_batches)
                    )

            for details, rewrite in zip(
                rewrite_details, completed_rewrites, strict=True
            ):
                rewrites_by_id[rewrite.metadata.id] = rewrite
                _notify(
                    progress,
                    "rewrite_current",
                    "completed",
                    **details,
                    rewrite_id=rewrite.metadata.id,
                )

            rewrites = sorted(rewrites_by_id.values(), key=lambda item: item.key)
            result.rewrite_documents = len(rewrites)
            repaired = self._repair_directory_topics(request, progress)
            result.directory_topics_updated += repaired
            if not rewrites:
                self.index_store.rebuild(self.repository.list_memory_documents())
                self.logger.info("maintenance_no_sources context=%s", context)
                return result

            known_directories = set(self.repository.list_directory_ids())
            for index, rewrite in enumerate(rewrites, start=1):
                facts = _collect_facts([rewrite])
                details = {
                    "current": index,
                    "total": len(rewrites),
                    "rewrite_id": rewrite.metadata.id,
                }
                _notify(progress, "route_directories", "running", **details)
                with observe_flow(
                    self.metrics,
                    self.logger,
                    module="maintainer",
                    flow="route_directories",
                    context={"rewrite_id": rewrite.metadata.id},
                ):
                    plan, route_reused = self._load_or_create_route(
                        request, rewrite, facts
                    )
                new_directories = {
                    item.directory_id
                    for item in plan.assignments
                    if item.directory_id not in known_directories
                }
                result.directories_created += len(new_directories)
                if new_directories:
                    self.metrics.documents_total.labels(
                        module="maintainer",
                        flow="route_directories",
                        kind="directory_created",
                    ).inc(len(new_directories))
                known_directories.update(new_directories)
                _notify(
                    progress,
                    "route_directories",
                    "completed",
                    **details,
                    destinations=len(plan.assignments),
                    directories_created=len(new_directories),
                    route_reused=route_reused,
                    route_key=self.layout.route_key(rewrite.metadata.id),
                )

                _notify(progress, "write_documents", "running", **details)
                with observe_flow(
                    self.metrics,
                    self.logger,
                    module="maintainer",
                    flow="write_documents",
                    context={"rewrite_id": rewrite.metadata.id},
                ):
                    created, affected = self._write_memory_documents(
                        request, rewrite, facts, plan
                    )
                result.memory_documents_created += created
                _notify(
                    progress,
                    "write_documents",
                    "completed",
                    **details,
                    memory_documents_created=created,
                    directory_ids=list(affected),
                )

                for summary_index, (directory_id, title) in enumerate(
                    affected.items(), start=1
                ):
                    summary_details = {
                        "current": summary_index,
                        "total": len(affected),
                        "directory_id": directory_id,
                        "rewrite_id": rewrite.metadata.id,
                    }
                    _notify(progress, "refresh_topics", "running", **summary_details)
                    with observe_flow(
                        self.metrics,
                        self.logger,
                        module="maintainer",
                        flow="refresh_topics",
                        context={"directory_id": directory_id},
                    ):
                        changed, document_count = self._refresh_directory_topic(
                            request, directory_id, title
                        )
                    result.directory_topics_updated += int(changed)
                    _notify(
                        progress,
                        "refresh_topics",
                        "completed",
                        **summary_details,
                        topic_updated=changed,
                        document_count=document_count,
                    )

            _notify(progress, "publish_index", "running")
            snapshot = self.index_store.rebuild(
                self.repository.list_memory_documents()
            )
            _notify(
                progress,
                "publish_index",
                "completed",
                facts=len(snapshot.facts),
                relations=len(snapshot.relations),
                manifest=(snapshot.manifest.digest if snapshot.manifest else ""),
            )
            self._cleanup_intermediates(request, result, progress)
            self.logger.info("maintenance_result result=%s", result.model_dump())
            return result

    def migrate_legacy_topics(
        self,
        *,
        dry_run: bool = True,
        delete_source: bool = False,
    ) -> LegacyMigrationResult:
        """Copy flat doc/topic files into directories, optionally deleting sources."""
        legacy_documents = self.repository.list_legacy_topics()
        result = LegacyMigrationResult(scanned=len(legacy_documents))
        with observe_flow(
            self.metrics,
            self.logger,
            module="maintainer",
            flow="migrate_legacy_topics",
            context={"documents": len(legacy_documents), "dry_run": dry_run},
        ):
            directories = self.repository.list_directory_topics()
            for legacy in legacy_documents:
                title = legacy.metadata.title or "Memory"
                fact = MarkdownFact(
                    id="f0001",
                    source_id=legacy.metadata.id,
                    heading=title,
                    markdown=legacy.content,
                )
                plan = DirectoryRouter(llm=self.llm).create_plan(
                    rewrite_id=f"MIGRATE_{legacy.metadata.id}",
                    facts=[fact],
                    directories=directories,
                    summary_tokens=self.config.document_summary_tokens,
                )
                assignment = plan.assignments[0]
                target_key = self.layout.memory_key(
                    assignment.directory_id, legacy.metadata.id
                )
                action = "planned"
                if not dry_run:
                    if self.repository.store.exists(target_key):
                        target = self.repository.read(target_key)
                        if target.content.strip() != legacy.content.strip():
                            raise ValueError(
                                f"migration target has different content: {target_key}"
                            )
                        action = "existing"
                    else:
                        summary = legacy.metadata.summary
                        if not summary:
                            summary = (
                                normalize_document_summary(
                                    assignment.summary,
                                    summary_tokens=(
                                        self.config.document_summary_tokens
                                    ),
                                )
                                if assignment.summary
                                else deterministic_document_summary(
                                    legacy.content,
                                    summary_tokens=(
                                        self.config.document_summary_tokens
                                    ),
                                )
                            )
                        target = MemoryDocument(
                            metadata=DocumentMetadata(
                                id=legacy.metadata.id,
                                kind="memory",
                                directory_id=assignment.directory_id,
                                title=legacy.metadata.title
                                or assignment.directory_title,
                                summary=summary,
                                created_at=legacy.metadata.created_at,
                                updated_at=legacy.metadata.updated_at,
                                source_ids=legacy.metadata.source_ids,
                            ),
                            content=legacy.content,
                            key=target_key,
                        )
                        self.repository.write(target_key, target)
                        result.copied += 1
                        action = "copied"
                    changed, _ = self._refresh_directory_topic(
                        MaintenanceRequest(),
                        assignment.directory_id,
                        assignment.directory_title,
                    )
                    result.directory_topics_updated += int(changed)
                    directories = self.repository.list_directory_topics()
                    if delete_source:
                        self.repository.store.delete(legacy.key)
                        result.deleted_sources += 1
                result.items.append(
                    LegacyMigrationItem(
                        document_id=legacy.metadata.id,
                        source_key=legacy.key,
                        directory_id=assignment.directory_id,
                        target_key=target_key,
                        action=action,
                    )
                )
        if not dry_run:
            self.index_store.rebuild(self.repository.list_memory_documents())
        return result

    def _repair_directory_topics(
        self,
        request: MaintenanceRequest,
        progress: MaintenanceProgress | None,
    ) -> int:
        topics = {
            item.metadata.id: item for item in self.repository.list_directory_topics()
        }
        repaired = 0
        directory_ids = self.repository.list_directory_ids()
        for index, directory_id in enumerate(directory_ids, start=1):
            documents = self.repository.list_memory_documents(directory_id)
            if not documents:
                continue
            previous = topics.get(directory_id)
            title = (
                previous.metadata.title
                if previous
                else documents[0].metadata.title or "Memory"
            )
            topic_key = self.layout.directory_topic_key(directory_id)
            legacy_key = self.layout.legacy_directory_summary_key(directory_id)
            topic = DirectoryTopicBuilder().build(
                directory_title=title,
                document_contents=[document.content for document in documents],
                existing_content=(
                    previous.content if previous and previous.key == topic_key else ""
                ),
            )
            if previous and _topic_is_current(
                previous,
                key=topic_key,
                digest=directory_content_digest(documents),
                document_count=len(documents),
                title=topic.title,
                summary=topic.summary,
                body=topic.body,
                legacy_exists=self.repository.store.exists(legacy_key),
            ):
                continue
            details = {
                "current": index,
                "total": len(directory_ids),
                "directory_id": directory_id,
                "repair": True,
            }
            _notify(progress, "refresh_topics", "running", **details)
            changed, document_count = self._refresh_directory_topic(
                request, directory_id, title
            )
            repaired += int(changed)
            _notify(
                progress,
                "refresh_topics",
                "completed",
                **details,
                topic_updated=changed,
                document_count=document_count,
            )
        return repaired

    def _archive_raw(
        self, request: MaintenanceRequest, current: MemoryDocument
    ) -> MemoryDocument:
        with observe_flow(
            self.metrics,
            self.logger,
            module="maintainer",
            flow="archive_raw",
            context={"current_id": current.metadata.id},
        ):
            timestamp = compact_beijing_timestamp(microseconds=True)
            raw_id = f"RAW_{current.metadata.id}_{timestamp}_{uuid4().hex[:8]}"
            raw = MemoryDocument(
                metadata=DocumentMetadata(
                    id=raw_id,
                    kind="raw",
                    # A RAW document is the durable form of this CURRENT snapshot.
                    # Keep the source creation time so retry runs retain the same
                    # oldest-first rewrite priority.
                    created_at=current.metadata.created_at,
                    source_ids=[current.metadata.id],
                ),
                content=current.content,
            )
            raw.key = self.layout.raw_key(raw_id)
            self.repository.write(raw.key, raw)
            self.metrics.documents_total.labels(
                module="maintainer", flow="archive_raw", kind="raw"
            ).inc()
            return raw

    def _rewrite(
        self,
        request: MaintenanceRequest,
        raw_documents: list[MemoryDocument],
        *,
        batch_tokens: int,
    ) -> MemoryDocument:
        raw_ids = [item.metadata.id for item in raw_documents]
        digest_input = "\0".join(
            f"{item.metadata.id}:{hashlib.sha256(item.content.encode()).hexdigest()}"
            for item in raw_documents
        )
        rewrite_id = (
            f"REWRITE_CURRENT_{hashlib.sha256(digest_input.encode()).hexdigest()[:24]}"
        )
        key = f"{self.layout.rewrite_prefix()}/{rewrite_id}.md"
        if self.repository.store.exists(key):
            self.logger.info("rewrite_reused rewrite_id=%s key=%s", rewrite_id, key)
            return self.repository.read(key)
        with observe_flow(
            self.metrics,
            self.logger,
            module="maintainer",
            flow="rewrite_current",
            context={"source_count": len(raw_documents)},
        ):
            facts = _collect_facts(raw_documents)
            if self.config.deterministic:
                content = _render_memory_headings(facts, "Memory")
            else:
                rewrite_request = LLMRequest(
                    operation="rewrite_current",
                    messages=[
                        ChatMessage(
                            role="user",
                            content=REWRITE_PROMPT.format(facts=_facts_json(facts)),
                        )
                    ],
                )

                def validate(raw: str) -> str:
                    plan = parse_json_model(raw, _ContentReferencePlan)
                    return _render_content_plan(plan, facts)

                try:
                    content = complete_structured(self.llm, rewrite_request, validate)
                except (ValueError, RuntimeError):
                    # A malformed reference plan must not discard an entire user's
                    # memory batch. After structured retries, retain every fact.
                    self.logger.warning(
                        "rewrite_plan_fallback rewrite_id=%s facts=%d",
                        rewrite_id,
                        len(facts),
                    )
                    content = _render_memory_headings(facts, "Memory")
            rewrite = MemoryDocument(
                metadata=DocumentMetadata(
                    id=rewrite_id,
                    kind="rewrite",
                    source_ids=raw_ids,
                ),
                content=content,
                key=key,
            )
            self.repository.write(key, rewrite)
            self.metrics.documents_total.labels(
                module="maintainer", flow="rewrite_current", kind="rewrite"
            ).inc()
            self.logger.info(
                "rewrite_written raw_ids=%s source_tokens=%d rewrite_id=%s",
                raw_ids,
                batch_tokens,
                rewrite_id,
            )
            return rewrite

    def _load_or_create_route(
        self,
        request: MaintenanceRequest,
        rewrite: MemoryDocument,
        facts: list[MarkdownFact],
    ) -> tuple[PersistedRoutePlan, bool]:
        route_key = self.layout.route_key(rewrite.metadata.id)
        if self.repository.store.exists(route_key):
            plan = PersistedRoutePlan.model_validate_json(
                self.repository.store.get_text(route_key)
            )
            _validate_persisted_route(plan, rewrite.metadata.id, facts)
            self.logger.info(
                "route_plan_reused rewrite_id=%s route_key=%s",
                rewrite.metadata.id,
                route_key,
            )
            return plan, True
        directories = self.repository.list_directory_topics()
        self.logger.info(
            "directory_catalog_loaded directory_count=%d rewrite_id=%s",
            len(directories),
            rewrite.metadata.id,
        )
        router = DirectoryRouter(llm=self.llm)
        plan = (
            router.create_deterministic_plan(
                rewrite_id=rewrite.metadata.id,
                facts=facts,
                directories=directories,
                summary_tokens=self.config.document_summary_tokens,
            )
            if self.config.deterministic
            else router.create_plan(
                rewrite_id=rewrite.metadata.id,
                facts=facts,
                directories=directories,
                summary_tokens=self.config.document_summary_tokens,
            )
        )
        self.repository.store.put_text(route_key, plan.model_dump_json(indent=2))
        return plan, False

    def _write_memory_documents(
        self,
        request: MaintenanceRequest,
        rewrite: MemoryDocument,
        facts: list[MarkdownFact],
        plan: PersistedRoutePlan,
    ) -> tuple[int, dict[str, str]]:
        facts_by_id = {fact.id: fact for fact in facts}
        created = 0
        affected: dict[str, str] = {}
        pending: list[tuple[Any, str, str]] = []
        for assignment in plan.assignments:
            selected = [facts_by_id[item] for item in assignment.fact_ids]
            content = _render_memory_headings(selected, assignment.directory_title)
            key = self.layout.memory_key(
                assignment.directory_id,
                assignment.document_id,
            )
            if self.repository.store.exists(key):
                existing = self.repository.read(key)
                if existing.content.strip() != content.strip():
                    raise ValueError(
                        f"deterministic memory key has different content: {key}"
                    )
                self.logger.info(
                    "memory_document_reused directory_id=%s document_id=%s",
                    assignment.directory_id,
                    assignment.document_id,
                )
            else:
                pending.append((assignment, content, key))
            affected[assignment.directory_id] = assignment.directory_title

        for assignment, content, key in pending:
            summary = (
                normalize_document_summary(
                    assignment.summary,
                    summary_tokens=self.config.document_summary_tokens,
                )
                if assignment.summary
                else deterministic_document_summary(
                    content,
                    summary_tokens=self.config.document_summary_tokens,
                )
            )
            document = MemoryDocument(
                metadata=DocumentMetadata(
                    id=assignment.document_id,
                    kind="memory",
                    directory_id=assignment.directory_id,
                    title=assignment.directory_title,
                    summary=summary,
                    source_ids=[rewrite.metadata.id],
                ),
                content=content,
                key=key,
            )
            self.repository.write(key, document)
            created += 1
            self.metrics.documents_total.labels(
                module="maintainer", flow="write_documents", kind="memory_created"
            ).inc()
            self.logger.info(
                "memory_document_written directory_id=%s document_id=%s bytes=%d",
                assignment.directory_id,
                assignment.document_id,
                len(content.encode()),
            )
        return created, affected

    def _refresh_directory_topic(
        self,
        request: MaintenanceRequest,
        directory_id: str,
        title: str,
    ) -> tuple[bool, int]:
        documents = self.repository.list_memory_documents(directory_id)
        digest = directory_content_digest(documents)
        key = self.layout.directory_topic_key(directory_id)
        legacy_key = self.layout.legacy_directory_summary_key(directory_id)
        previous: MemoryDocument | None = None
        previous_topic_content = ""
        if self.repository.store.exists(key):
            previous = self.repository.read(key)
            previous_topic_content = previous.content
            title = previous.metadata.title or title
        elif self.repository.store.exists(legacy_key):
            previous = self.repository.read(legacy_key)
            title = previous.metadata.title or title
        topic = DirectoryTopicBuilder().build(
            directory_title=title,
            document_contents=[document.content for document in documents],
            existing_content=previous_topic_content,
        )
        if previous and _topic_is_current(
            previous,
            key=key,
            digest=digest,
            document_count=len(documents),
            title=topic.title,
            summary=topic.summary,
            body=topic.body,
            legacy_exists=self.repository.store.exists(legacy_key),
        ):
            self.logger.info(
                "directory_topic_unchanged directory_id=%s documents=%d headings=%d",
                directory_id,
                len(documents),
                len(topic.headings),
            )
            return False, len(documents)
        now = beijing_now()
        document = MemoryDocument(
            metadata=DocumentMetadata(
                id=directory_id,
                kind="directory_topic",
                title=topic.title,
                summary=topic.summary,
                document_count=len(documents),
                content_digest=digest,
                created_at=previous.metadata.created_at if previous else now,
                updated_at=now,
            ),
            content=topic.body,
            key=key,
        )
        self.repository.write(key, document)
        if self.repository.store.exists(legacy_key):
            self.repository.store.delete(legacy_key)
            self.logger.info(
                "legacy_directory_summary_deleted directory_id=%s key=%s",
                directory_id,
                legacy_key,
            )
        self.metrics.documents_total.labels(
            module="maintainer", flow="refresh_topics", kind="topic_updated"
        ).inc()
        self.logger.info(
            "directory_topic_refreshed directory_id=%s document_count=%d "
            "heading_count=%d digest=%s",
            directory_id,
            len(documents),
            len(topic.headings),
            digest,
        )
        return True, len(documents)

    def _cleanup_intermediates(
        self,
        request: MaintenanceRequest,
        result: MaintenanceResult,
        progress: MaintenanceProgress | None,
    ) -> None:
        raw_keys = self.repository.store.list_keys(self.layout.raw_prefix())
        rewrite_keys = self.repository.store.list_keys(self.layout.rewrite_prefix())
        _notify(
            progress,
            "cleanup_intermediates",
            "running",
            total=len(raw_keys) + len(rewrite_keys),
        )
        for key in raw_keys:
            self.repository.store.delete(key)
            result.deleted_raw_keys.append(key)
        for key in rewrite_keys:
            self.repository.store.delete(key)
            if key.endswith(".json"):
                result.deleted_route_keys.append(key)
            else:
                result.deleted_rewrite_keys.append(key)
        _notify(
            progress,
            "cleanup_intermediates",
            "completed",
            archived_current=len(result.deleted_current_keys),
            deleted_raw=len(result.deleted_raw_keys),
            deleted_rewrite=len(result.deleted_rewrite_keys),
            deleted_route=len(result.deleted_route_keys),
        )


def _batch_documents_by_tokens(
    documents: list[MemoryDocument], *, max_tokens: int
) -> list[tuple[list[MemoryDocument], int]]:
    """Greedily group oldest documents first without splitting a snapshot."""

    batches: list[tuple[list[MemoryDocument], int]] = []
    current: list[MemoryDocument] = []
    tokens = 0
    for document in _order_documents_by_creation(documents):
        document_tokens = estimate_tokens(document.content)
        if current and tokens + document_tokens > max_tokens:
            batches.append((current, tokens))
            current = []
            tokens = 0
        current.append(document)
        tokens += document_tokens
    if current:
        batches.append((current, tokens))
    return batches


def _order_documents_by_creation(
    documents: list[MemoryDocument],
) -> list[MemoryDocument]:
    """Return a stable oldest-first order independent of object-store key order."""

    def sort_key(document: MemoryDocument) -> tuple[datetime, str, str]:
        created_at = document.metadata.created_at
        return as_beijing_time(created_at), document.metadata.id, document.key

    return sorted(documents, key=sort_key)


def _collect_facts(documents: list[MemoryDocument]) -> list[MarkdownFact]:
    facts: list[MarkdownFact] = []
    for document in documents:
        for heading, markdown in _markdown_blocks(
            strip_yaml_front_matter(document.content)
        ):
            facts.append(
                MarkdownFact(
                    id=f"f{len(facts) + 1:04d}",
                    source_id=document.metadata.id,
                    heading=heading,
                    markdown=markdown,
                )
            )
    if not facts:
        raise ValueError("memory content contains no Markdown facts")
    return facts


def _markdown_blocks(content: str) -> list[tuple[str, str]]:
    content = normalize_headings_to_h1(content)
    blocks: list[tuple[str, str]] = []
    heading = "Memory"
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        markdown = "\n".join(current).strip()
        if markdown:
            blocks.append((heading, markdown))
        current.clear()

    for line in content.strip().splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        heading_match = None if in_fence else re.match(r"^#\s+(.+?)\s*$", line)
        if heading_match:
            flush()
            heading = heading_match.group(1)
            continue
        if not in_fence and (
            re.match(r"^(?:[-+*]|\d+[.)])\s+", line)
            or re.match(r"^\s*<[^<>]*\bseq\s*=\s*[^<>]+>", line)
        ):
            flush()
        current.append(line)
    flush()
    return blocks


def _facts_json(facts: list[MarkdownFact]) -> str:
    return json.dumps([fact.model_dump() for fact in facts], ensure_ascii=False)


def _validate_content_plan(
    plan: _ContentReferencePlan, facts: list[MarkdownFact]
) -> None:
    allowed = {fact.id for fact in facts}
    kept = [fact_id for topic in plan.topics for fact_id in topic.fact_ids]
    discarded = [item.discarded_id for item in plan.duplicates]
    retained = [item.retained_id for item in plan.duplicates]
    if (
        not plan.topics
        or any(not topic.fact_ids for topic in plan.topics)
        or len(kept) != len(set(kept))
        or len(discarded) != len(set(discarded))
        or set(kept) & set(discarded)
        or set(kept) | set(discarded) != allowed
        or not set(retained) <= set(kept)
    ):
        raise ValueError("invalid content reference plan")


_STATE_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|once|twice|thrice)(?!\w)",
    re.IGNORECASE,
)
_STATE_MARKER_RE = re.compile(
    r"\b(?:initial(?:ly)?|previous(?:ly)?|formerly|used to|now|current(?:ly)?|"
    r"changed?|increased?|decreased?|switched?|new personal best)\b",
    re.IGNORECASE,
)


def _preserve_distinct_state_duplicates(
    plan: _ContentReferencePlan, facts: list[MarkdownFact]
) -> _ContentReferencePlan:
    """Restore discarded facts that carry a distinct value or state transition.

    A different ``time=`` alone is deliberately not sufficient.  Separate extraction
    batches can resolve the same source-relative phrase (for example, "last Thursday")
    against different session dates.  Treating those inferred dates as authoritative
    turns one retold event into multiple occurrences.  The rewrite prompt still tells
    the model to retain genuinely separate dated events; this repair is only a guard
    against losing value changes and explicit old/new state history.
    """

    by_id = {fact.id: fact for fact in facts}
    fact_order = {fact.id: index for index, fact in enumerate(facts)}
    topic_ids = [list(topic.fact_ids) for topic in plan.topics]
    topic_by_fact = {
        fact_id: index
        for index, fact_ids in enumerate(topic_ids)
        for fact_id in fact_ids
    }
    duplicates: list[_DuplicateReference] = []

    def state_signature(markdown: str) -> tuple[tuple[str, ...], tuple[str, ...], bool, str]:
        first_line = markdown.splitlines()[0] if markdown else ""
        times = tuple(
            value.strip().casefold()
            for value in re.findall(r"(?:^|,)time=([^,>]+)", first_line)
        )
        body = re.sub(r"^\s*-?\s*<[^>]+>\s*", "", markdown).strip().casefold()
        values = tuple(match.group(0).casefold() for match in _STATE_NUMBER_RE.finditer(body))
        return times, values, bool(_STATE_MARKER_RE.search(body)), body

    for duplicate in plan.duplicates:
        discarded = by_id.get(duplicate.discarded_id)
        retained = by_id.get(duplicate.retained_id)
        topic_index = topic_by_fact.get(duplicate.retained_id)
        if discarded is None or retained is None or topic_index is None:
            duplicates.append(duplicate)
            continue
        discarded_sig = state_signature(discarded.markdown)
        retained_sig = state_signature(retained.markdown)
        distinct = discarded_sig[3] != retained_sig[3] and (
            discarded_sig[1]
            and retained_sig[1]
            and discarded_sig[1] != retained_sig[1]
            or discarded_sig[2]
            or retained_sig[2]
        )
        if not distinct:
            duplicates.append(duplicate)
            continue
        topic_ids[topic_index].append(duplicate.discarded_id)
        topic_by_fact[duplicate.discarded_id] = topic_index

    topics = [
        topic.model_copy(
            update={"fact_ids": sorted(topic_ids[index], key=fact_order.__getitem__)}
        )
        for index, topic in enumerate(plan.topics)
    ]
    return plan.model_copy(update={"topics": topics, "duplicates": duplicates})


def _render_content_plan(plan: _ContentReferencePlan, facts: list[MarkdownFact]) -> str:
    # Reject unknown/missing ids before the state-history repair needs to sort them.
    _validate_content_plan(plan, facts)
    plan = _preserve_distinct_state_duplicates(plan, facts)
    _validate_content_plan(plan, facts)
    by_id = {fact.id: fact for fact in facts}
    return "\n\n".join(
        _render_memory(topic.title, [by_id[item] for item in topic.fact_ids])
        for topic in plan.topics
    )


def _render_memory(title: str, facts: list[MarkdownFact]) -> str:
    return f"# {_clean_title(title)}\n\n" + "\n".join(fact.markdown for fact in facts)


def _render_memory_headings(facts: list[MarkdownFact], fallback_title: str) -> str:
    """Render routed facts under their rewrite H1 headings, preserving order."""

    grouped: dict[str, list[MarkdownFact]] = {}
    for fact in facts:
        heading = _clean_title(fact.heading or fallback_title)
        grouped.setdefault(heading, []).append(fact)
    return "\n\n".join(
        _render_memory(heading, grouped_facts)
        for heading, grouped_facts in grouped.items()
    )


def _clean_title(title: str) -> str:
    return " ".join(re.sub(r"^#+\s*", "", title).split()) or "Memory"


def _validate_persisted_route(
    plan: PersistedRoutePlan,
    rewrite_id: str,
    facts: list[MarkdownFact],
) -> None:
    assigned = [item for assignment in plan.assignments for item in assignment.fact_ids]
    allowed = {fact.id for fact in facts}
    if (
        plan.rewrite_id != rewrite_id
        or len(assigned) != len(set(assigned))
        or set(assigned) != allowed
        or len({item.directory_id for item in plan.assignments})
        != len(plan.assignments)
    ):
        raise ValueError("invalid persisted route plan")


def _topic_is_current(
    document: MemoryDocument,
    *,
    key: str,
    digest: str,
    document_count: int,
    title: str,
    summary: str,
    body: str,
    legacy_exists: bool,
) -> bool:
    return (
        document.key == key
        and document.metadata.content_digest == digest
        and document.metadata.document_count == document_count
        and document.metadata.title == title
        and document.metadata.summary == summary
        and document.content.strip() == body
        and not legacy_exists
    )


def _notify(
    progress: MaintenanceProgress | None,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    if progress is not None:
        progress(stage, {"status": status, **details})


__all__ = ["MaintenanceProgress", "MemoryMaintainer"]
