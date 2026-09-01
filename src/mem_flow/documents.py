"""Read and edit maintained doc memory leaves within one bound scope."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from .hierarchy import DirectoryTopicBuilder, directory_content_digest
from .index import FactIndexStore
from .models import (
    DocDocument,
    DocLineUpdateRequest,
    DocLineUpdateResult,
    DocReadRequest,
    DocumentMetadata,
    MemoryDocument,
    beijing_now,
)
from .observability import FlowMetrics, observe_flow
from .repository import MemoryRepository
from .utils.paths import KeyLayout


class DocDocumentManager(BaseModel):
    """Body-only CRUD subset for immutable-by-default maintained memory leaves.

    The manager resolves ids through the instance-scoped repository. Callers cannot
    supply an S3 key, directory id, store id, or user id, so edits cannot escape the
    namespace selected when ``MemFlow`` was initialized.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repository: MemoryRepository
    layout: KeyLayout
    metrics: FlowMetrics
    index_store: FactIndexStore

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.documents")

    def get(self, request: DocReadRequest) -> DocDocument:
        context = {"document_id": request.document_id}
        with observe_flow(
            self.metrics,
            self.logger,
            module="documents",
            flow="get_doc",
            context=context,
        ):
            document = self._find_memory(request.document_id)
            result = _to_doc_document(document)
            self.metrics.documents_total.labels(
                module="documents", flow="get_doc", kind="memory_read"
            ).inc()
            self.logger.info(
                "doc_document_read document_id=%s directory_id=%s key=%s lines=%d",
                document.metadata.id,
                document.metadata.directory_id,
                document.key,
                result.line_count,
            )
            return result

    def update_lines(self, request: DocLineUpdateRequest) -> DocLineUpdateResult:
        context = {
            "document_id": request.document_id,
            "start_line": request.start_line,
            "end_line": request.end_line,
        }
        with observe_flow(
            self.metrics,
            self.logger,
            module="documents",
            flow="update_doc_lines",
            context=context,
        ):
            document = self._find_memory(request.document_id)
            lines = document.content.splitlines()
            if request.end_line > len(lines):
                raise ValueError(
                    "line range exceeds document body: "
                    f"end_line={request.end_line}, line_count={len(lines)}"
                )

            replacement_lines = request.replacement.splitlines()
            new_lines = [
                *lines[: request.start_line - 1],
                *replacement_lines,
                *lines[request.end_line :],
            ]
            new_content = "\n".join(new_lines).strip()
            if not new_content:
                raise ValueError("doc memory document content cannot be empty")

            changed = new_content != document.content.strip()
            if changed:
                document.content = new_content
                self.repository.write_body(document.key, new_content)
                self._refresh_directory_topic(document.metadata.directory_id)
                self.index_store.rebuild(self.repository.list_memory_documents())
                self.metrics.documents_total.labels(
                    module="documents",
                    flow="update_doc_lines",
                    kind="memory_updated",
                ).inc()

            result = DocLineUpdateResult(
                document=_to_doc_document(document),
                start_line=request.start_line,
                end_line=request.end_line,
                replacement_line_count=len(replacement_lines),
                changed=changed,
            )
            self.logger.info(
                "doc_document_lines_updated document_id=%s directory_id=%s "
                "key=%s start_line=%d end_line=%d replacement_lines=%d changed=%s",
                document.metadata.id,
                document.metadata.directory_id,
                document.key,
                request.start_line,
                request.end_line,
                len(replacement_lines),
                changed,
            )
            return result

    def delete(self, request: DocReadRequest) -> None:
        """Delete one maintained memory leaf and refresh its derived topic."""

        context = {"document_id": request.document_id}
        with observe_flow(
            self.metrics,
            self.logger,
            module="documents",
            flow="delete_doc",
            context=context,
        ):
            document = self._find_memory(request.document_id)
            self.repository.store.delete(document.key)
            remaining = self.repository.list_memory_documents(
                document.metadata.directory_id
            )
            if remaining:
                self._refresh_directory_topic(document.metadata.directory_id)
            else:
                self._delete_directory_metadata(document.metadata.directory_id)
            self.index_store.rebuild(self.repository.list_memory_documents())
            self.metrics.documents_total.labels(
                module="documents", flow="delete_doc", kind="memory_deleted"
            ).inc()
            self.logger.info(
                "doc_document_deleted document_id=%s directory_id=%s key=%s",
                document.metadata.id,
                document.metadata.directory_id,
                document.key,
            )

    def delete_all(self) -> int:
        """Delete every object in the MemFlow instance's bound user scope."""

        with observe_flow(
            self.metrics,
            self.logger,
            module="documents",
            flow="delete_all",
        ):
            keys = self.repository.store.list_keys("")
            for key in keys:
                self.repository.store.delete(key)
            self.metrics.documents_total.labels(
                module="documents", flow="delete_all", kind="scope_object_deleted"
            ).inc(len(keys))
            self.logger.info("doc_scope_deleted objects=%d", len(keys))
            return len(keys)

    def _find_memory(self, document_id: str) -> MemoryDocument:
        matches = [
            document
            for document in self.repository.list_memory_documents()
            if document.metadata.id == document_id
        ]
        if not matches:
            raise LookupError(f"doc memory document not found: {document_id}")
        if len(matches) > 1:
            raise ValueError(f"duplicate doc memory document id: {document_id}")
        return matches[0]

    def _refresh_directory_topic(self, directory_id: str) -> None:
        if not directory_id:
            raise ValueError("doc memory document has no directory_id")
        documents = self.repository.list_memory_documents(directory_id)
        topic_key = self.layout.directory_topic_key(directory_id)
        legacy_key = self.layout.legacy_directory_summary_key(directory_id)
        previous: MemoryDocument | None = None
        if self.repository.store.exists(topic_key):
            previous = self.repository.read(topic_key)
        elif self.repository.store.exists(legacy_key):
            previous = self.repository.read(legacy_key)

        title = (
            previous.metadata.title
            if previous and previous.metadata.title
            else documents[0].metadata.title or "Memory"
        )
        topic = DirectoryTopicBuilder().build(
            directory_title=title,
            document_contents=[document.content for document in documents],
            existing_content=previous.content if previous else "",
        )
        now = beijing_now()
        self.repository.write(
            topic_key,
            MemoryDocument(
                metadata=DocumentMetadata(
                    id=directory_id,
                    kind="directory_topic",
                    title=topic.title,
                    summary=topic.summary,
                    document_count=len(documents),
                    content_digest=directory_content_digest(documents),
                    created_at=previous.metadata.created_at if previous else now,
                    updated_at=now,
                ),
                content=topic.body,
                key=topic_key,
            ),
        )
        if self.repository.store.exists(legacy_key):
            self.repository.store.delete(legacy_key)
        self.logger.info(
            "doc_directory_topic_refreshed directory_id=%s key=%s documents=%d",
            directory_id,
            topic_key,
            len(documents),
        )

    def _delete_directory_metadata(self, directory_id: str) -> None:
        if not directory_id:
            raise ValueError("doc memory document has no directory_id")
        for key in (
            self.layout.directory_topic_key(directory_id),
            self.layout.legacy_directory_summary_key(directory_id),
        ):
            if self.repository.store.exists(key):
                self.repository.store.delete(key)
        self.logger.info(
            "doc_directory_metadata_deleted directory_id=%s", directory_id
        )


def _to_doc_document(document: MemoryDocument) -> DocDocument:
    return DocDocument(
        document_id=document.metadata.id,
        content=document.content,
        line_count=len(document.content.splitlines()),
    )


__all__ = ["DocDocumentManager"]
