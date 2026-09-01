"""Document persistence isolated from memory workflows."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from .models import MemoryDocument, MemoryScope, SearchSource
from .storage import ObjectStore
from .utils.codec import decode_document, encode_document
from .utils.paths import KeyLayout


class MemoryRepository(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: ObjectStore
    layout: KeyLayout
    scope: MemoryScope

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.repository")

    def read(self, key: str) -> MemoryDocument:
        self.logger.debug("document_read key=%s", key)
        return decode_document(self.store.get_text(key), key=key)

    def write(self, key: str, document: MemoryDocument) -> None:
        document.metadata.store_id = self.scope.store_id
        document.metadata.user_id = self.scope.user_id
        document.key = key
        self.logger.debug(
            "document_write key=%s id=%s kind=%s",
            key,
            document.metadata.id,
            document.metadata.kind,
        )
        self.store.put_text(key, encode_document(document))

    def write_body(self, key: str, content: str) -> None:
        """Replace only Markdown body bytes and preserve Front Matter verbatim."""

        raw = self.store.get_text(key)
        if not raw.startswith("---\n"):
            raise ValueError(f"invalid mem_flow document front matter: {key}")
        try:
            front_matter, _ = raw[4:].split("\n---\n", 1)
        except ValueError as exc:
            raise ValueError(f"invalid mem_flow document: {key}") from exc
        self.logger.debug("document_body_write key=%s", key)
        self.store.put_text(key, f"---\n{front_matter}\n---\n{content.strip()}\n")

    def list_current(self) -> list[MemoryDocument]:
        return self._list_documents(self.layout.current_prefix())

    def list_raw(self) -> list[MemoryDocument]:
        return self._list_documents(self.layout.raw_prefix())

    def list_rewrites(self) -> list[MemoryDocument]:
        return self._list_documents(self.layout.rewrite_prefix())

    def list_evidence(self) -> list[MemoryDocument]:
        return self._list_documents(self.layout.evidence_prefix())

    def list_searchable_working(
        self, sources: set[SearchSource]
    ) -> list[MemoryDocument]:
        documents: list[MemoryDocument] = []
        if SearchSource.EVIDENCE in sources:
            documents.extend(self.list_evidence())
        if SearchSource.CURRENT in sources:
            documents.extend(self.list_current())
        if SearchSource.RAW in sources:
            documents.extend(self.list_raw())
        if SearchSource.REWRITE in sources:
            documents.extend(self.list_rewrites())
        return sorted(documents, key=lambda item: (item.metadata.kind, item.key))

    def list_directory_topics(self) -> list[MemoryDocument]:
        """List TOPIC.md objects, falling back to legacy SUMMARY.md per directory."""
        topics: dict[str, MemoryDocument] = {}
        for key in self.store.list_keys(self.layout.doc_prefix()):
            try:
                parsed = self.layout.parse_doc_key(key)
            except ValueError:
                self.logger.warning("invalid_doc_key_skipped key=%s", key)
                continue
            if parsed.kind not in {"directory_topic", "legacy_summary"}:
                continue
            document = self.read(key)
            expected_kind = (
                "directory_topic"
                if parsed.kind == "directory_topic"
                else "directory_summary"
            )
            if (
                document.metadata.kind != expected_kind
                or document.metadata.id != parsed.directory_id
            ):
                raise ValueError(f"directory topic metadata does not match key: {key}")
            if parsed.kind == "directory_topic" or parsed.directory_id not in topics:
                topics[parsed.directory_id] = document
        return sorted(topics.values(), key=lambda item: item.metadata.id)

    def list_directory_summaries(self) -> list[MemoryDocument]:
        """Compatibility alias for callers transitioning to TOPIC.md."""
        return self.list_directory_topics()

    def list_directory_ids(self) -> list[str]:
        directory_ids: set[str] = set()
        for key in self.store.list_keys(self.layout.doc_prefix()):
            try:
                parsed = self.layout.parse_doc_key(key)
            except ValueError:
                continue
            if parsed.directory_id:
                directory_ids.add(parsed.directory_id)
        return sorted(directory_ids)

    def list_memory_documents(
        self, directory_id: str | None = None
    ) -> list[MemoryDocument]:
        prefix = (
            self.layout.directory_prefix(directory_id)
            if directory_id
            else self.layout.doc_prefix()
        )
        memories: list[MemoryDocument] = []
        for key in self.store.list_keys(prefix):
            try:
                parsed = self.layout.parse_doc_key(key)
            except ValueError:
                continue
            if parsed.kind != "memory" or (
                directory_id and parsed.directory_id != directory_id
            ):
                continue
            document = self.read(key)
            if document.metadata.kind not in {"memory", "topic"}:
                raise ValueError(f"non-memory metadata stored at memory key: {key}")
            if (
                document.metadata.directory_id
                and document.metadata.directory_id != parsed.directory_id
            ):
                raise ValueError(f"memory directory metadata does not match key: {key}")
            memories.append(document)
        return sorted(memories, key=lambda item: item.key)

    def list_legacy_topics(self) -> list[MemoryDocument]:
        topics: list[MemoryDocument] = []
        for key in self.store.list_keys(self.layout.doc_prefix()):
            try:
                parsed = self.layout.parse_doc_key(key)
            except ValueError:
                continue
            if parsed.kind == "legacy":
                document = self.read(key)
                if document.metadata.kind in {"topic", "memory"}:
                    topics.append(document)
        return sorted(topics, key=lambda item: item.key)

    def list_topics(self) -> list[MemoryDocument]:
        """Compatibility view of legacy topics plus hierarchical memory leaves."""
        return [*self.list_legacy_topics(), *self.list_memory_documents()]

    def _list_documents(self, prefix: str) -> list[MemoryDocument]:
        documents: list[MemoryDocument] = []
        for key in self.store.list_keys(prefix):
            if key.endswith(".md"):
                documents.append(self.read(key))
        self.logger.info("documents_listed prefix=%s count=%d", prefix, len(documents))
        return sorted(documents, key=lambda item: item.key)


__all__ = ["MemoryRepository"]
