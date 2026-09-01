"""Convenient composition root for the three independently callable modules."""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from .config import MemFlowConfig
from .documents import DocDocumentManager
from .extractor import MemoryExtractor
from .index import FactIndexStore
from .llm import FlowLLM, LLMConcurrencyLimiter, OpenAIFlowLLM
from .maintainer import MaintenanceProgress, MemoryMaintainer
from .models import (
    DocDocument,
    DocLineUpdateRequest,
    DocLineUpdateResult,
    DocReadRequest,
    ExtractionRequest,
    ExtractionResult,
    MaintenanceRequest,
    MaintenanceResult,
    LegacyMigrationResult,
    MemoryScope,
    SearchRequest,
    SearchResult,
)
from .observability import FlowMetrics
from .repository import MemoryRepository
from .retriever import MemoryRetriever
from .storage import ObjectStore, ScopedObjectStore, create_object_store
from .utils.paths import KeyLayout


class MemFlow(BaseModel):
    """Holds independent extractor, maintainer, and retriever instances."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: MemFlowConfig
    scope: MemoryScope
    extractor: MemoryExtractor
    maintainer: MemoryMaintainer
    retriever: MemoryRetriever
    documents: DocDocumentManager

    @property
    def instance_id(self) -> str:
        """Configured instance id, or an initialization-time UUID string."""
        return self.extractor.instance_id

    @classmethod
    def create(
        cls,
        config: MemFlowConfig,
        *,
        store_id: str,
        user_id: str,
        store: ObjectStore | None = None,
        llm: FlowLLM | None = None,
        metrics: FlowMetrics | None = None,
        metrics_registry: object | None = None,
        instance_id: str | UUID | None = None,
        llm_limiter: LLMConcurrencyLimiter | None = None,
    ) -> "MemFlow":
        metrics = metrics or FlowMetrics.create(
            namespace=config.metrics.namespace,
            enabled=config.metrics.enabled,
            registry=metrics_registry,
        )
        scope = MemoryScope(store_id=store_id, user_id=user_id)
        object_store = (
            store if store is not None else create_object_store(config, metrics)
        )
        scoped_store = ScopedObjectStore.create(
            object_store,
            fixed_prefix=config.storage_prefix,
            scope=scope,
        )
        flow_llm = llm or OpenAIFlowLLM.create(config.llm, metrics, limiter=llm_limiter)
        layout = KeyLayout()
        repository = MemoryRepository(store=scoped_store, layout=layout, scope=scope)
        index_store = FactIndexStore(repository=repository, config=config.index)
        instance_options = (
            {} if instance_id is None else {"instance_id": str(instance_id)}
        )
        extractor = MemoryExtractor(
            repository=repository,
            layout=layout,
            llm=flow_llm,
            metrics=metrics,
            config=config.extraction,
            **instance_options,
        )
        flow = cls(
            config=config,
            scope=scope,
            extractor=extractor,
            maintainer=MemoryMaintainer(
                repository=repository,
                layout=layout,
                llm=flow_llm,
                metrics=metrics,
                config=config.maintenance,
                index_store=index_store,
            ),
            retriever=MemoryRetriever(
                repository=repository,
                llm=flow_llm,
                metrics=metrics,
                config=config.retrieval,
                index_store=index_store,
            ),
            documents=DocDocumentManager(
                repository=repository,
                layout=layout,
                metrics=metrics,
                index_store=index_store,
            ),
        )
        logging.getLogger("mem_flow.client").info(
            "mem_flow_initialized instance_id=%s store_id=%s user_id=%s storage=%s",
            flow.instance_id,
            scope.store_id,
            scope.user_id,
            config.storage.type,
        )
        return flow

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        return self.extractor.extract(request)

    def maintain(
        self,
        request: MaintenanceRequest,
        *,
        progress: MaintenanceProgress | None = None,
    ) -> MaintenanceResult:
        return self.maintainer.maintain(request, progress=progress)

    def search(self, request: SearchRequest) -> SearchResult:
        return self.retriever.search(request)

    def rebuild_index(self) -> dict[str, object]:
        """Rebuild the disposable structured sidecar from authoritative Markdown."""

        snapshot = self.retriever.index_store.rebuild(
            self.retriever.repository.list_memory_documents()
        )
        return {
            "persisted": snapshot.persisted,
            "facts": len(snapshot.facts),
            "entities": len(snapshot.entities),
            "relations": len(snapshot.relations),
            "manifest_digest": snapshot.manifest.digest if snapshot.manifest else "",
        }

    def validate_index(self) -> dict[str, object]:
        """Validate ACTIVE and all referenced artifacts against Markdown digests."""

        documents = self.retriever.repository.list_memory_documents()
        snapshot = self.retriever.index_store.load(documents)
        return {
            "valid": snapshot is not None,
            "facts": len(snapshot.facts) if snapshot else 0,
            "manifest_digest": (
                snapshot.manifest.digest if snapshot and snapshot.manifest else ""
            ),
        }

    def get_doc(self, request: DocReadRequest) -> DocDocument:
        """Read one maintained memory leaf by document id."""
        return self.documents.get(request)

    def update_doc_lines(self, request: DocLineUpdateRequest) -> DocLineUpdateResult:
        """Replace an inclusive one-based line range in a memory leaf body."""
        return self.documents.update_lines(request)

    def delete_doc(self, request: DocReadRequest) -> None:
        """Delete one maintained memory leaf by document id."""
        self.documents.delete(request)

    def delete_all(self) -> int:
        """Delete all objects in the bound store and user scope."""
        return self.documents.delete_all()

    def migrate_legacy_topics(
        self,
        *,
        dry_run: bool = True,
        delete_source: bool = False,
    ) -> LegacyMigrationResult:
        return self.maintainer.migrate_legacy_topics(
            dry_run=dry_run, delete_source=delete_source
        )


__all__ = ["MemFlow"]
