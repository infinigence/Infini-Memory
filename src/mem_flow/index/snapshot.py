"""Content-addressed publication of structured fact sidecars."""

from __future__ import annotations

import hashlib
import json
import logging

from pydantic import BaseModel, ConfigDict

from ..config import IndexConfig
from ..models import MemoryDocument
from ..repository import MemoryRepository
from ..utils.codec import strip_yaml_front_matter
from .models import (
    ActiveIndexPointer,
    FactIndexSnapshot,
    FactRelation,
    IndexManifest,
    IndexSource,
    StructuredFact,
)
from .projector import build_entities, build_relations, project_document, project_documents


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: object) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    elif isinstance(value, list):
        payload = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in value
        ]
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _manifest_seed(
    schema_version: int,
    sources: list[IndexSource],
    relation_artifact_key: str,
    relation_artifact_digest: str,
) -> str:
    return _json(
        {
            "schema_version": schema_version,
            "sources": [item.model_dump(mode="json") for item in sources],
            "relation_artifact_key": relation_artifact_key,
            "relation_artifact_digest": relation_artifact_digest,
        }
    )


class FactIndexStore(BaseModel):
    """Persist and validate indexes without making them a source of truth."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repository: MemoryRepository
    config: IndexConfig

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.index")

    @property
    def prefix(self) -> str:
        return f"index/v{self.config.schema_version}"

    @property
    def active_key(self) -> str:
        return f"{self.prefix}/ACTIVE.json"

    def rebuild(self, documents: list[MemoryDocument] | None = None) -> FactIndexSnapshot:
        documents = documents if documents is not None else self.repository.list_memory_documents()
        snapshot = project_documents(documents)
        if not self.config.enabled:
            return snapshot

        sources: list[IndexSource] = []
        persisted_facts: list[StructuredFact] = []
        for document in sorted(documents, key=lambda item: item.metadata.id):
            body_digest = _digest(strip_yaml_front_matter(document.content))
            facts = project_document(document)
            facts_json = _json(facts)
            artifact_digest = _digest(facts_json)
            artifact_key = (
                f"{self.prefix}/sources/{_digest(document.metadata.id)[:20]}/"
                f"{body_digest}_{artifact_digest}.json"
            )
            self.repository.store.put_text(artifact_key, facts_json)
            sources.append(
                IndexSource(
                    document_id=document.metadata.id,
                    document_key=document.key,
                    document_digest=body_digest,
                    artifact_key=artifact_key,
                    artifact_digest=artifact_digest,
                    fact_count=len(facts),
                )
            )
            persisted_facts.extend(facts)

        relations = build_relations(persisted_facts)
        relations_json = _json(relations)
        relation_digest = _digest(relations_json)
        relation_key = f"{self.prefix}/relations/RELATIONS_{relation_digest}.json"
        self.repository.store.put_text(relation_key, relations_json)
        manifest_seed = _manifest_seed(
            self.config.schema_version,
            sources,
            relation_key,
            relation_digest,
        )
        manifest_digest = _digest(manifest_seed)
        manifest = IndexManifest(
            schema_version=self.config.schema_version,
            digest=manifest_digest,
            sources=sources,
            relation_artifact_key=relation_key,
            relation_artifact_digest=relation_digest,
        )
        manifest_key = f"{self.prefix}/manifests/MANIFEST_{manifest_digest}.json"
        self.repository.store.put_text(manifest_key, _json(manifest))
        pointer = ActiveIndexPointer(
            schema_version=self.config.schema_version,
            manifest_key=manifest_key,
            manifest_digest=manifest_digest,
        )
        self.repository.store.put_text(self.active_key, _json(pointer))
        snapshot.manifest = manifest
        snapshot.persisted = True
        self.logger.info(
            "fact_index_published manifest=%s sources=%d facts=%d relations=%d",
            manifest_digest,
            len(sources),
            len(persisted_facts),
            len(relations),
        )
        return snapshot

    def load(self, documents: list[MemoryDocument]) -> FactIndexSnapshot | None:
        if not self.config.enabled or not self.repository.store.exists(self.active_key):
            return None
        try:
            pointer = ActiveIndexPointer.model_validate_json(
                self.repository.store.get_text(self.active_key)
            )
            if pointer.schema_version != self.config.schema_version:
                return None
            manifest = IndexManifest.model_validate_json(
                self.repository.store.get_text(pointer.manifest_key)
            )
            if manifest.digest != pointer.manifest_digest:
                return None
            if _digest(
                _manifest_seed(
                    manifest.schema_version,
                    manifest.sources,
                    manifest.relation_artifact_key,
                    manifest.relation_artifact_digest,
                )
            ) != manifest.digest:
                return None
            actual = {
                item.metadata.id: _digest(strip_yaml_front_matter(item.content))
                for item in documents
            }
            expected = {item.document_id: item.document_digest for item in manifest.sources}
            if actual != expected:
                self.logger.info("fact_index_stale expected=%d actual=%d", len(expected), len(actual))
                return None
            facts: list[StructuredFact] = []
            for source in manifest.sources:
                raw_payload = self.repository.store.get_text(source.artifact_key)
                if _digest(raw_payload) != source.artifact_digest:
                    return None
                payload = json.loads(raw_payload)
                source_facts = [StructuredFact.model_validate(item) for item in payload]
                if len(source_facts) != source.fact_count:
                    return None
                facts.extend(source_facts)
            raw_relations = self.repository.store.get_text(
                manifest.relation_artifact_key
            )
            if _digest(raw_relations) != manifest.relation_artifact_digest:
                return None
            relations_payload = json.loads(raw_relations)
            relations = [FactRelation.model_validate(item) for item in relations_payload]
            return FactIndexSnapshot(
                facts=facts,
                entities=build_entities(facts),
                relations=relations,
                manifest=manifest,
                persisted=True,
            )
        except Exception:
            self.logger.warning("fact_index_invalid", exc_info=True)
            return None

    def load_with_overlay(self, documents: list[MemoryDocument]) -> FactIndexSnapshot:
        persistent_documents = [
            item for item in documents if item.metadata.kind in {"memory", "topic"}
        ]
        persisted = self.load(persistent_documents)
        working = [
            item
            for item in documents
            if item.metadata.kind in {"current", "raw", "rewrite"}
        ]
        if persisted is None:
            snapshot = project_documents(documents)
            snapshot.persisted = False
            return snapshot
        if not working:
            return persisted
        combined = project_documents([*persistent_documents, *working])
        combined.manifest = persisted.manifest
        combined.persisted = True
        return combined


__all__ = ["FactIndexStore"]
