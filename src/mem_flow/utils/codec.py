"""Markdown document codec using standard YAML Front Matter."""

from __future__ import annotations

import yaml

from ..models import DocumentMetadata, MemoryDocument


def strip_yaml_front_matter(content: str) -> str:
    """Return only a Markdown document body.

    Repository reads already decode the persisted envelope, but prompt builders also
    use this helper as a defensive boundary.  That prevents an accidentally nested
    or directly supplied persisted document from exposing storage metadata to an
    LLM.  A leading Markdown horizontal rule is preserved unless the delimited
    section is a valid YAML mapping.
    """

    text = content.strip()
    if not text.startswith("---\n"):
        return text
    try:
        metadata_raw, body = text[4:].split("\n---\n", 1)
        metadata = yaml.safe_load(metadata_raw)
    except (ValueError, yaml.YAMLError):
        return text
    return body.strip() if isinstance(metadata, dict) else text


def encode_document(document: MemoryDocument) -> str:
    if document.metadata.kind in {"current", "raw", "rewrite"}:
        excluded_fields = {
            "summary",
            "title",
            "directory_id",
            "document_count",
            "content_digest",
        }
    elif document.metadata.kind in {"memory", "topic"}:
        excluded_fields = {"document_count", "content_digest"}
    elif document.metadata.kind == "evidence":
        excluded_fields = {
            "directory_id",
            "document_count",
            "content_digest",
            "source_ids",
        }
    else:
        excluded_fields = {"directory_id", "source_ids"}
    metadata = document.metadata.model_dump(mode="json", exclude=excluded_fields)
    front_matter = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()
    return f"---\n{front_matter}\n---\n{document.content.strip()}\n"


def decode_document(raw: str, *, key: str = "") -> MemoryDocument:
    if not raw.startswith("---\n"):
        raise ValueError(f"invalid mem_flow document front matter: {key}")
    try:
        metadata_raw, content = raw[4:].split("\n---\n", 1)
        # JSON objects are valid YAML, so historical mem_flow documents remain
        # readable and are migrated to multiline YAML on their next write.
        metadata_data = yaml.safe_load(metadata_raw)
        if not isinstance(metadata_data, dict):
            raise ValueError("front matter must be a YAML mapping")
        metadata = DocumentMetadata.model_validate(metadata_data)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid mem_flow document: {key}") from exc
    return MemoryDocument(metadata=metadata, content=content.strip(), key=key)


__all__ = ["decode_document", "encode_document", "strip_yaml_front_matter"]
