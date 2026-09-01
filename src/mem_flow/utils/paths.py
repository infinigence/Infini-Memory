"""Centralized user-scope-relative object key construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedDocKey:
    kind: str
    directory_id: str = ""
    filename: str = ""


class KeyLayout:
    """Build paths visible inside one bound MemFlow instance."""

    def current_prefix(self) -> str:
        return "current"

    def current_key(self, instance_id: str) -> str:
        return f"{self.current_prefix()}/CURRENT_{_safe_segment(instance_id)}.md"

    def current_full_key(self, instance_id: str, full_number: int) -> str:
        return f"{self.current_prefix()}/CURRENT_{_safe_segment(instance_id)}_full_{full_number}.md"

    def rewrite_prefix(self) -> str:
        return "rewrite"

    def raw_prefix(self) -> str:
        return "raw"

    def raw_key(self, document_id: str) -> str:
        return f"{self.raw_prefix()}/{_safe_segment(document_id)}.md"

    def evidence_prefix(self) -> str:
        return "evidence"

    def evidence_key(self, document_id: str) -> str:
        return f"{self.evidence_prefix()}/{_safe_segment(document_id)}.md"

    def doc_prefix(self) -> str:
        return "doc"

    def directory_prefix(self, directory_id: str) -> str:
        return f"{self.doc_prefix()}/{_safe_segment(directory_id)}"

    def directory_topic_key(self, directory_id: str) -> str:
        return f"{self.directory_prefix(directory_id)}/TOPIC.md"

    def legacy_directory_summary_key(self, directory_id: str) -> str:
        return f"{self.directory_prefix(directory_id)}/SUMMARY.md"

    def directory_summary_key(self, directory_id: str) -> str:
        """Compatibility alias returning the current TOPIC.md key."""
        return self.directory_topic_key(directory_id)

    def memory_key(self, directory_id: str, document_id: str) -> str:
        safe_document_id = _safe_segment(document_id)
        if safe_document_id.casefold() in {"topic", "summary"}:
            raise ValueError("TOPIC and SUMMARY are reserved for directory metadata")
        return f"{self.directory_prefix(directory_id)}/{safe_document_id}.md"

    def route_key(self, rewrite_id: str) -> str:
        return f"{self.rewrite_prefix()}/ROUTE_{_safe_segment(rewrite_id)}.json"

    def parse_doc_key(self, key: str) -> ParsedDocKey:
        prefix = f"{self.doc_prefix()}/"
        if not key.startswith(prefix):
            raise ValueError(f"key is outside doc prefix: {key}")
        parts = key[len(prefix) :].split("/")
        if len(parts) == 1 and parts[0].endswith(".md"):
            return ParsedDocKey(kind="legacy", filename=parts[0])
        if len(parts) != 2 or not parts[1].endswith(".md"):
            raise ValueError(f"invalid hierarchical doc key: {key}")
        directory_id, filename = parts
        _safe_segment(directory_id)
        if filename == "TOPIC.md":
            return ParsedDocKey(
                kind="directory_topic",
                directory_id=directory_id,
                filename=filename,
            )
        if filename == "SUMMARY.md":
            return ParsedDocKey(
                kind="legacy_summary",
                directory_id=directory_id,
                filename=filename,
            )
        _safe_segment(filename.removesuffix(".md"))
        return ParsedDocKey(kind="memory", directory_id=directory_id, filename=filename)

    # Compatibility aliases for callers transitioning from the flat layout.
    def topic_prefix(self) -> str:
        return self.doc_prefix()

    def topic_key(self, document_id: str) -> str:
        return f"{self.doc_prefix()}/{_safe_segment(document_id)}.md"


def _safe_segment(value: str) -> str:
    value = value.strip()
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"unsafe S3 key segment: {value!r}")
    return value


__all__ = ["KeyLayout", "ParsedDocKey"]
