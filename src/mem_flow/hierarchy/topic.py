"""Build a directory index from the H1 headings of its memory documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from pydantic import BaseModel

from ..models import MemoryDocument
from .models import DirectoryTopicContent


_H1_RE = re.compile(r"^#(?!#)\s+(.+?)\s*$")


class DirectoryTopicBuilder(BaseModel):
    """Create TOPIC.md from leaf headings without copying leaf fact bodies.

    Existing headings retain their order and newly discovered headings are appended.
    The output contains one Markdown H1 per line and no prose overview.
    """

    def build(
        self,
        *,
        directory_title: str,
        document_contents: list[str],
        existing_content: str = "",
    ) -> DirectoryTopicContent:
        normalized_title = _normalize_heading(directory_title) or "Memory"
        document_headings = _unique_headings(
            heading
            for content in document_contents
            for heading in extract_h1_headings(content)
        )
        if document_headings:
            available = {heading.casefold() for heading in document_headings}
            existing = [
                heading
                for heading in extract_h1_headings(existing_content)
                if heading.casefold() in available
            ]
            headings = _unique_headings([*existing, *document_headings])
        else:
            # Module-created leaves always contain H1 headings. Keep legacy leaves
            # without headings routable until they are migrated.
            headings = [normalized_title]
        return DirectoryTopicContent(
            title=normalized_title,
            summary=normalized_title,
            headings=headings,
            body="\n".join(f"# {heading}" for heading in headings),
        )


def extract_h1_headings(content: str) -> list[str]:
    """Return normalized H1 headings outside fenced code blocks."""

    headings: list[str] = []
    fence_character = ""
    for line in content.splitlines():
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not fence_character:
                fence_character = marker
            elif marker == fence_character:
                fence_character = ""
            continue
        match = None if fence_character else _H1_RE.match(line)
        if match:
            normalized = _normalize_heading(match.group(1))
            if normalized:
                headings.append(normalized)
    return _unique_headings(headings)


def directory_content_digest(documents: list[MemoryDocument]) -> str:
    """Hash the leaf ids, summaries, and bodies represented by a TOPIC file."""

    payload = "\0".join(
        f"{item.metadata.id}:{item.metadata.summary}:"
        f"{hashlib.sha256(item.content.encode()).hexdigest()}"
        for item in sorted(documents, key=lambda document: document.metadata.id)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _normalize_heading(value: str) -> str:
    normalized = re.sub(r"^#+\s*", "", value.strip())
    normalized = re.sub(r"\s+#+\s*$", "", normalized)
    return " ".join(normalized.split())


def _unique_headings(headings: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in headings:
        heading = _normalize_heading(str(value))
        key = heading.casefold()
        if heading and key not in seen:
            unique.append(heading)
            seen.add(key)
    return unique


__all__ = [
    "DirectoryTopicBuilder",
    "directory_content_digest",
    "extract_h1_headings",
]
