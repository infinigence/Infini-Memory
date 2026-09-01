"""Markdown normalization helpers shared across mem_flow stages."""

from __future__ import annotations

import re


_ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}[ \t]+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")


def normalize_headings_to_h1(content: str) -> str:
    """Promote every ATX heading outside fenced code blocks to H1.

    mem_flow uses headings as flat topic labels rather than a nested document
    outline. Models may still emit H2-H6 headings, so normalize the complete
    supported ATX range before content crosses a pipeline stage boundary.
    """

    normalized_lines: list[str] = []
    fence_character = ""
    fence_length = 0
    for line in content.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            remainder = fence_match.group(2).strip()
            if not fence_character:
                fence_character = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_character
                and len(marker) >= fence_length
                and not remainder
            ):
                fence_character = ""
                fence_length = 0
            normalized_lines.append(line)
            continue

        heading_match = None if fence_character else _ATX_HEADING_RE.match(line)
        if heading_match:
            line = f"# {heading_match.group(1)}"
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


__all__ = ["normalize_headings_to_h1"]
