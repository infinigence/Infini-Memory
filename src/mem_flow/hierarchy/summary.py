"""Create retrieval summaries for immutable memory leaves."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, ConfigDict

from ..config import MaintenanceConfig
from ..llm import FlowLLM, complete_structured
from ..models import ChatMessage, LLMRequest
from ..prompts import DOCUMENT_SUMMARY_PROMPT
from ..utils.codec import strip_yaml_front_matter
from ..utils.parsing import parse_json_model
from .models import DocumentSummaryResponse


class DocumentSummaryBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: FlowLLM
    config: MaintenanceConfig

    def summarize_document(self, content: str) -> str:
        body = strip_yaml_front_matter(content)
        request = LLMRequest(
            operation="document_summary",
            messages=[
                ChatMessage(
                    role="user",
                    content=DOCUMENT_SUMMARY_PROMPT.format(
                        summary_tokens=self.config.document_summary_tokens,
                        content=body,
                    ),
                )
            ],
        )
        try:
            return complete_structured(
                self.llm,
                request,
                lambda raw: normalize_document_summary(
                    parse_json_model(raw, DocumentSummaryResponse).summary,
                    summary_tokens=self.config.document_summary_tokens,
                ),
            )
        except (ValueError, RuntimeError):
            logging.getLogger("mem_flow.hierarchy.summary").warning(
                "document_summary_fallback content_chars=%d", len(body)
            )
            return deterministic_document_summary(
                body, summary_tokens=self.config.document_summary_tokens
            )


def deterministic_document_summary(content: str, *, summary_tokens: int) -> str:
    """Build a metadata-free compatibility summary without an LLM call."""

    body = strip_yaml_front_matter(content)
    lines: list[str] = []
    for line in body.splitlines():
        compact = line.strip()
        if not compact:
            continue
        compact = re.sub(r"^#{1,6}\s+", "", compact)
        compact = re.sub(r"^(?:[-+*]|\d+[.)])\s+", "", compact)
        compact = re.sub(r"^<[^<>]*\bseq\s*=\s*[^<>]+>\s*", "", compact)
        if compact:
            lines.append(compact)
    return normalize_document_summary(
        " ".join(lines), summary_tokens=summary_tokens
    )


def normalize_document_summary(value: str, *, summary_tokens: int) -> str:
    """Remove model-owned metadata and enforce the configured summary bound."""

    summary = re.sub(
        r"<[^<>]*\bseq\s*=\s*[^<>]+>\s*", "", value, flags=re.IGNORECASE
    )
    summary = " ".join(summary.split())
    # Tokenization is model-specific; four characters per requested token is
    # the existing conservative deterministic approximation.
    return summary[: max(1, summary_tokens * 4)] or "Memory"


DirectorySummaryBuilder = DocumentSummaryBuilder


__all__ = [
    "DirectorySummaryBuilder",
    "DocumentSummaryBuilder",
    "deterministic_document_summary",
    "normalize_document_summary",
]
