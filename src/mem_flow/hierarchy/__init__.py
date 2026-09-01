"""Directory routing and TOPIC.md construction for hierarchical memories."""

from .models import (
    DirectoryAssignment,
    DirectoryRoutePlan,
    DirectoryTopicContent,
    DocumentSummaryResponse,
    MarkdownFact,
    PersistedRouteAssignment,
    PersistedRoutePlan,
)
from .router import DirectoryRouter
from .summary import (
    DirectorySummaryBuilder,
    DocumentSummaryBuilder,
    deterministic_document_summary,
    normalize_document_summary,
)
from .topic import (
    DirectoryTopicBuilder,
    directory_content_digest,
    extract_h1_headings,
)

__all__ = [
    "DirectoryAssignment",
    "DirectoryRoutePlan",
    "DirectoryRouter",
    "DirectorySummaryBuilder",
    "DirectoryTopicBuilder",
    "DirectoryTopicContent",
    "DocumentSummaryBuilder",
    "DocumentSummaryResponse",
    "MarkdownFact",
    "PersistedRouteAssignment",
    "PersistedRoutePlan",
    "directory_content_digest",
    "extract_h1_headings",
    "deterministic_document_summary",
    "normalize_document_summary",
]
