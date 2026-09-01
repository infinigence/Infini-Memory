"""Pydantic contracts used by directory routing and topic generation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class MarkdownFact(BaseModel):
    id: str
    source_id: str
    heading: str
    markdown: str


class DirectoryAssignment(BaseModel):
    directory_id: str | None = None
    new_directory_title: str | None = None
    fact_ids: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)

    @field_validator("summary")
    @classmethod
    def normalize_summary(cls, value: str) -> str:
        summary = " ".join(value.split())
        if not summary:
            raise ValueError("summary cannot be blank")
        return summary

    @model_validator(mode="after")
    def validate_destination(self) -> "DirectoryAssignment":
        if (self.directory_id is None) == (self.new_directory_title is None):
            raise ValueError(
                "exactly one of directory_id and new_directory_title is required"
            )
        if self.new_directory_title is not None:
            title = " ".join(self.new_directory_title.split())
            if not title:
                raise ValueError("new_directory_title cannot be blank")
            self.new_directory_title = title
        return self


class DirectoryRoutePlan(BaseModel):
    assignments: list[DirectoryAssignment] = Field(min_length=1)


class PersistedRouteAssignment(BaseModel):
    directory_id: str
    directory_title: str
    document_id: str
    fact_ids: list[str] = Field(min_length=1)
    # Older persisted ROUTE objects did not include summaries. Keep them
    # readable and derive a deterministic summary when replaying such a route.
    summary: str = ""


class PersistedRoutePlan(BaseModel):
    rewrite_id: str
    assignments: list[PersistedRouteAssignment] = Field(min_length=1)


class DocumentSummaryResponse(BaseModel):
    summary: str = Field(min_length=1)


class DirectoryTopicContent(BaseModel):
    """H1 navigation index persisted in a directory TOPIC.md."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    headings: list[str] = Field(min_length=1)
    body: str = Field(min_length=1)


__all__ = [
    "DirectoryAssignment",
    "DirectoryRoutePlan",
    "DirectoryTopicContent",
    "DocumentSummaryResponse",
    "MarkdownFact",
    "PersistedRouteAssignment",
    "PersistedRoutePlan",
]
