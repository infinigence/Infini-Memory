"""LLM-assisted fact routing with code-owned ids and conservation checks."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from ..llm import FlowLLM, complete_structured
from ..models import ChatMessage, LLMRequest, MemoryDocument, compact_beijing_timestamp
from ..prompts import DIRECTORY_ROUTE_PROMPT
from ..utils.codec import strip_yaml_front_matter
from ..utils.parsing import parse_json_model
from .models import (
    DirectoryAssignment,
    DirectoryRoutePlan,
    MarkdownFact,
    PersistedRouteAssignment,
    PersistedRoutePlan,
)
from .summary import deterministic_document_summary, normalize_document_summary


class DirectoryRouter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: FlowLLM

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.hierarchy.router")

    def create_plan(
        self,
        *,
        rewrite_id: str,
        facts: list[MarkdownFact],
        directories: list[MemoryDocument],
        summary_tokens: int = 100,
    ) -> PersistedRoutePlan:
        catalog = json.dumps(
            [
                {
                    "directory_id": item.metadata.id,
                    "body": strip_yaml_front_matter(item.content),
                }
                for item in directories
            ],
            ensure_ascii=False,
        )
        fact_json = json.dumps(
            [fact.model_dump() for fact in facts], ensure_ascii=False
        )
        request = LLMRequest(
            operation="route_directories",
            messages=[
                ChatMessage(
                    role="user",
                    content=DIRECTORY_ROUTE_PROMPT.format(
                        catalog=catalog,
                        facts=fact_json,
                        summary_tokens=summary_tokens,
                    ),
                )
            ],
        )

        def validate(raw: str) -> PersistedRoutePlan:
            plan = parse_json_model(raw, DirectoryRoutePlan)
            self._validate(plan, facts, directories)
            return self._materialize(
                rewrite_id,
                plan,
                directories,
                summary_tokens=summary_tokens,
            )

        try:
            persisted = complete_structured(self.llm, request, validate)
        except (ValueError, RuntimeError):
            # Preserve fact conservation even when the model repeatedly emits
            # invalid JSON, unknown ids, or duplicate/missing assignments.
            # Matching an existing directory title is deterministic; otherwise
            # the original rewrite H1 becomes the new directory title.
            self.logger.warning(
                "directory_route_fallback rewrite_id=%s facts=%d",
                rewrite_id,
                len(facts),
            )
            fallback = self._fallback_plan(
                facts, directories, summary_tokens=summary_tokens
            )
            persisted = self._materialize(
                rewrite_id,
                fallback,
                directories,
                summary_tokens=summary_tokens,
            )
        known_directory_ids = {item.metadata.id for item in directories}
        existing_targets = sum(
            item.directory_id in known_directory_ids for item in persisted.assignments
        )
        self.logger.info(
            "directory_route_completed rewrite_id=%s existing_targets=%d "
            "new_targets=%d destinations=%d facts=%d",
            rewrite_id,
            existing_targets,
            len(persisted.assignments) - existing_targets,
            len(persisted.assignments),
            len(facts),
        )
        return persisted

    def create_deterministic_plan(
        self,
        *,
        rewrite_id: str,
        facts: list[MarkdownFact],
        directories: list[MemoryDocument],
        summary_tokens: int = 100,
    ) -> PersistedRoutePlan:
        """Route by stable H1 identity without making an LLM request."""

        fallback = self._fallback_plan(
            facts, directories, summary_tokens=summary_tokens
        )
        return self._materialize(
            rewrite_id,
            fallback,
            directories,
            summary_tokens=summary_tokens,
        )

    @staticmethod
    def _fallback_plan(
        facts: list[MarkdownFact],
        directories: list[MemoryDocument],
        *,
        summary_tokens: int,
    ) -> DirectoryRoutePlan:
        existing_by_title = {
            item.metadata.title.casefold(): item.metadata.id for item in directories
        }
        grouped: OrderedDict[str, list[str]] = OrderedDict()
        titles: dict[str, str] = {}
        for fact in facts:
            title = " ".join(fact.heading.split()) or "Memory"
            normalized = title.casefold()
            titles.setdefault(normalized, title)
            grouped.setdefault(normalized, []).append(fact.id)
        return DirectoryRoutePlan(
            assignments=[
                DirectoryAssignment(
                    directory_id=existing_by_title.get(normalized),
                    new_directory_title=(
                        None
                        if normalized in existing_by_title
                        else titles[normalized]
                    ),
                    fact_ids=fact_ids,
                    summary=deterministic_document_summary(
                        "\n".join(
                            fact.markdown for fact in facts if fact.id in fact_ids
                        ),
                        summary_tokens=summary_tokens,
                    ),
                )
                for normalized, fact_ids in grouped.items()
            ]
        )

    @staticmethod
    def _validate(
        plan: DirectoryRoutePlan,
        facts: list[MarkdownFact],
        directories: list[MemoryDocument],
    ) -> None:
        allowed_facts = {fact.id for fact in facts}
        assigned = [
            fact_id
            for assignment in plan.assignments
            for fact_id in assignment.fact_ids
        ]
        allowed_directories = {item.metadata.id for item in directories}
        destinations = [
            (
                f"existing:{assignment.directory_id}"
                if assignment.directory_id is not None
                else f"new:{assignment.new_directory_title.casefold()}"
            )
            for assignment in plan.assignments
        ]
        if (
            len(assigned) != len(set(assigned))
            or set(assigned) != allowed_facts
            or len(destinations) != len(set(destinations))
            or any(
                assignment.directory_id is not None
                and assignment.directory_id not in allowed_directories
                for assignment in plan.assignments
            )
        ):
            raise ValueError("invalid directory route plan")

    @staticmethod
    def _materialize(
        rewrite_id: str,
        plan: DirectoryRoutePlan,
        directories: list[MemoryDocument],
        *,
        summary_tokens: int,
    ) -> PersistedRoutePlan:
        titles = {item.metadata.id: item.metadata.title for item in directories}
        new_ids: dict[str, str] = {}
        grouped: OrderedDict[str, tuple[str, list[str], str]] = OrderedDict()
        for assignment in plan.assignments:
            if assignment.directory_id is not None:
                directory_id = assignment.directory_id
                title = titles[directory_id]
            else:
                title = assignment.new_directory_title or "Memory"
                normalized = title.casefold()
                directory_id = new_ids.get(normalized, "")
                if not directory_id:
                    timestamp = compact_beijing_timestamp()
                    directory_id = f"dir_{timestamp}_{uuid4().hex[:12]}"
                    new_ids[normalized] = directory_id
            if directory_id not in grouped:
                grouped[directory_id] = (title, [], assignment.summary)
            grouped[directory_id][1].extend(assignment.fact_ids)

        assignments: list[PersistedRouteAssignment] = []
        for directory_id, (title, fact_ids, summary) in grouped.items():
            digest = hashlib.sha256(
                f"{rewrite_id}\0{directory_id}".encode()
            ).hexdigest()[:24]
            assignments.append(
                PersistedRouteAssignment(
                    directory_id=directory_id,
                    directory_title=title,
                    document_id=f"memory_{digest}",
                    fact_ids=fact_ids,
                    summary=normalize_document_summary(
                        summary, summary_tokens=summary_tokens
                    ),
                )
            )
        return PersistedRoutePlan(rewrite_id=rewrite_id, assignments=assignments)


__all__ = ["DirectoryRouter"]
