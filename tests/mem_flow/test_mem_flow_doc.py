"""Multi-round state tests for mem_flow's hierarchical doc directory."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

from mem_flow import (
    FlowMetrics,
    LLMConfig,
    MaintenanceConfig,
    MaintenanceRequest,
    MemFlow,
    MemFlowConfig,
    S3Config,
)
from mem_flow.models import DocumentMetadata, LLMRequest, MemoryDocument


class InMemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def get_text(self, key: str) -> str:
        return self.objects[key]

    def put_text(self, key: str, content: str) -> None:
        self.objects[key] = content

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        normalized = prefix.rstrip("/") + "/"
        return sorted(key for key in self.objects if key.startswith(normalized))


class DocLifecycleLLM:
    """Return deterministic reference plans without modifying memory facts."""

    def __init__(self) -> None:
        self.operations: list[str] = []

    def complete(self, request: LLMRequest) -> str:
        self.operations.append(request.operation)
        prompt = request.messages[-1].content
        if request.operation == "rewrite_current":
            facts = _facts(prompt)
            grouped: dict[str, list[str]] = {}
            for fact in facts:
                grouped.setdefault(str(fact["heading"]), []).append(str(fact["id"]))
            return json.dumps(
                {
                    "topics": [
                        {"title": title, "fact_ids": fact_ids}
                        for title, fact_ids in grouped.items()
                    ],
                    "duplicates": [],
                }
            )
        if request.operation == "route_directories":
            facts = _facts(prompt)
            catalog = json.loads(
                prompt.split("DIRECTORY_CATALOG:\n", 1)[1].split("\n\nFACTS_JSON:", 1)[
                    0
                ]
            )
            existing = {
                title_match.group(1): str(item["directory_id"])
                for item in catalog
                if (
                    title_match := re.search(
                        r"^#\s+(.+?)\s*$", str(item["body"]), re.MULTILINE
                    )
                )
            }
            grouped: dict[str, list[str]] = {}
            for fact in facts:
                grouped.setdefault(str(fact["heading"]), []).append(str(fact["id"]))
            return json.dumps(
                {
                    "assignments": [
                        {
                            "directory_id": existing.get(title),
                            "new_directory_title": (
                                None if title in existing else title
                            ),
                            "fact_ids": fact_ids,
                            "summary": _summary_for_fact_ids(facts, fact_ids),
                        }
                        for title, fact_ids in grouped.items()
                    ]
                }
            )
        if request.operation == "document_summary":
            content = prompt.split("MEMORY_DOCUMENT:\n", 1)[1]
            facts = [
                re.sub(r"^-\s+<[^>]+>\s*", "", line).strip()
                for line in content.splitlines()
                if line.startswith("- <seq=")
            ]
            return json.dumps({"summary": " ".join(facts)})
        raise AssertionError(f"unexpected LLM operation: {request.operation}")


class MultiHeadingTopicLLM(DocLifecycleLLM):
    """Route several related rewrite headings into one directory."""

    def __init__(self) -> None:
        super().__init__()
        self.route_catalogs: list[list[dict[str, object]]] = []

    def complete(self, request: LLMRequest) -> str:
        if request.operation != "route_directories":
            return super().complete(request)
        self.operations.append(request.operation)
        prompt = request.messages[-1].content
        facts = _facts(prompt)
        catalog = json.loads(
            prompt.split("DIRECTORY_CATALOG:\n", 1)[1].split("\n\nFACTS_JSON:", 1)[0]
        )
        self.route_catalogs.append(catalog)
        return json.dumps(
            {
                "assignments": [
                    {
                        "directory_id": (
                            str(catalog[0]["directory_id"]) if catalog else None
                        ),
                        "new_directory_title": None if catalog else "Preferences",
                        "fact_ids": [str(fact["id"]) for fact in facts],
                        "summary": _summary_for_fact_ids(
                            facts, [str(fact["id"]) for fact in facts]
                        ),
                    }
                ]
            }
        )


class ParallelMaintenanceLLM(DocLifecycleLLM):
    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.active_summaries = 0
        self.max_active_summaries = 0
        self.active_rewrites = 0
        self.max_active_rewrites = 0

    def complete(self, request: LLMRequest) -> str:
        if request.operation == "rewrite_current":
            with self.lock:
                self.active_rewrites += 1
                self.max_active_rewrites = max(
                    self.max_active_rewrites, self.active_rewrites
                )
            time.sleep(0.02)
            try:
                return super().complete(request)
            finally:
                with self.lock:
                    self.active_rewrites -= 1
        if request.operation != "document_summary":
            return super().complete(request)
        with self.lock:
            self.active_summaries += 1
            self.max_active_summaries = max(
                self.max_active_summaries, self.active_summaries
            )
        time.sleep(0.02)
        try:
            return super().complete(request)
        finally:
            with self.lock:
                self.active_summaries -= 1


class InvalidRewritePlanLLM(DocLifecycleLLM):
    def __init__(self, invalid_responses: int) -> None:
        super().__init__()
        self.config = LLMConfig(retry_attempts=5, retry_initial_seconds=0)
        self.invalid_responses = invalid_responses
        self.rewrite_attempts = 0

    def complete(self, request: LLMRequest) -> str:
        if request.operation == "rewrite_current":
            self.rewrite_attempts += 1
            if self.rewrite_attempts <= self.invalid_responses:
                self.operations.append(request.operation)
                return json.dumps(
                    {
                        "topics": [
                            {"title": "Invalid", "fact_ids": ["unknown-fact"]}
                        ],
                        "duplicates": [],
                    }
                )
        return super().complete(request)


class InvalidRouteAndSummaryLLM(DocLifecycleLLM):
    def __init__(self) -> None:
        super().__init__()
        self.config = LLMConfig(retry_attempts=5, retry_initial_seconds=0)
        self.route_attempts = 0
        self.summary_attempts = 0

    def complete(self, request: LLMRequest) -> str:
        if request.operation == "route_directories":
            self.operations.append(request.operation)
            self.route_attempts += 1
            return "not json"
        if request.operation == "document_summary":
            self.operations.append(request.operation)
            self.summary_attempts += 1
            return '{"summary": "unterminated}'
        return super().complete(request)


class RuntimeFailureLLM(DocLifecycleLLM):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation

    def complete(self, request: LLMRequest) -> str:
        if request.operation == self.operation:
            self.operations.append(request.operation)
            raise RuntimeError("MaaS unavailable after transport retries")
        return super().complete(request)


def _facts(prompt: str) -> list[dict[str, object]]:
    return json.loads(prompt.split("FACTS_JSON:\n", 1)[1])


def _summary_for_fact_ids(
    facts: list[dict[str, object]], fact_ids: list[str]
) -> str:
    selected = {str(item) for item in fact_ids}
    return " ".join(
        re.sub(r"^[-*+]\s+<[^>]+>\s*", "", str(fact["markdown"])).strip()
        for fact in facts
        if str(fact["id"]) in selected
    )


def _directory_digest(documents: list[MemoryDocument]) -> str:
    payload = "\0".join(
        f"{document.metadata.id}:{document.metadata.summary}:"
        f"{hashlib.sha256(document.content.encode()).hexdigest()}"
        for document in sorted(documents, key=lambda item: item.metadata.id)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_current(
    flow: MemFlow,
    *,
    instance_id: str,
    created_at: datetime,
    sections: dict[str, list[tuple[int, str]]],
) -> str:
    content = "\n\n".join(
        f"# {title}\n\n"
        + "\n".join(f"- <seq={sequence}> {fact}" for sequence, fact in facts)
        for title, facts in sections.items()
    )
    key = flow.maintainer.layout.current_key(instance_id)
    flow.maintainer.repository.write(
        key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id=f"CURRENT_{instance_id}",
                kind="current",
                created_at=created_at,
                updated_at=created_at,
            ),
            content=content,
            key=key,
        ),
    )
    return key


def _assert_doc_state(
    flow: MemFlow,
    expected: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    repository = flow.maintainer.repository
    layout = flow.maintainer.layout
    topics = repository.list_directory_topics()
    topics_by_title = {item.metadata.title: item for item in topics}
    assert set(topics_by_title) == set(expected)
    assert len(repository.list_directory_ids()) == len(expected)
    assert repository.list_legacy_topics() == []

    leaf_bytes: dict[str, str] = {}
    topic_bytes: dict[str, str] = {}
    expected_doc_keys: set[str] = set()
    for title, expected_facts in expected.items():
        topic = topics_by_title[title]
        directory_id = topic.metadata.id
        documents = repository.list_memory_documents(directory_id)
        assert topic.key == layout.directory_topic_key(directory_id)
        assert topic.metadata.kind == "directory_topic"
        assert topic.metadata.title == title
        assert topic.metadata.summary == title
        assert topic.metadata.document_count == len(documents)
        assert topic.metadata.content_digest == _directory_digest(documents)
        assert topic.content == f"# {title}"
        assert len(documents) == len(expected_facts)

        combined_content = "\n".join(document.content for document in documents)
        combined_summary = " ".join(document.metadata.summary for document in documents)
        for fact in expected_facts:
            assert combined_content.count(fact) == 1
            assert fact in combined_summary
            assert fact not in topic.metadata.summary
            assert fact not in topic.content

        expected_doc_keys.add(topic.key)
        topic_bytes[title] = repository.store.get_text(topic.key)
        for document in documents:
            assert document.metadata.kind == "memory"
            assert document.metadata.directory_id == directory_id
            assert document.metadata.title == title
            assert len(document.metadata.source_ids) == 1
            assert document.key == layout.memory_key(directory_id, document.metadata.id)
            expected_doc_keys.add(document.key)
            leaf_bytes[document.key] = repository.store.get_text(document.key)

    assert set(repository.store.list_keys(layout.doc_prefix())) == expected_doc_keys
    assert repository.list_current() == []
    assert repository.list_raw() == []
    assert repository.list_rewrites() == []
    assert not any(
        key.endswith(".json") for key in repository.store.list_keys("rewrite")
    )
    return leaf_bytes, topic_bytes


def test_topic_appends_new_memory_h1_headings_without_front_matter() -> None:
    """TOPIC is the ordered union of leaf H1s and is the next routing catalog."""

    store = InMemoryObjectStore()
    llm = MultiHeadingTopicLLM()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-topic-test",
                fixed_prefix="inf_mem_test",
            )
        ),
        store_id="topic_headings",
        user_id="alice",
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="topic-test",
    )
    started_at = datetime(2026, 7, 29, tzinfo=timezone.utc)

    _write_current(
        flow,
        instance_id="round-1",
        created_at=started_at,
        sections={
            "Food Preferences": [(1700000001, "Likes apples.")],
            "Drink Preferences": [(1700000002, "Prefers tea.")],
        },
    )
    first = flow.maintain(MaintenanceRequest())
    topics = flow.maintainer.repository.list_directory_topics()
    memories = flow.maintainer.repository.list_memory_documents()

    assert (first.directories_created, first.memory_documents_created) == (1, 1)
    assert len(topics) == len(memories) == 1
    assert memories[0].content.startswith("# Food Preferences\n")
    assert "\n# Drink Preferences\n" in memories[0].content
    assert topics[0].content == "# Food Preferences\n# Drink Preferences"
    first_topic_bytes = flow.maintainer.repository.store.get_text(topics[0].key)

    _write_current(
        flow,
        instance_id="round-2",
        created_at=started_at + timedelta(minutes=1),
        sections={"Music Preferences": [(1700000101, "Likes jazz.")]},
    )
    second = flow.maintain(MaintenanceRequest())
    topic = flow.maintainer.repository.list_directory_topics()[0]
    memories = flow.maintainer.repository.list_memory_documents(topic.metadata.id)

    assert (second.directories_created, second.memory_documents_created) == (0, 1)
    assert len(memories) == 2
    assert topic.content == (
        "# Food Preferences\n# Drink Preferences\n# Music Preferences"
    )
    assert flow.maintainer.repository.store.get_text(topic.key) != first_topic_bytes
    assert llm.route_catalogs[0] == []
    assert llm.route_catalogs[1] == [
        {
            "directory_id": topic.metadata.id,
            "body": "# Food Preferences\n# Drink Preferences",
        }
    ]
    assert "---" not in str(llm.route_catalogs[1][0]["body"])


def test_maintenance_gets_all_document_summaries_from_single_route_call() -> None:
    store = InMemoryObjectStore()
    llm = ParallelMaintenanceLLM()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-parallel-summary-test",
                fixed_prefix="inf_mem_test",
            )
        ),
        store_id="parallel_summary",
        user_id="alice",
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="parallel-summary-test",
    )
    _write_current(
        flow,
        instance_id="round-1",
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        sections={
            f"Topic {index}": [(index, f"Fact {index}.")] for index in range(1, 5)
        },
    )

    result = flow.maintain(MaintenanceRequest(workers=4))

    assert result.memory_documents_created == 4
    assert llm.operations.count("route_directories") == 1
    assert "document_summary" not in llm.operations
    assert {
        document.metadata.summary
        for document in flow.maintainer.repository.list_memory_documents()
    } == {f"Fact {index}." for index in range(1, 5)}


def test_maintenance_removes_sequence_metadata_from_route_summary() -> None:
    class TaggedSummaryLLM(DocLifecycleLLM):
        def complete(self, request: LLMRequest) -> str:
            raw = super().complete(request)
            if request.operation != "route_directories":
                return raw
            plan = json.loads(raw)
            for assignment in plan["assignments"]:
                assignment["summary"] = (
                    f"<seq=999,source=AI> {assignment['summary']}"
                )
            return json.dumps(plan)

    store = InMemoryObjectStore()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-summary-sanitize-test",
                fixed_prefix="inf_mem_test",
            )
        ),
        store_id="summary_sanitize",
        user_id="alice",
        store=store,
        llm=TaggedSummaryLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    _write_current(
        flow,
        instance_id="round-1",
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        sections={"Preferences": [(1, "Likes tea.")]},
    )

    flow.maintain(MaintenanceRequest())
    summary = flow.maintainer.repository.list_memory_documents()[0].metadata.summary

    assert summary == "Likes tea."


def test_maintenance_parallelizes_independent_rewrite_batches() -> None:
    store = InMemoryObjectStore()
    llm = ParallelMaintenanceLLM()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-parallel-rewrite-test",
                fixed_prefix="inf_mem_test",
            ),
            maintenance=MaintenanceConfig(rewrite_batch_max_tokens=1),
        ),
        store_id="parallel_rewrite",
        user_id="alice",
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="parallel-rewrite-test",
    )
    started_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
    for index in range(4):
        _write_current(
            flow,
            instance_id=f"worker-{index}",
            created_at=started_at + timedelta(minutes=index),
            sections={f"Topic {index}": [(index + 1, f"Fact {index}.")]},
        )

    result = flow.maintain(MaintenanceRequest(workers=4))

    assert result.rewrite_documents == 4
    assert llm.max_active_rewrites >= 2


def test_maintenance_falls_back_after_structured_retry_budget() -> None:
    store = InMemoryObjectStore()
    llm = InvalidRewritePlanLLM(invalid_responses=2)
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-rewrite-retry-test",
                fixed_prefix="inf_mem_test",
            )
        ),
        store_id="rewrite_retry",
        user_id="alice",
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="rewrite-retry-test",
    )
    _write_current(
        flow,
        instance_id="round-1",
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        sections={"Preferences": [(1, "Likes tea.")]},
    )

    result = flow.maintain(MaintenanceRequest())

    assert result.memory_documents_created == 1
    assert llm.rewrite_attempts == 2
    assert "Likes tea." in flow.maintainer.repository.list_memory_documents()[0].content


def test_maintenance_preserves_facts_after_invalid_plan_retries_exhausted() -> None:
    store = InMemoryObjectStore()
    llm = InvalidRewritePlanLLM(invalid_responses=5)
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-rewrite-fallback-test",
                fixed_prefix="inf_mem_test",
            )
        ),
        store_id="rewrite_fallback",
        user_id="alice",
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="rewrite-fallback-test",
    )
    _write_current(
        flow,
        instance_id="round-1",
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        sections={"Preferences": [(1, "Likes tea.")]},
    )

    result = flow.maintain(MaintenanceRequest())

    assert result.memory_documents_created == 1
    assert llm.rewrite_attempts == 2
    assert "Likes tea." in flow.maintainer.repository.list_memory_documents()[0].content


def test_maintenance_falls_back_for_invalid_route_without_summary_call() -> None:
    store = InMemoryObjectStore()
    llm = InvalidRouteAndSummaryLLM()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-structured-fallback-test",
                fixed_prefix="inf_mem_test",
            )
        ),
        store_id="structured_fallback",
        user_id="alice",
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="structured-fallback-test",
    )
    _write_current(
        flow,
        instance_id="round-1",
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        sections={"Preferences": [(1, "Likes tea.")]},
    )

    result = flow.maintain(MaintenanceRequest())
    memory = flow.maintainer.repository.list_memory_documents()[0]

    assert result.memory_documents_created == 1
    assert llm.route_attempts == 2
    assert llm.summary_attempts == 0
    assert "Likes tea." in memory.content
    assert "Likes tea." in memory.metadata.summary


def test_maintenance_falls_back_after_rewrite_transport_retries_exhausted() -> None:
    store = InMemoryObjectStore()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(endpoint_url="http://unused.local", bucket="unused")
        ),
        store_id="rewrite_transport_fallback",
        user_id="alice",
        store=store,
        llm=RuntimeFailureLLM("rewrite_current"),
        metrics=FlowMetrics.create(enabled=False),
    )
    _write_current(
        flow,
        instance_id="round-1",
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        sections={"Preferences": [(1, "Likes tea.")]},
    )

    result = flow.maintain(MaintenanceRequest())

    assert result.memory_documents_created == 1
    assert "Likes tea." in flow.maintainer.repository.list_memory_documents()[0].content


def test_maintenance_falls_back_after_route_transport_retries_exhausted() -> None:
    store = InMemoryObjectStore()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(endpoint_url="http://unused.local", bucket="unused")
        ),
        store_id="route_transport_fallback",
        user_id="alice",
        store=store,
        llm=RuntimeFailureLLM("route_directories"),
        metrics=FlowMetrics.create(enabled=False),
    )
    _write_current(
        flow,
        instance_id="round-1",
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        sections={"Preferences": [(1, "Likes tea.")]},
    )

    result = flow.maintain(MaintenanceRequest())

    assert result.memory_documents_created == 1
    assert "Likes tea." in flow.maintainer.repository.list_memory_documents()[0].content


def test_doc_state_after_multiple_maintenance_rounds() -> None:
    """Doc remains append-only and internally consistent across three rounds."""
    store = InMemoryObjectStore()
    llm = DocLifecycleLLM()
    flow = MemFlow.create(
        MemFlowConfig(
            s3=S3Config(
                endpoint_url="http://unused.local",
                bucket="mem-flow-doc-test",
                fixed_prefix="inf_mem_test",
            )
        ),
        store_id="doc_lifecycle",
        user_id="alice",
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="doc-test",
    )
    started_at = datetime(2026, 7, 29, tzinfo=timezone.utc)

    _write_current(
        flow,
        instance_id="round-1",
        created_at=started_at,
        sections={
            "Preferences": [(1700000001, "Likes apples.")],
            "Pets": [(1700000002, "Has a cat named Momo.")],
        },
    )
    first = flow.maintain(MaintenanceRequest())
    assert (
        first.directories_created,
        first.memory_documents_created,
        first.directory_topics_updated,
    ) == (2, 2, 2)
    first_leaves, first_topics = _assert_doc_state(
        flow,
        {
            "Preferences": ["Likes apples."],
            "Pets": ["Has a cat named Momo."],
        },
    )
    first_directory_ids = {
        item.metadata.title: item.metadata.id
        for item in flow.maintainer.repository.list_directory_topics()
    }

    _write_current(
        flow,
        instance_id="round-2",
        created_at=started_at + timedelta(minutes=1),
        sections={
            "Preferences": [(1700000101, "Likes pears.")],
            "Work": [(1700000102, "Joined Acme.")],
        },
    )
    second = flow.maintain(MaintenanceRequest())
    assert (
        second.directories_created,
        second.memory_documents_created,
        second.directory_topics_updated,
    ) == (1, 2, 2)
    second_leaves, second_topics = _assert_doc_state(
        flow,
        {
            "Preferences": ["Likes apples.", "Likes pears."],
            "Pets": ["Has a cat named Momo."],
            "Work": ["Joined Acme."],
        },
    )
    assert all(
        flow.maintainer.repository.store.get_text(key) == value
        for key, value in first_leaves.items()
    )
    assert second_topics["Pets"] == first_topics["Pets"]
    assert second_topics["Preferences"] != first_topics["Preferences"]
    second_directory_ids = {
        item.metadata.title: item.metadata.id
        for item in flow.maintainer.repository.list_directory_topics()
    }
    assert {
        title: second_directory_ids[title] for title in first_directory_ids
    } == first_directory_ids

    _write_current(
        flow,
        instance_id="round-3-preferences",
        created_at=started_at + timedelta(minutes=2),
        sections={"Preferences": [(1700000201, "Prefers tea.")]},
    )
    _write_current(
        flow,
        instance_id="round-3-pets",
        created_at=started_at + timedelta(minutes=3),
        sections={"Pets": [(1700000202, "Momo visits the vet annually.")]},
    )
    third = flow.maintain(MaintenanceRequest())
    assert third.current_documents == 2
    assert third.rewrite_documents == 1
    assert (
        third.directories_created,
        third.memory_documents_created,
        third.directory_topics_updated,
    ) == (0, 2, 2)
    third_leaves, third_topics = _assert_doc_state(
        flow,
        {
            "Preferences": ["Likes apples.", "Likes pears.", "Prefers tea."],
            "Pets": [
                "Has a cat named Momo.",
                "Momo visits the vet annually.",
            ],
            "Work": ["Joined Acme."],
        },
    )
    assert all(
        flow.maintainer.repository.store.get_text(key) == value
        for key, value in second_leaves.items()
    )
    assert third_topics["Work"] == second_topics["Work"]
    assert third_topics["Preferences"] != second_topics["Preferences"]
    assert third_topics["Pets"] != second_topics["Pets"]
    assert len(third_leaves) == 6

    final_directory_ids = {
        item.metadata.title: item.metadata.id
        for item in flow.maintainer.repository.list_directory_topics()
    }
    assert final_directory_ids == second_directory_ids
    assert len({document_id for document_id in third_leaves}) == 6
    assert llm.operations.count("rewrite_current") == 3
    assert llm.operations.count("route_directories") == 3
    assert "document_summary" not in llm.operations
    assert "directory_topic" not in llm.operations
    assert not (
        {"split_plan", "update_topic", "merge_plan", "merge_topic"}
        & set(llm.operations)
    )
