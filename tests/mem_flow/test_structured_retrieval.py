"""Offline contracts for the rebuildable fact index and AUTO query path."""

from __future__ import annotations

from datetime import date

from mem_flow import (
    FlowMetrics,
    MemFlow,
    MemFlowConfig,
    MetricsConfig,
    RetrievalConfig,
    SearchRequest,
    SearchSource,
    SearchStrategy,
    StorageConfig,
)
from mem_flow.index import FactRelation, assign_fact_ids, project_documents
from mem_flow.models import DocumentMetadata, LLMRequest, MemoryDocument
from mem_flow.retriever.coverage import assess_coverage
from mem_flow.retriever.executor import execute_query_plan
from mem_flow.retriever.planner import QueryOperator, compile_query_plan
from mem_flow.retriever.retriever import _execution_is_high_confidence
from mem_flow.retriever.verifier import verify_evidence
from mem_flow.utils.paths import KeyLayout


class RejectingLLM:
    def complete(self, request: LLMRequest) -> str:
        raise AssertionError(
            f"structured answer unexpectedly called LLM: {request.operation}"
        )


def test_structured_execution_requires_a_small_evidence_boundary() -> None:
    assert _execution_is_high_confidence(
        plan_subgoal_count=1,
        executable_fact_count=2,
        execution_complete=True,
        execution_fact_count=2,
    )
    assert not _execution_is_high_confidence(
        plan_subgoal_count=1,
        executable_fact_count=20,
        execution_complete=True,
        execution_fact_count=20,
    )


def _flow(tmp_path) -> MemFlow:
    return MemFlow.create(
        MemFlowConfig(
            storage=StorageConfig(type="local", path=tmp_path),
            retrieval=RetrievalConfig(strategy=SearchStrategy.AUTO),
            metrics=MetricsConfig(enabled=False),
        ),
        store_id="structured",
        user_id="alice",
        llm=RejectingLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )


def _write_memory(
    flow: MemFlow,
    *,
    document_id: str,
    content: str,
    directory_id: str = "activities",
) -> None:
    key = KeyLayout().memory_key(directory_id, document_id)
    flow.retriever.repository.write(
        key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id=document_id,
                kind="memory",
                directory_id=directory_id,
                title="Activities",
            ),
            content=content,
        ),
    )


def _write_evidence(flow: MemFlow, evidence_id: str, content: str) -> None:
    flow.retriever.repository.write(
        KeyLayout().evidence_key(evidence_id),
        MemoryDocument(
            metadata=DocumentMetadata(id=evidence_id, kind="evidence"),
            content=content,
        ),
    )


def test_fact_ids_are_code_owned_stable_and_idempotent() -> None:
    content = (
        "# Purchases\n\n"
        "- <seq=1710000000,origin=EVIDENCE_A,observed=2024-03-09> "
        "The user bought a smoker."
    )

    first = assign_fact_ids(content)
    second = assign_fact_ids(first)

    assert first == second
    assert ",origin=EVIDENCE_A,fid=f_" in first
    assert first.count("fid=") == 1


def test_projection_keeps_dual_time_state_and_conservative_relations() -> None:
    content = assign_fact_ids(
        "# Preferences\n\n"
        "- <seq=1,time=2024-01-02,observed=2024-01-03> The user likes jazz music.\n"
        "- <seq=2,time=2024-01-02,observed=2024-02-01> The user likes jazz music.\n"
        "- <seq=3,observed=2024-03-01> The user now likes classical music instead.\n"
        "- <seq=4,time=2025-04-01> The user will attend a music workshop."
    )
    snapshot = project_documents(
        [
            MemoryDocument(
                metadata=DocumentMetadata(id="memory_preferences", kind="memory"),
                content=content,
                key="doc/preferences/memory_preferences.md",
            )
        ]
    )

    assert snapshot.facts[0].event_time is not None
    assert snapshot.facts[0].observed_at is not None
    assert snapshot.facts[-1].status.value == "planned"
    assert any(relation.kind == "same_event" for relation in snapshot.relations)
    assert any(relation.kind == "supersedes" for relation in snapshot.relations)


def test_generic_executor_deduplicates_and_runs_sum_and_elapsed() -> None:
    content = assign_fact_ids(
        "# Events\n\n"
        "- <seq=1,time=2024-01-01> The user paid $25 for concert tickets.\n"
        "- <seq=2,time=2024-01-01> The user paid $25 for concert tickets.\n"
        "- <seq=3,time=2024-01-07> The user paid $15 for lunch.\n"
        "- <seq=4,time=2024-01-10> The user attended a gardening workshop.\n"
        "- <seq=5,time=2024-01-16> The user planted tomato saplings."
    )
    snapshot = project_documents(
        [
            MemoryDocument(
                metadata=DocumentMetadata(id="memory_events", kind="memory"),
                content=content,
                key="doc/events/memory_events.md",
            )
        ]
    )
    sum_plan = compile_query_plan("What was the total I spent on tickets and lunch?")
    elapsed_plan = compile_query_plan(
        "How many days passed between the gardening workshop and planting tomato saplings?"
    )

    summed = execute_query_plan(sum_plan, snapshot.facts[:3], snapshot.relations)
    elapsed = execute_query_plan(elapsed_plan, snapshot.facts[3:], snapshot.relations)

    assert any(step.operator == QueryOperator.SUM for step in sum_plan.steps)
    assert summed.render() == "40 USD"
    assert elapsed.render() == "6 days"


def test_generic_executor_resolves_before_and_after_boundaries() -> None:
    content = assign_fact_ids(
        "# Appliances\n\n"
        "- <seq=1,time=2024-01-01> The user owned a toaster oven.\n"
        "- <seq=2,time=2024-03-01> The user bought an air fryer.\n"
        "- <seq=3,time=2024-04-01> The user donated the toaster oven."
    )
    snapshot = project_documents(
        [
            MemoryDocument(
                metadata=DocumentMetadata(id="memory_appliances", kind="memory"),
                content=content,
                key="doc/home/memory_appliances.md",
            )
        ]
    )

    before = execute_query_plan(
        compile_query_plan("What did I own before the air fryer?"),
        snapshot.facts,
        snapshot.relations,
    )
    after = execute_query_plan(
        compile_query_plan("What happened after I bought the air fryer?"),
        snapshot.facts,
        snapshot.relations,
    )

    assert before.complete is True
    assert "toaster oven" in (before.render() or "")
    assert after.complete is True
    assert "donated" in (after.render() or "")


def test_count_excludes_plans_and_cancellations_unless_requested() -> None:
    snapshot = project_documents(
        [
            MemoryDocument(
                metadata=DocumentMetadata(id="memory_status", kind="memory"),
                content=assign_fact_ids(
                    "# Workshops\n\n"
                    "- <seq=1> The user attended a pottery workshop.\n"
                    "- <seq=2> The user will attend a painting workshop.\n"
                    "- <seq=3> The user cancelled a gardening workshop."
                ),
                key="doc/workshops/memory_status.md",
            )
        ]
    )

    completed = execute_query_plan(
        compile_query_plan("How many workshops did I attend?"),
        snapshot.facts,
        snapshot.relations,
    )
    planned = execute_query_plan(
        compile_query_plan("How many workshops did I plan?"),
        snapshot.facts,
        snapshot.relations,
    )

    assert completed.render() == "1"
    assert planned.render() == "1"


def test_content_addressed_index_validates_and_detects_stale_markdown(tmp_path) -> None:
    flow = _flow(tmp_path)
    _write_memory(
        flow,
        document_id="memory_workshop",
        content=assign_fact_ids(
            "# Workshops\n\n"
            "- <seq=1,time=2024-01-10> The user attended a pottery workshop."
        ),
    )

    rebuilt = flow.rebuild_index()
    valid = flow.validate_index()
    document = flow.retriever.repository.list_memory_documents()[0]
    flow.retriever.repository.write_body(
        document.key,
        document.content.replace("pottery", "painting"),
    )

    assert rebuilt["persisted"] is True
    assert rebuilt["facts"] == 1
    assert valid["valid"] is True
    assert flow.validate_index()["valid"] is False
    assert flow.rebuild_index()["manifest_digest"] != rebuilt["manifest_digest"]


def test_index_validation_rejects_corrupted_fact_artifact(tmp_path) -> None:
    flow = _flow(tmp_path)
    _write_memory(
        flow,
        document_id="memory_integrity",
        content=assign_fact_ids(
            "# Integrity\n\n- <seq=1> The user likes immutable evidence."
        ),
    )
    flow.rebuild_index()
    snapshot = flow.retriever.index_store.load(
        flow.retriever.repository.list_memory_documents()
    )
    assert snapshot and snapshot.manifest
    artifact_key = snapshot.manifest.sources[0].artifact_key

    flow.retriever.repository.store.put_text(artifact_key, "[]")

    assert flow.validate_index()["valid"] is False


def test_assistant_only_evidence_requires_an_explicit_assistant_question() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_AI", kind="evidence"),
        content="[ASSISTANT]\nI recommend an electric smoker.",
        key="evidence/EVIDENCE_AI.md",
    )
    memory = MemoryDocument(
        metadata=DocumentMetadata(id="memory_advice", kind="memory"),
        content=assign_fact_ids(
            "# Advice\n\n"
            "- <seq=1,origin=EVIDENCE_AI,source=AI> "
            "The assistant recommended an electric smoker."
        ),
        key="doc/advice/memory_advice.md",
    )
    snapshot = project_documents([memory])
    plan = compile_query_plan("What smoker did the assistant recommend?")
    coverage = assess_coverage(plan, snapshot.facts)

    rejected = verify_evidence(
        snapshot.facts,
        [memory, evidence],
        coverage,
        threshold=0.7,
    )
    accepted = verify_evidence(
        snapshot.facts,
        [memory, evidence],
        coverage,
        threshold=0.7,
        allow_assistant_only=True,
    )

    assert rejected.sufficient is False
    assert rejected.reason == "assistant_suggestion_only"
    assert accepted.sufficient is True


def test_unresolved_explicit_preference_conflict_forces_abstention() -> None:
    memory = MemoryDocument(
        metadata=DocumentMetadata(id="memory_conflict", kind="memory"),
        content=assign_fact_ids(
            "# Preferences\n\n"
            "- <seq=1,origin=EVIDENCE_LIKE> The user likes jazz.\n"
            "- <seq=2,origin=EVIDENCE_AVOID> The user avoids jazz."
        ),
        key="doc/preferences/memory_conflict.md",
    )
    evidence = [
        MemoryDocument(
            metadata=DocumentMetadata(id=evidence_id, kind="evidence"),
            content="[USER]\n" + text,
            key=f"evidence/{evidence_id}.md",
        )
        for evidence_id, text in (
            ("EVIDENCE_LIKE", "I like jazz."),
            ("EVIDENCE_AVOID", "I avoid jazz."),
        )
    ]
    snapshot = project_documents([memory])
    plan = compile_query_plan("Does the user like jazz?")
    coverage = assess_coverage(plan, snapshot.facts)

    verified = verify_evidence(
        snapshot.facts,
        [memory, *evidence],
        coverage,
        threshold=0.7,
        relations=snapshot.relations,
    )

    assert any(relation.kind == "contradicts" for relation in snapshot.relations)
    assert verified.sufficient is False
    assert verified.reason.startswith("unresolved_conflicts:")


def test_direct_synthesis_can_resolve_reported_relation_conflicts() -> None:
    memory = MemoryDocument(
        metadata=DocumentMetadata(id="memory_colors", kind="memory"),
        content=assign_fact_ids(
            "# Home\n\n"
            "- <seq=1,origin=EVIDENCE_OLD> The old hallway was blue.\n"
            "- <seq=2,origin=EVIDENCE_NEW> The user repainted the bedroom light gray."
        ),
        key="doc/home/memory_colors.md",
    )
    evidence = [
        MemoryDocument(
            metadata=DocumentMetadata(id=evidence_id, kind="evidence"),
            content="[USER]\n" + text,
            key=f"evidence/{evidence_id}.md",
        )
        for evidence_id, text in (
            ("EVIDENCE_OLD", "The old hallway was blue."),
            ("EVIDENCE_NEW", "I repainted the bedroom light gray."),
        )
    ]
    snapshot = project_documents([memory])
    plan = compile_query_plan("What color did I repaint my bedroom?")
    coverage = assess_coverage(plan, snapshot.facts)
    conflicting_relations = [
        relation for relation in snapshot.relations if relation.kind == "contradicts"
    ]
    if not conflicting_relations:
        conflicting_relations = [
            FactRelation(
                relation_id="rel_test",
                kind="contradicts",
                source_fact_id=snapshot.facts[0].fact_id,
                target_fact_id=snapshot.facts[1].fact_id,
                confidence=1.0,
            )
        ]

    verified = verify_evidence(
        snapshot.facts,
        [memory, *evidence],
        coverage,
        threshold=0.7,
        relations=conflicting_relations,
        block_on_conflicts=False,
    )

    assert verified.sufficient is True
    assert verified.reason == ""
    assert coverage.unresolved_conflicts


def test_auto_route_executes_cross_document_count_with_origin_verification(
    tmp_path,
) -> None:
    flow = _flow(tmp_path)
    _write_evidence(flow, "EVIDENCE_ART", "[USER]\nI attended a pottery workshop.")
    _write_evidence(flow, "EVIDENCE_GARDEN", "[USER]\nI attended a gardening workshop.")
    _write_memory(
        flow,
        document_id="memory_art",
        content=assign_fact_ids(
            "# Art\n\n"
            "- <seq=1,origin=EVIDENCE_ART,time=2024-01-10> "
            "The user attended a pottery workshop."
        ),
        directory_id="art",
    )
    _write_memory(
        flow,
        document_id="memory_garden",
        content=assign_fact_ids(
            "# Gardening\n\n"
            "- <seq=2,origin=EVIDENCE_GARDEN,time=2024-02-12> "
            "The user attended a gardening workshop."
        ),
        directory_id="garden",
    )
    flow.rebuild_index()

    result = flow.search(
        SearchRequest(
            query="How many workshops did I attend?",
            strategy=SearchStrategy.AUTO,
            sources={SearchSource.DOC, SearchSource.EVIDENCE},
            limit=10,
            answer=True,
            as_of=date(2024, 3, 1),
            debug=True,
        )
    )

    assert result.route == "structured"
    assert result.answer == "2"
    assert result.coverage and result.coverage["complete"] is True
    assert result.execution and result.execution["complete"] is True
    assert {hit.id for hit in result.hits} >= {
        "memory_art",
        "memory_garden",
        "EVIDENCE_ART",
        "EVIDENCE_GARDEN",
    }
