"""End-to-end unit test for all mem_flow modules.

Default execution is offline. Optional flags replace either fake independently:

    uv run pytest tests/mem_flow/test_mem_flow.py -v --mem-flow-real-llm
    uv run pytest tests/mem_flow/test_mem_flow.py -v --mem-flow-real-s3
    uv run pytest tests/mem_flow/test_mem_flow.py -v --mem-flow-real-llm --mem-flow-real-s3
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import yaml
from prometheus_client import CollectorRegistry, generate_latest

import mem_flow.prompts as prompts_module
import mem_flow.retriever.agentic as agentic_module
import mem_flow.retriever.guidance as guidance_module
import mem_flow.retriever.retriever as retriever_module
from mem_flow import (
    ChatMessage,
    DocLineUpdateRequest,
    DocReadRequest,
    ExtractionConfig,
    ExtractionRequest,
    FlowMetrics,
    LLMConfig,
    MaintenanceConfig,
    MaintenanceRequest,
    MemFlow,
    MemFlowConfig,
    MemoryScope,
    MetricsConfig,
    RetrievalConfig,
    S3Config,
    SearchRequest,
    SearchSource,
    SearchStrategy,
)
from mem_flow.hierarchy import DirectoryTopicBuilder, MarkdownFact, extract_h1_headings
from mem_flow.llm import OpenAIFlowLLM
from mem_flow.maintainer.maintainer import _ContentReferencePlan, _render_content_plan
from mem_flow.index.projector import project_document
from mem_flow.models import (
    BEIJING_TIMEZONE,
    DocumentMetadata,
    DocumentSelection,
    LLMRequest,
    MemoryDocument,
    compact_beijing_timestamp,
)
from mem_flow.prompts import (
    AGENTIC_ANSWER_PROMPT,
    ANSWER_PROMPT,
    DIRECTORY_ROUTE_PROMPT,
    DOCUMENT_SUMMARY_PROMPT,
    EXTRACT_PROMPT,
    REWRITE_PROMPT,
)
from mem_flow.retriever.agentic import AgenticRetriever
from mem_flow.retriever.agentic import _SYSTEM_PROMPT as AGENTIC_SYSTEM_PROMPT
from mem_flow.retriever.agentic import _expand_focus_query
from mem_flow.retriever.agentic import _has_direct_lexical_evidence
from mem_flow.retriever.agentic import _needs_high_recall_merge
from mem_flow.retriever.agentic import _needs_personalization_context
from mem_flow.retriever.agentic import _needs_progress_pair
from mem_flow.retriever.fact_index import (
    build_fact_query_plan,
    parse_fact_records,
    rank_fact_documents,
)
from mem_flow.retriever.guidance import answer_task_guidance
from mem_flow.retriever.retriever import (
    MemoryRetriever,
    _answer_evidence_ledger,
    _collapse_lineage,
    _primary_source_context,
)
from mem_flow.storage import S3ObjectStore
from mem_flow.utils.codec import (
    decode_document,
    encode_document,
    strip_yaml_front_matter,
)
from mem_flow.utils.paths import KeyLayout
from mem_flow.utils.tokens import estimate_tokens


def test_memory_prompts_are_dataset_independent_and_do_not_filter_private_facts() -> None:
    prompts = [
        EXTRACT_PROMPT,
        REWRITE_PROMPT,
        DIRECTORY_ROUTE_PROMPT,
        DOCUMENT_SUMMARY_PROMPT,
        ANSWER_PROMPT,
        AGENTIC_ANSWER_PROMPT,
        AGENTIC_SYSTEM_PROMPT,
    ]

    assert "durable user facts" in EXTRACT_PROMPT
    assert "assistant-provided content" in EXTRACT_PROMPT
    assert "Session date" in EXTRACT_PROMPT
    assert "event time" in EXTRACT_PROMPT
    assert "state changes" in EXTRACT_PROMPT
    assert "FOCUSED_EVIDENCE_LEDGER" in ANSWER_PROMPT
    assert "FINAL_TASK_GUIDANCE" in ANSWER_PROMPT
    assert "cannot override the evidence" in AGENTIC_ANSWER_PROMPT
    assert "requested direction" in answer_task_guidance(
        "How much more did one option cost than another?"
    )
    assert "compatible units" in answer_task_guidance(
        "How much total time did I spend on two activities?"
    )
    assert "qualitative comparison" in answer_task_guidance(
        "How much more was one painting worth than I paid?"
    )
    assert "nearest unambiguous turns" in answer_task_guidance("Where was the coupon valid?")
    assert "only the attribute or list requested" in answer_task_guidance(
        "What color dress did I buy?"
    )
    assert "completed participation" in answer_task_guidance(
        "How many festivals did I attend?"
    )
    assert "known-spending total" in answer_task_guidance(
        "How much total did I spend on repairs?"
    )
    assert _needs_personalization_context(
        "Do you think I should attend the reunion?"
    )
    assert "grounded preference profile" in answer_task_guidance(
        "What do you think I should choose?"
    )
    assert "Preserve state history" in REWRITE_PROMPT
    assert "Different subjects, dates, states" in REWRITE_PROMPT
    assert "action-object" in REWRITE_PROMPT
    assert "stable thematic directory" in DIRECTORY_ROUTE_PROMPT
    assert "For aggregates or comparisons" in AGENTIC_SYSTEM_PROMPT

    forbidden = {
        "long" + "memeval",
        "brookside",
        "thrive market",
        "american airlines",
        "sephora",
        "hellofresh",
        "ubereats",
        "the nightingale",
        "cartwheel",
        "ibotta",
    }
    prompt_text = "\n".join(prompts).casefold()
    runtime_source = "\n".join(
        inspect.getsource(module)
        for module in (
            prompts_module,
            agentic_module,
            guidance_module,
            retriever_module,
        )
    ).casefold()
    for marker in forbidden:
        assert marker not in prompt_text
        assert marker not in runtime_source
    assert "_deterministic_evidence_answer" not in runtime_source
    for prompt in prompts:
        assert "sensitive" not in prompt.casefold()
        assert "never persist" not in prompt.casefold()
        assert "sensitive_fact_ids" not in prompt


def test_memory_prompts_use_unified_database_record_contract() -> None:
    assert "supplied conversation" in EXTRACT_PROMPT
    assert "collection of atomic records" in EXTRACT_PROMPT
    assert "exact subjects" in EXTRACT_PROMPT
    assert "facts as immutable rows" in REWRITE_PROMPT
    assert "provenance" in REWRITE_PROMPT


def test_rewrite_plan_cannot_discard_distinct_state_history() -> None:
    facts = [
        MarkdownFact(
            id="f0001",
            source_id="raw-a",
            heading="Exercise",
            markdown="- <seq=1,time=2023-01-01> User attended yoga twice a week.",
        ),
        MarkdownFact(
            id="f0002",
            source_id="raw-b",
            heading="Exercise",
            markdown=(
                "- <seq=2,time=2023-02-01> User now attends yoga three times a week."
            ),
        ),
    ]
    plan = _ContentReferencePlan.model_validate(
        {
            "topics": [{"title": "Exercise", "fact_ids": ["f0002"]}],
            "duplicates": [{"discarded_id": "f0001", "retained_id": "f0002"}],
        }
    )

    rendered = _render_content_plan(plan, facts)

    assert "twice a week" in rendered
    assert "three times a week" in rendered


def test_rewrite_plan_can_deduplicate_restatement_with_conflicting_inferred_dates() -> (
    None
):
    facts = [
        MarkdownFact(
            id="f0001",
            source_id="raw-a",
            heading="Baking",
            markdown=(
                "- <seq=1,time=2023-05-18> User baked cookies last Thursday "
                "using the convection setting; they were crisp outside and chewy inside."
            ),
        ),
        MarkdownFact(
            id="f0002",
            source_id="raw-b",
            heading="Baking",
            markdown=(
                "- <seq=2,time=2023-05-25> User used the convection setting to bake "
                "cookies that were crisp outside and chewy inside."
            ),
        ),
    ]
    plan = _ContentReferencePlan.model_validate(
        {
            "topics": [{"title": "Baking", "fact_ids": ["f0002"]}],
            "duplicates": [{"discarded_id": "f0001", "retained_id": "f0002"}],
        }
    )

    rendered = _render_content_plan(plan, facts)

    assert "2023-05-25" in rendered
    assert "2023-05-18" not in rendered


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How many properties did I view?", True),
        ("How much time did I spend in total?", True),
        ("Which grocery store did I spend the most money at?", True),
        ("Which hotel had the lowest nightly price?", True),
        ("What is the order of the museums from earliest to latest?", True),
        ("Which device did I set up first, the thermostat or router?", True),
        ("Which streaming service did I start using most recently?", True),
        ("Which event did I attend a week ago?", True),
        ("Which relative's life event did I attend a week ago?", True),
        ("How many followers do I have currently?", True),
        ("Did I exercise more frequently than previously?", True),
        ("What did I own before the air fryer?", True),
        ("Which game did I beat?", False),
    ],
)
def test_agentic_detects_queries_that_need_high_recall_merge(
    query: str, expected: bool
) -> None:
    assert _needs_high_recall_merge(query) is expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How many points do I need to earn to redeem a free product?", True),
        ("How much is left to reach my savings goal?", True),
        ("How many art events did I attend?", False),
    ],
)
def test_agentic_detects_progress_pair_queries(query: str, expected: bool) -> None:
    assert _needs_progress_pair(query) is expected


class InMemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}
        self.deleted_objects: dict[str, str] = {}

    def get_text(self, key: str) -> str:
        return self.objects[key]

    def put_text(self, key: str, content: str) -> None:
        self.objects[key] = content

    def delete(self, key: str) -> None:
        content = self.objects.pop(key, None)
        if content is not None:
            self.deleted_objects[key] = content

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        return sorted(key for key in self.objects if key.startswith(prefix))


class FakeFlowLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.extract_count = 0
        self.rewrite_sources: list[list[str]] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request.operation)
        prompt = request.messages[-1].content
        if request.operation == "extract":
            self.extract_count += 1
            timestamp = re.search(r"<seq=(\d{10})", prompt)
            assert timestamp, "extract prompt must provide a seconds-level timestamp"
            seq = timestamp.group(1)
            extractions = [
                (
                    "# Identity and preferences\n\n"
                    f"- <seq={seq}> The user's name is Alice.\n"
                    f"- <seq={seq},label=leisure> The user likes apples and jazz.\n"
                    f"- <seq={seq},source=AI> The user's preferred language is Chinese."
                ),
                (
                    "# Travel and health\n\n"
                    f"- <seq={seq},time=2026-10-12,label=travel> The user will visit Kyoto.\n"
                    f"- <seq={seq}> The user is allergic to peanuts."
                ),
                (
                    "# Work and relationships\n\n"
                    f"- <seq={seq},time=2024-03> The user joined Acme.\n"
                    f"- <seq={seq}> Bob is the user's emergency contact.\n"
                    f"- <seq={seq}> Notes are stored in C:\\Users\\Alice\\memory."
                ),
            ]
            return extractions[(self.extract_count - 1) % len(extractions)]
        if request.operation == "rewrite_current":
            facts = self._facts(prompt)
            self.rewrite_sources.append(
                list(dict.fromkeys(fact["source_id"] for fact in facts))
            )
            return json.dumps(self._content_reference_plan(prompt))
        if request.operation == "route_directories":
            facts = self._facts(prompt)
            catalog_text = prompt.split("DIRECTORY_CATALOG:\n", 1)[1].split(
                "\n\nFACTS_JSON:", 1
            )[0]
            catalog = json.loads(catalog_text)
            existing_by_title = {
                title_match.group(1): item["directory_id"]
                for item in catalog
                if (
                    title_match := re.search(
                        r"^#\s+(.+?)\s*$", str(item["body"]), re.MULTILINE
                    )
                )
            }
            by_heading: dict[str, list[str]] = {}
            for fact in facts:
                by_heading.setdefault(fact["heading"], []).append(fact["id"])
            return json.dumps(
                {
                    "assignments": [
                        {
                            "directory_id": existing_by_title.get(heading),
                            "new_directory_title": (
                                None if heading in existing_by_title else heading
                            ),
                            "fact_ids": fact_ids,
                            "summary": " ".join(
                                re.sub(r"^[-*+]\s+<[^>]+>\s*", "", fact["markdown"])
                                for fact in facts
                                if fact["id"] in fact_ids
                            ),
                        }
                        for heading, fact_ids in by_heading.items()
                    ]
                }
            )
        if request.operation == "document_summary":
            content = prompt.split("MEMORY_DOCUMENT:\n", 1)[-1]
            summary = re.sub(r"<[^>]*seq=[^>]+>\s*", "", content)
            summary = " ".join(
                line.strip("#- ") for line in summary.splitlines() if line.strip()
            )
            return json.dumps({"summary": summary[:240]})
        if request.operation == "search_scope":
            working_text = prompt.split("WORKING_DOCUMENTS:\n", 1)[1].split(
                "\nDIRECTORIES:\n", 1
            )[0]
            directories_text = prompt.split("\nDIRECTORIES:\n", 1)[1]
            working = json.loads(working_text)
            directories = json.loads(directories_text)
            return json.dumps(
                {
                    "working_document_ids": [item["id"] for item in working],
                    "directory_ids": [item["id"] for item in directories],
                }
            )
        if request.operation == "search_documents":
            catalog = json.loads(prompt.split("DOCUMENT_CATALOG:\n", 1)[-1])
            return json.dumps({"document_ids": [item["id"] for item in catalog]})
        if request.operation == "search_answer":
            return "The stored memories contain the requested user information."
        raise AssertionError(f"unexpected operation: {request.operation}")

    @staticmethod
    def _facts(prompt: str) -> list[dict[str, str]]:
        return json.loads(prompt.split("FACTS_JSON:\n", 1)[-1])

    def _content_reference_plan(self, prompt: str) -> dict[str, object]:
        facts = self._facts(prompt)
        by_heading: dict[str, list[str]] = {}
        for fact in facts:
            by_heading.setdefault(fact["heading"], []).append(fact["id"])
        return {
            "topics": [
                {"title": heading, "fact_ids": fact_ids}
                for heading, fact_ids in by_heading.items()
            ],
            "duplicates": [],
        }


def test_directory_topic_collects_unique_memory_h1_headings() -> None:
    result = DirectoryTopicBuilder().build(
        directory_title="  ## 偏好  ",
        existing_content="# 饮食偏好",
        document_contents=[
            "# 饮食偏好\n\n- 喜欢苹果。\n\n# 饮品偏好\n\n- 喜欢茶。",
            "# 饮食偏好\n\n- 喜欢梨。\n\n```markdown\n# 非文档标题\n```",
        ],
    )

    assert result.title == "偏好"
    assert result.summary == "偏好"
    assert result.headings == ["饮食偏好", "饮品偏好"]
    assert result.body == "# 饮食偏好\n# 饮品偏好"
    assert "喜欢苹果" not in result.body
    assert "非文档标题" not in result.body
    assert extract_h1_headings("# C#\n~~~markdown\n# 围栏内标题\n~~~") == ["C#"]


def test_agentic_selection_uses_structured_output_and_tool_evidence() -> None:
    structured, structured_source = AgenticRetriever._selection_from_result(
        {
            "structured_response": DocumentSelection(
                document_ids=["memory_structured"]
            ),
            "messages": [],
        }
    )
    assert structured.document_ids == ["memory_structured"]
    assert structured_source == "structured_response"

    content, content_source = AgenticRetriever._selection_from_result(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": '{"document_ids":["memory_content"]}',
                        }
                    ],
                }
            ]
        }
    )
    assert content.document_ids == ["memory_content"]
    assert content_source == "final_content"

    recovered, recovered_source = AgenticRetriever._selection_from_result(
        {
            "messages": [
                {"role": "assistant", "content": "\n"},
                {
                    "role": "tool",
                    "name": "search",
                    "content": json.dumps(
                        [
                            {"document_id": "memory_search"},
                            {"document_id": "memory_read"},
                        ]
                    ),
                },
                {
                    "role": "tool",
                    "name": "read_lines",
                    "content": json.dumps(
                        {"document_id": "memory_read", "lines": "1: fact"}
                    ),
                },
                {"role": "assistant", "content": ""},
            ]
        }
    )
    assert recovered.document_ids == ["memory_read", "memory_search"]
    assert recovered_source == "tool_evidence"


def test_agentic_recursion_limit_keeps_latest_tool_state() -> None:
    from langgraph.errors import GraphRecursionError

    class BoundedAgent:
        def stream(self, *_args, **_kwargs):
            yield {
                "messages": [
                    {
                        "role": "tool",
                        "name": "read_lines",
                        "content": json.dumps(
                            {"document_id": "memory_verified", "lines": "1: fact"}
                        ),
                    }
                ]
            }
            raise GraphRecursionError("bounded")

    result, exhausted = AgenticRetriever._stream_agent_result(
        BoundedAgent(),
        {"messages": []},
        recursion_limit=4,
    )

    assert exhausted is True
    selection, source = AgenticRetriever._selection_from_result(result)
    assert selection.document_ids == ["memory_verified"]
    assert source == "tool_evidence"


def test_agentic_focuses_answer_context_on_matching_fact_lines() -> None:
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_album", kind="memory"),
        content=(
            "# Unrelated\n\n"
            + "\n".join(f"- Generic unrelated fact {index}." for index in range(50))
            + "\n\n# Music\n\n"
            "- The signed debut album poster was limited to 500 copies worldwide.\n"
            "- The user also owns a turntable."
        ),
        key="doc/music/memory_album.md",
    )

    focused = AgenticRetriever._focus_document_for_answer(
        "How many copies of the debut album were released worldwide?",
        document,
        max_lines=3,
    )

    assert "500 copies worldwide" in focused.content
    assert len(focused.content.splitlines()) <= 6


def test_agentic_focus_keeps_local_answer_context() -> None:
    document = MemoryDocument(
        metadata=DocumentMetadata(id="evidence_coupon", kind="evidence"),
        content=(
            "# Coupon\n\n[ASSISTANT]\nThe Cartwheel app saves money at Target.\n\n"
            "[USER]\nI redeemed a $5 coupon on coffee creamer.\n\n"
            "[ASSISTANT]\nTarget sends coupons to email subscribers."
        ),
        key="evidence/EVIDENCE_coupon.md",
    )

    focused = AgenticRetriever._focus_document_for_answer(
        "Where did I redeem a $5 coupon on coffee creamer?",
        document,
        max_lines=6,
    )

    assert "redeemed a $5 coupon" in focused.content
    assert "Target" in focused.content


def test_agentic_focus_expands_generic_action_and_state_variants() -> None:
    expanded = _expand_focus_query(
        "How much did I spend on purchases after my subscription changed?"
    ).casefold()

    assert "paid" in expanded
    assert "bought" in expanded
    assert "active" in expanded
    assert "updated" in expanded
    category_expanded = _expand_focus_query(
        "Order the sports events I participated in during the past month."
    ).casefold()
    assert "triathlons" in category_expanded
    assert "took part" in category_expanded


def test_agentic_detects_direct_lexical_fact_without_domain_hints() -> None:
    relevant = MemoryDocument(
        metadata=DocumentMetadata(id="record", kind="memory"),
        content="User completed the Zephyr protocol last weekend.",
    )
    unrelated = MemoryDocument(
        metadata=DocumentMetadata(id="other", kind="memory"),
        content="User attended a meeting last weekend.",
    )

    query = "Question date: 2024/05/30\nQuestion: What Zephyr protocol did I complete last weekend?"
    assert _has_direct_lexical_evidence(query, [relevant])
    assert not _has_direct_lexical_evidence(query, [unrelated])
    recommendation = _expand_focus_query("Can you recommend an option?").casefold()
    assert "suggestion" in recommendation
    acquisition = _expand_focus_query("What did I purchase?").casefold()
    assert "bought" in acquisition
    completed_game = _expand_focus_query("What game did I beat?").casefold()
    assert "dlc" in completed_game
    assert "finished" in completed_game
    inherited_item = _expand_focus_query("Which antique did I inherit?").casefold()
    assert "heirloom" in inherited_item
    assert "passed down" in inherited_item
    flight_history = _expand_focus_query("Which airlines did I use?").casefold()
    assert "flights" in flight_history
    assert "flew" in flight_history
    publication_advice = _expand_focus_query(
        "Can you recommend publications or conferences?"
    ).casefold()
    assert "papers" in publication_advice
    assert "research" in publication_advice
    phone_advice = _expand_focus_query("Any tips for my phone battery?").casefold()
    assert "power bank" in phone_advice
    drink_advice = _expand_focus_query("Which cocktail should I make?").casefold()
    assert "mixology" in drink_advice


def test_agentic_balanced_fallback_reserves_each_entity() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_age_topic", kind="evidence"),
            content=(
                "# Source conversation\n\n[USER]\n"
                "I need skincare advice for my age group."
            ),
            key="evidence/EVIDENCE_age_topic.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_yoga", kind="memory"),
            content="# Yoga\n\n- User planned many yoga sessions.",
            key="doc/fitness/memory_yoga.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_jog", kind="memory"),
            content="# Jogging\n\n- User completed a 30-minute jog.",
            key="doc/fitness/memory_jog.md",
        ),
        *[
            MemoryDocument(
                metadata=DocumentMetadata(id=f"memory_noise_{index}", kind="memory"),
                content="# Yoga\n\n- Yoga yoga yoga hours week.",
                key=f"doc/noise/memory_noise_{index}.md",
            )
            for index in range(8)
        ],
    ]
    retriever = AgenticRetriever(
        documents=documents,
        llm=FakeFlowLLM(),
        config=RetrievalConfig(),
        limit=8,
    )

    selected = retriever._balanced_fallback(
        "How many hours of jogging and yoga did I do last week?"
    )

    assert "memory_jog" in {document.metadata.id for document in selected}


def test_agentic_fallback_reserves_atomic_source_evidence() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_music", kind="memory"),
            content="# Music\n\n" + ("music service recommendations " * 100),
            key="doc/music/memory_music.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_spotify", kind="evidence"),
            content=(
                "# Source conversation\n\n[USER]\n"
                "I've been listening to songs on Spotify lately."
            ),
            key="evidence/EVIDENCE_spotify.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_repair", kind="evidence"),
            content="# Source conversation\n\n[USER]\nI serviced my watch.",
            key="evidence/EVIDENCE_repair.md",
        ),
    ]
    retriever = AgenticRetriever(
        documents=documents,
        llm=FakeFlowLLM(),
        config=RetrievalConfig(),
        limit=3,
    )

    selected = retriever._fallback(
        "What music streaming service have I been using lately?"
    )

    assert selected[0].metadata.id == "EVIDENCE_spotify"


def test_agentic_fallback_keeps_broad_leaves_first_for_aggregation() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_projects", kind="memory"),
            content=(
                "# Projects\n\n- <seq=1> User led the Atlas project.\n"
                "- <seq=2> User currently leads the Beacon project."
            ),
            key="doc/work/memory_projects.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_project", kind="evidence"),
            content="# Source conversation\n\n[USER]\nI led the Atlas project.",
            key="evidence/EVIDENCE_project.md",
        ),
    ]
    retriever = AgenticRetriever(
        documents=documents,
        llm=FakeFlowLLM(),
        config=RetrievalConfig(),
        limit=2,
    )

    selected = retriever._fallback(
        "How many projects have I led or am currently leading?"
    )

    assert selected[0].metadata.id == "memory_projects"


def test_answer_evidence_ledger_reranks_facts_across_documents() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_workshop_a", kind="memory"),
            content=(
                "# Workshops\n\n"
                "- User paid $200 for a writing workshop.\n"
                "- User paid $20 for a mindfulness workshop.\n"
                "- User attended a free photography workshop."
            ),
            key="doc/workshops/memory_workshop_a.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_workshop_b", kind="memory"),
            content=(
                "# Marketing\n\n"
                "- User attended a digital marketing workshop, paying $500.\n"
                "- Workshop directory can filter by cost: $100-$500 or $500-$1000.\n"
                "- Additional fees: $500 for initial approval.\n"
                "- Generic unrelated marketing note."
            ),
            key="doc/marketing/memory_workshop_b.md",
        ),
    ]

    ledger = _answer_evidence_ledger(
        "How much total money did I spend on attending workshops?", documents
    )

    assert "$500" in ledger
    assert "$200" in ledger
    assert "$20" in ledger
    assert "free photography workshop" in ledger


def test_answer_evidence_ledger_keeps_contextual_facts_and_document_diversity() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_tank", kind="memory"),
            content=(
                "# Aquarium\n\n"
                "- <seq=1> Tank contains 10 neon tetras, 5 gouramis, and 1 pleco.\n"
                "- <seq=2> User cleans the aquarium weekly.\n"
                "- <seq=3> User enjoys the aquarium."
            ),
            key="doc/pets/memory_tank.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_betta", kind="memory"),
            content="# Pets\n\n- <seq=4> Bubbles is the user's betta fish.",
            key="doc/pets/memory_betta.md",
        ),
    ]

    ledger = _answer_evidence_ledger(
        "How many fish do I have across my aquariums?",
        documents,
        limit=4,
    )

    assert "10 neon tetras" in ledger
    assert "Bubbles" in ledger


def test_answer_evidence_ledger_preserves_raw_speaker_attribution() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_camera", kind="evidence"),
        content=(
            "# Source conversation 2023-01-01\n\n"
            "[ASSISTANT]\nYou could buy a Nikon camera.\n\n"
            "[USER]\nI bought a Sony camera."
        ),
        key="evidence/EVIDENCE_camera.md",
    )

    ledger = _answer_evidence_ledger("Which camera did I buy?", [evidence])

    assert "[user] I bought a Sony camera" in ledger
    assert "[assistant] You could buy a Nikon camera" in ledger

    assistant_ledger = _answer_evidence_ledger(
        "Which camera did you suggest I could buy?", [evidence]
    )
    assert "[assistant] You could buy a Nikon camera" in assistant_ledger


def test_answer_evidence_ledger_excludes_legacy_assistant_recommendation_noise() -> (
    None
):
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_games", kind="memory"),
        content=(
            "# Games\n\n"
            "- <seq=1> User completed Alpha in 12 hours.\n"
            "- <seq=2> Assistant's first recommendation is Beta (20-30 hours).\n"
            "- <seq=3> The assistant recommended Gamma with 40 hours of playtime."
        ),
    )

    ledger = _answer_evidence_ledger(
        "How many hours have I spent playing games in total?", [document]
    )

    assert "Alpha" in ledger
    assert "Beta" not in ledger
    assert "Gamma" not in ledger


def test_answer_evidence_ledger_keeps_adjacent_assistant_clarification() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_coupon", kind="evidence"),
        content=(
            "# Coupon\n\n[ASSISTANT]\nThe Cartwheel app saves money at Target.\n\n"
            "[USER]\nI redeemed a $5 coupon on coffee creamer.\n\n"
            "[ASSISTANT]\nTarget sends coupons to email subscribers."
        ),
        key="evidence/EVIDENCE_coupon.md",
    )

    ledger = _answer_evidence_ledger(
        "Where did I redeem a $5 coupon on coffee creamer?", [evidence]
    )

    assert "redeemed a $5 coupon" in ledger
    assert "Target" in ledger


def test_answer_evidence_ledger_prefers_selected_lossless_source_over_copy() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_race", kind="evidence"),
        content="# Race\n\n[USER]\nI completed the harbor race.",
    )
    derived = MemoryDocument(
        metadata=DocumentMetadata(id="memory_race", kind="memory"),
        content=(
            "# Race\n\n- <seq=1,origin=EVIDENCE_race> "
            "The user said: I completed the harbor race."
        ),
    )

    ledger = _answer_evidence_ledger(
        "How many races did I complete?",
        [evidence, derived],
    )

    assert ledger.count("harbor race") == 1
    assert "EVIDENCE_race" in ledger
    assert "memory_race" not in ledger


def test_answer_evidence_ledger_compacts_long_turn_around_query_terms() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="memory_music", kind="memory"),
        content=(
            "# Conversation turns\n\n- <seq=1> The user said: "
            + ("unrelated recommendation text " * 100)
            + "I've been listening on Spotify lately."
        ),
        key="doc/music/memory_music.md",
    )

    ledger = _answer_evidence_ledger(
        "Which music service have I been using lately?", [evidence]
    )

    assert "Spotify lately" in ledger
    assert len(ledger) < 1200


def test_answer_evidence_ledger_keeps_indirectly_requested_assistant_memory() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="memory_recipe", kind="memory"),
        content=(
            "# Conversation turns\n\n"
            "- <seq=1> The user said: I enjoy winter cooking.\n"
            "- <seq=2,source=AI> The assistant said: One recipe is Cedar Stew."
        ),
        key="doc/cooking/memory_recipe.md",
    )

    ledger = _answer_evidence_ledger(
        "Can you remind me of the recipe from our previous conversation?",
        [evidence],
    )

    assert "Cedar Stew" in ledger


def test_answer_evidence_ledger_keeps_assistant_response_after_matching_request() -> (
    None
):
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_recipe", kind="evidence"),
        content=(
            "# Source conversation\n\n"
            "[USER]\nHow do I make a classic French omelette?\n\n"
            "[ASSISTANT]\nIngredients:\n- 3 eggs\n- butter\n- salt\n"
            "1. Whisk the eggs.\n2. Heat the pan."
        ),
    )

    ledger = _answer_evidence_ledger(
        "How many eggs did you suggest for the classic French omelette?",
        [evidence],
        limit=8,
    )

    assert "[list_context=Ingredients:] - 3 eggs" in ledger


def test_answer_evidence_ledger_promotes_requested_assistant_list_ordinal() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_jobs", kind="evidence"),
        content=(
            "# Source conversation\n\n"
            "[USER]\nList several remote jobs for writers.\n\n"
            "[ASSISTANT]\nRemote writing jobs:\n"
            + "\n".join(f"{number}. Job {number}" for number in range(1, 30))
        ),
    )

    ledger = _answer_evidence_ledger(
        "What was the 27th job you listed for writers?",
        [evidence],
        limit=8,
    )

    assert "27. Job 27" in ledger
    assert ledger.index("27. Job 27") < ledger.index("1. Job 1")


def test_answer_evidence_ledger_binds_first_list_to_user_request() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_jobs", kind="evidence"),
        content=(
            "# Source conversation\n\n[USER]\n"
            "Brainstorm work from home jobs for seniors.\n\n[ASSISTANT]\n"
            "Session ID: jobs\nSession date: 2023/05/26\n"
            "1. Virtual assistant\n7. Transcriptionist"
        ),
    )

    ledger = _answer_evidence_ledger(
        "What was the seventh job in the work from home jobs for seniors list?",
        [evidence],
    )

    assert "[list_context=Brainstorm work from home jobs for seniors.]" in ledger


def test_answer_evidence_ledger_understands_word_ordinals() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_bottles", kind="evidence"),
        content=(
            "# Source conversation\n\n[ASSISTANT]\n"
            "Five bottles for cocktails:\n"
            "1. Vermouth\n2. Campari\n3. Sherry\n4. Amaro\n5. Absinthe"
        ),
    )

    ledger = _answer_evidence_ledger(
        "What was the fifth bottle you recommended for cocktails?",
        [evidence],
        limit=5,
    )

    assert "5. Absinthe" in ledger


def test_answer_evidence_ledger_keeps_direct_assistant_value_before_context() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_campaign", kind="evidence"),
        content=(
            "# Source conversation\n\n[USER]\nPlan an influencer campaign.\n\n"
            "[ASSISTANT]\nInfluencer campaign plan:\n"
            + "\n".join(f"* General tactic {number}" for number in range(40))
            + "\n* Influencer marketing: $2,000"
        ),
    )

    ledger = _answer_evidence_ledger(
        "How much did you allocate for influencer marketing?",
        [evidence],
        limit=12,
    )

    assert "$2,000" in ledger


def test_answer_evidence_ledger_binds_bullets_to_numbered_subsection() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_refineries", kind="evidence"),
        content=(
            "# Source conversation\n\n[ASSISTANT]\nRefinery processes:\n"
            "1. Lake Charles Refinery:\n"
            "* Atmospheric distillation\n* Fluid catalytic cracking\n"
            "* Alkylation\n* Hydrotreating\n"
            "2. Other Refinery:\n* Hydrocracking\n* Delayed coking"
        ),
    )

    ledger = _answer_evidence_ledger(
        "What processes did you list for the Lake Charles Refinery?",
        [evidence],
        limit=8,
    )

    assert "[list_context=1. Lake Charles Refinery:] * Hydrotreating" in ledger
    assert "[list_context=2. Other Refinery:] * Hydrocracking" in ledger


def test_collapse_lineage_preserves_authoritative_evidence_source() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_source", kind="evidence"),
        content="# Source conversation\n\n[ASSISTANT]\n27. Sound effects",
    )
    memory = MemoryDocument(
        metadata=DocumentMetadata(
            id="memory_derived",
            kind="memory",
            source_ids=["EVIDENCE_source"],
        ),
        content="# Conversation\n\n- The assistant supplied a long list.",
    )

    collapsed = _collapse_lineage([memory, evidence])

    assert [item.metadata.id for item in collapsed] == [
        "memory_derived",
        "EVIDENCE_source",
    ]


def test_agentic_answer_focusing_keeps_complete_evidence_session() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_list", kind="evidence"),
        content=(
            "# Source conversation\n\n[ASSISTANT]\n"
            + "\n".join(f"{number}. Item {number}" for number in range(1, 40))
        ),
    )
    retriever = AgenticRetriever(
        documents=[evidence],
        llm=FakeFlowLLM(),
        config=RetrievalConfig(agentic_answer_max_lines_per_document=4),
        limit=1,
    )

    focused = retriever._focus_documents_for_answer(
        "What was the 27th item you listed?", [evidence]
    )

    assert focused[0].content == evidence.content
    assert "27. Item 27" in focused[0].content


def test_answer_evidence_ledger_annotates_generic_relative_target_distance() -> None:
    evidence = MemoryDocument(
        metadata=DocumentMetadata(id="memory_events", kind="memory"),
        content=(
            "# Events\n\n"
            "- <seq=1,observed=2024-02-01> The user completed a conference talk.\n"
            "- <seq=2,observed=2024-02-15> The user completed a product demo."
        ),
        key="doc/work/memory_events.md",
    )

    ledger = _answer_evidence_ledger(
        "Question date: 2024-02-29\nQuestion: What did I complete four weeks ago?",
        [evidence],
    )

    assert "target_date=2024-02-01,distance_days=0" in ledger
    assert "target_date=2024-02-01,distance_days=14" in ledger


def test_answer_evidence_ledger_promotes_current_age_for_future_projection() -> None:
    profile = MemoryDocument(
        metadata=DocumentMetadata(id="memory_profile", kind="memory"),
        content=(
            "# Profile\n\n"
            "- <seq=1> The user is currently 34 years old.\n"
            + "\n".join(
                f"- <seq={number}> The user has an unrelated preference {number}."
                for number in range(2, 24)
            )
        ),
    )
    event = MemoryDocument(
        metadata=DocumentMetadata(id="memory_event", kind="memory"),
        content=(
            "# Future event\n\n"
            "- <seq=30> The user plans to attend the reunion in six years."
        ),
    )

    ledger = _answer_evidence_ledger(
        "How old will I be when I attend the reunion?",
        [event, profile],
        limit=8,
    )

    assert "currently 34 years old" in ledger
    assert "reunion in six years" in ledger


def test_auto_direct_answer_uses_compact_ranked_evidence_context() -> None:
    class CompactAnswerLLM:
        def __init__(self) -> None:
            self.prompt = ""

        def complete(self, request: LLMRequest) -> str:
            assert request.operation == "search_answer"
            self.prompt = request.messages[-1].content
            return json.dumps(
                {
                    "evidence": ["bedroom walls were repainted light gray"],
                    "calculation": "direct lookup",
                    "answer": "a lighter shade of gray",
                }
            )

    llm = CompactAnswerLLM()
    flow = _create_flow(
        _config(),
        store=InMemoryObjectStore(),
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
    )
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_bedroom", kind="memory"),
        content=(
            "# Conversation turns\n\n"
            "- <seq=1> The user repainted the bedroom walls a lighter shade of gray.\n"
            "- <seq=2> The user also discussed unrelated gardening advice."
        ),
        key="doc/home/memory_bedroom.md",
    )

    answer = flow.retriever._answer(
        SearchRequest(
            query="What color did I repaint my bedroom walls?",
            strategy=SearchStrategy.AUTO,
            answer=True,
        ),
        [document],
    )

    assert answer == "a lighter shade of gray"
    assert "FOCUSED_EVIDENCE_LEDGER" in llm.prompt
    assert "PRIMARY_SOURCE_EXCERPTS" in llm.prompt
    assert "MEMORY_CONTEXT" not in llm.prompt


def test_primary_source_context_preserves_adjacent_turn_relation() -> None:
    relevant = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_coupon", kind="evidence"),
        content=(
            "# Source conversation\n\n[USER]\n"
            "I use the rewards app from Corner Market.\n"
            "[USER]\nI redeemed a coupon on coffee creamer yesterday."
        ),
    )
    unrelated = MemoryDocument(
        metadata=DocumentMetadata(id="EVIDENCE_other", kind="evidence"),
        content="# Source conversation\n\n[USER]\nI bought tea elsewhere.",
    )

    context = _primary_source_context(
        "Where did I redeem a coupon on coffee creamer?",
        [relevant, unrelated],
        document_limit=1,
    )

    assert "SOURCE EVIDENCE_coupon" in context
    assert "Corner Market" in context
    assert "redeemed a coupon" in context
    assert "bought tea elsewhere" not in context


def test_primary_source_context_keeps_more_sources_for_aggregate_query() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id=f"EVIDENCE_{index}", kind="evidence"),
            content=f"# Source conversation\n\n[USER]\nI completed activity {index}.",
        )
        for index in range(1, 6)
    ]

    aggregate = _primary_source_context(
        "How many activities did I complete?", documents
    )
    direct = _primary_source_context("Which activity did I complete?", documents)

    assert "SOURCE EVIDENCE_4" in aggregate
    assert "SOURCE EVIDENCE_5" not in aggregate
    assert "SOURCE EVIDENCE_2" in direct
    assert "SOURCE EVIDENCE_3" not in direct


def test_auto_answer_auditor_revises_unsupported_draft() -> None:
    class ReviewingLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, request: LLMRequest) -> str:
            assert request.operation == "search_answer"
            self.prompts.append(request.messages[-1].content)
            if len(self.prompts) == 1:
                return json.dumps(
                    {
                        "evidence": [],
                        "calculation": "location missing",
                        "answer": "The location was not recorded.",
                    }
                )
            return json.dumps(
                {
                    "verdict": "revise",
                    "evidence": ["the nearby reply named the shop"],
                    "calculation": "same-source coreference",
                    "answer": "Target",
                }
            )

    llm = ReviewingLLM()
    retriever = MemoryRetriever.model_construct(
        llm=llm,
        config=RetrievalConfig(),
    )
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_coupon", kind="evidence"),
        content=(
            "# Conversation turns\n\n"
            + "".join(
                f"- <seq={index},source=human> Unrelated archive note {index}.\n"
                for index in range(100)
            )
            + "- <seq=1,source=human> I redeemed a coupon on coffee creamer.\n"
            "- <seq=2,source=AI> Where did you use it?\n"
            "- <seq=3,source=human> Target."
        ),
    )

    answer = retriever._answer(
        SearchRequest(
            query="Where did I redeem the coupon?",
            strategy=SearchStrategy.AUTO,
            answer=True,
        ),
        [document],
    )

    assert answer == "Target"
    assert len(llm.prompts) == 2
    assert "Independently audit DRAFT_RESULT" in llm.prompts[1]
    assert "AUDIT_SOURCE_CONTEXT" in llm.prompts[1]
    assert "Target." in llm.prompts[1]


def test_auto_answer_auditor_does_not_replace_grounded_draft_with_abstention() -> None:
    class RegressingReviewLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: LLMRequest) -> str:
            self.calls += 1
            answer = (
                "Keep the portable charger accessible."
                if self.calls == 1
                else "Stored memory does not contain specific troubleshooting tips."
            )
            return json.dumps({"answer": answer})

    llm = RegressingReviewLLM()
    retriever = MemoryRetriever.model_construct(
        llm=llm,
        config=RetrievalConfig(),
    )
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_power", kind="evidence"),
        content="- <seq=1,source=human> I have a portable charger.",
    )

    answer = retriever._answer(
        SearchRequest(
            query="Any tips for my phone battery?",
            strategy=SearchStrategy.AUTO,
            answer=True,
        ),
        [document],
    )

    assert answer == "Keep the portable charger accessible."
    assert llm.calls == 1


def test_target_date_evidence_recalls_resolved_source_session() -> None:
    retriever = MemoryRetriever.model_construct(config=RetrievalConfig())
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_old", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2023/04/07 (Fri)\n"
                "[USER]\nI listened to a jazz quartet."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_target", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2023/04/28 (Fri)\n"
                "[USER]\nI discovered a bluegrass band with a banjo player."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_target", kind="memory"),
            content=(
                "# Music\n\n- <observed=2023-04-28> User discovered a bluegrass band."
            ),
        ),
    ]

    selected = retriever._target_date_evidence(
        "Question date: 2023/05/05\n"
        "Question: What artist did I start listening to last Friday?",
        documents,
        3,
    )

    assert [item.metadata.id for item in selected] == ["EVIDENCE_target"]


def test_temporal_window_evidence_recalls_category_members() -> None:
    retriever = MemoryRetriever.model_construct(config=RetrievalConfig())
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_triathlon", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2023/06/01 (Thu)\n"
                "[USER]\nI completed the Spring Triathlon today."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_soccer", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2023/06/17 (Sat)\n"
                "[USER]\nI joined the annual charity soccer tournament."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_old", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2023/04/01 (Sat)\n"
                "[USER]\nI played in a tennis tournament."
            ),
        ),
    ]

    selected = retriever._temporal_window_evidence(
        "Question date: 2023/06/18\n"
        "Question: What is the order of the sports events I participated in "
        "during the past month?",
        documents,
        10,
    )

    assert {item.metadata.id for item in selected} == {
        "EVIDENCE_triathlon",
        "EVIDENCE_soccer",
    }


def test_temporal_window_evidence_supports_named_month() -> None:
    retriever = MemoryRetriever.model_construct(config=RetrievalConfig())
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_january", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2024/01/08 (Mon)\n"
                "[USER]\nI watched a championship game."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_february", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2024/02/08 (Thu)\n"
                "[USER]\nI watched another championship game."
            ),
        ),
    ]

    selected = retriever._temporal_window_evidence(
        "Question date: 2024/03/01\nQuestion: Which sports events did I watch in January?",
        documents,
        10,
    )

    assert [item.metadata.id for item in selected] == ["EVIDENCE_january"]


@pytest.mark.parametrize(
    ("question", "expected_ids"),
    [
        (
            "How much did I spend since the start of the year?",
            {"EVIDENCE_january", "EVIDENCE_april"},
        ),
        (
            "How many items did I acquire in the past few months?",
            {"EVIDENCE_january", "EVIDENCE_april"},
        ),
    ],
)
def test_temporal_window_evidence_supports_period_start_and_vague_counts(
    question: str,
    expected_ids: set[str],
) -> None:
    retriever = MemoryRetriever.model_construct(config=RetrievalConfig())
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_january", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2023/01/25 (Wed)\n[USER]\n"
                "I acquired an item and spent money."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_april", kind="evidence"),
            content=(
                "# Session\n\nSession date: 2023/04/10 (Mon)\n[USER]\n"
                "I acquired an item and spent money."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="EVIDENCE_old", kind="evidence"),
            content="# Session\n\nSession date: 2022/12/31 (Sat)\n[USER]\nA fact.",
        ),
    ]

    selected = retriever._temporal_window_evidence(
        f"Question date: 2023/04/20\nQuestion: {question}",
        documents,
        10,
    )

    assert {item.metadata.id for item in selected} == expected_ids


def test_fact_projection_ignores_session_date_as_quantity() -> None:
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_album", kind="memory"),
        content=(
            "# Collection\n\n"
            "- <seq=3,observed=2023-05-27> The user said: Session ID: source_1 "
            "Session date: 2023/05/27 (Sat) 15:55 The poster was limited to "
            "500 copies worldwide."
        ),
    )

    fact = project_document(document)[0]

    assert fact.object.number == 500
    assert fact.object.raw == "500"


def test_agentic_answer_uses_observed_dates_for_explicit_elapsed_arithmetic() -> None:
    class StubLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: LLMRequest) -> str:
            self.calls += 1
            return json.dumps(
                {
                    "evidence": ["first event", "second event"],
                    "calculation": "six elapsed days",
                    "answer": "6 days",
                }
            )

    llm = StubLLM()
    retriever = MemoryRetriever.model_construct(
        llm=llm,
        config=RetrievalConfig(),
    )
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_keyboard", kind="memory"),
            content=(
                "# Music\n\n"
                "- <seq=1,observed=2023-03-25> User started playing favorite "
                "songs on an old keyboard."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_bluegrass", kind="memory"),
            content=(
                "# Music\n\n"
                "- <seq=2,observed=2023-03-31> User discovered a bluegrass band."
            ),
        ),
    ]
    request = SearchRequest(
        query=(
            "Question date: 2023/04/05\nQuestion: How many days passed between "
            "the day I started playing my favorite songs on my old keyboard and "
            "the day I discovered a bluegrass band?"
        ),
        strategy=SearchStrategy.AGENTIC,
        answer=True,
    )

    assert retriever._answer(request, documents) == "6 days"
    assert llm.calls == 1


def test_agentic_answer_rejects_zero_day_observed_date_collision() -> None:
    class StubLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: LLMRequest) -> str:
            self.calls += 1
            return json.dumps({"answer": "7 days"})

    llm = StubLLM()
    retriever = MemoryRetriever.model_construct(
        llm=llm,
        config=RetrievalConfig(),
    )
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_museums", kind="memory"),
            content=(
                "# Museums\n\n"
                "- <seq=1,observed=2023-03-31> User discussed both the Museum "
                "of Modern Art and a Metropolitan Museum exhibit."
            ),
        )
    ]
    request = SearchRequest(
        query=(
            "How many days passed between my visit to the Museum of Modern Art "
            "and the Ancient Civilizations exhibit at the Metropolitan Museum?"
        ),
        strategy=SearchStrategy.AGENTIC,
        answer=True,
    )

    assert retriever._answer(request, documents) == "7 days"
    assert llm.calls == 1


def test_agentic_answer_rejects_zero_day_since_observed_date_collision() -> None:
    class StubLLM:
        def complete(self, request: LLMRequest) -> str:
            return json.dumps({"answer": "24 days"})

    retriever = MemoryRetriever.model_construct(
        llm=StubLLM(),
        config=RetrievalConfig(),
    )
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_music", kind="memory"),
            content=(
                "# Music\n\n"
                "- <seq=36,observed=2023-02-01> User started ukulele lessons "
                "and planned to service an acoustic guitar.\n"
                "- <seq=39,time=2023-02-25> User took the acoustic guitar to "
                "a guitar tech for servicing."
            ),
        )
    ]
    request = SearchRequest(
        query=(
            "How many days had passed since I started taking ukulele lessons "
            "when I took my acoustic guitar to the guitar tech for servicing?"
        ),
        strategy=SearchStrategy.AGENTIC,
        answer=True,
    )

    assert retriever._answer(request, documents) == "24 days"


def test_answer_evidence_ledger_promotes_latest_matching_current_state() -> None:
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_schedule", kind="memory"),
        content=(
            "# Routine\n\n"
            "- <seq=3> User wakes up around 8:30 am on Saturdays.\n"
            "- <seq=7> User woke at 9:30 am last Saturday.\n"
            "- <seq=48> User likes to wake up at 7:30 am on Saturdays."
        ),
    )

    ledger = _answer_evidence_ledger(
        "What time do I wake up on Saturdays?", [document], limit=3
    )

    assert ledger.index("seq=48") < ledger.index("seq=3")


def test_fact_index_projects_dual_time_and_fuses_retrieval_signals() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_noise", kind="memory"),
            content=(
                "# Art\n\n"
                "- <seq=9,observed=2023-03-10,time=2023-03-10> "
                "The user got interested in painting."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_smoker", kind="memory"),
            content=(
                "# Cooking equipment\n\n"
                "- <seq=10,observed=2023-03-15,time=2023-03-15> "
                "The user just got a smoker."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_assistant", kind="memory"),
            content=(
                "# Advice\n\n"
                "- <seq=11,observed=2023-03-16,source=AI> "
                "The assistant recommended an electric smoker."
            ),
        ),
    ]
    query = (
        "Question date: 2023-03-25\nQuestion: How many days ago did I buy my smoker?"
    )
    plan = build_fact_query_plan(query)
    records = parse_fact_records(documents)
    ranked = rank_fact_documents(
        query,
        documents,
        2,
        expanded_query=f"{query} bought purchased acquired ordered got smoker",
    )

    assert plan.temporal is True
    assert plan.question_date is not None
    assert records[1].event_date == records[1].observed_date
    assert records[2].source == "ai"
    assert ranked[0].metadata.id == "memory_smoker"


def test_fact_index_prioritizes_relative_calendar_target_date() -> None:
    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_recent_charity", kind="memory"),
            content=(
                "# Charity\n\n"
                "- <seq=2,observed=2023-04-20,time=2023-04-20> "
                "The user participated in a neighborhood charity auction."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_target_charity", kind="memory"),
            content=(
                "# Charity\n\n"
                "- <seq=1,observed=2023-03-30,time=2023-03-30> "
                "The user participated in the Walk for Hunger charity event."
            ),
        ),
    ]
    query = (
        "Question date: 2023-04-30\n"
        "Question: What charity event did I participate in a month ago?"
    )

    plan = build_fact_query_plan(query)
    ranked = rank_fact_documents(query, documents, 2)

    assert plan.target_date == date(2023, 3, 30)
    assert ranked[0].metadata.id == "memory_target_charity"

    jewelry_documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_old_jewelry", kind="memory"),
            content=(
                "# Gifts\n\n"
                "- <seq=1,observed=2023-04-15> "
                "The user's friend gave them a jewelry box."
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_saturday_jewelry", kind="memory"),
            content=(
                "# Gifts\n\n"
                "- <seq=2,observed=2023-04-29> "
                "The user's aunt gave them a piece of jewelry."
            ),
        ),
    ]
    weekday_query = (
        "Question date: 2023-04-30\n"
        "Question: I received a piece of jewelry last Saturday from whom?"
    )
    weekday_plan = build_fact_query_plan(weekday_query)

    assert weekday_plan.target_date == date(2023, 4, 29)
    assert (
        rank_fact_documents(weekday_query, jewelry_documents, 2)[0].metadata.id
        == "memory_saturday_jewelry"
    )


def test_extractor_adds_authoritative_observed_date_to_atomic_records() -> None:
    config = _config()
    flow = _create_flow(
        config,
        store=InMemoryObjectStore(),
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )

    result = flow.extract(
        ExtractionRequest(
            sequence=1700000007,
            messages=[
                ChatMessage(
                    role="user",
                    content="Session date: 2023/03/15 (Wed) 10:00\nI got a smoker.",
                )
            ],
        )
    )

    assert "<seq=1700000007,origin=EVIDENCE_" in result.extracted_content
    assert ",observed=2023-03-15>" in result.extracted_content
    evidence = flow.extractor.repository.read(result.evidence_key)
    assert evidence.metadata.kind == "evidence"
    assert "[USER]\nSession date: 2023/03/15" in evidence.content


def test_extractor_validates_per_fact_observed_dates_in_multi_session_batch() -> None:
    class MultiSessionLLM:
        def complete(self, _request: LLMRequest) -> str:
            return (
                "# Preferences\n"
                "- <seq=0,observed=2023-03-15> User likes tea.\n"
                "- <seq=0,observed=2023-03-16> User likes coffee.\n"
                "- <seq=0,observed=2099-01-01> User likes water."
            )

    store = InMemoryObjectStore()
    flow = _create_flow(
        _config(),
        store=store,
        llm=MultiSessionLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )

    result = flow.extract(
        ExtractionRequest(
            sequence=9,
            messages=[
                ChatMessage(
                    role="user",
                    content="Session date: 2023/03/15\nTea",
                    source_id="session-1",
                    observed_at=date(2023, 3, 15),
                    turn_index=0,
                ),
                ChatMessage(
                    role="user",
                    content="Session date: 2023/03/16\nCoffee",
                    source_id="session-2",
                    observed_at=date(2023, 3, 16),
                    turn_index=0,
                ),
            ],
        )
    )

    assert "observed=2023-03-15" in result.extracted_content
    assert "observed=2023-03-16" in result.extracted_content
    assert "2099-01-01" not in result.extracted_content
    origins = re.findall(r"origin=(EVIDENCE_[^,>]+)", result.extracted_content)
    assert len(set(origins)) == 2
    assert len(flow.extractor.repository.list_evidence()) == 2


def test_agentic_uses_wider_answer_context_for_aggregation_queries() -> None:
    events = [
        "- User attended an art lecture.",
        "- User attended an art exhibition.",
        "- User volunteered at the Children's Museum Art Afternoon event.",
        "- User visited the History Museum on a guided tour.",
    ]
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_art", kind="memory"),
        content="# Art\n\n"
        + "\n".join(
            [
                *events[:2],
                *(f"- Generic art interest {i}." for i in range(50)),
                *events[2:],
            ]
        ),
        key="doc/art/memory_art.md",
    )
    retriever = AgenticRetriever(
        documents=[document],
        llm=FakeFlowLLM(),
        config=RetrievalConfig(
            agentic_answer_max_lines_per_document=3,
            agentic_aggregation_max_lines_per_document=80,
        ),
        limit=10,
    )

    focused = retriever._focus_documents_for_answer(
        "How many different art-related events did I attend?", [document]
    )[0]

    assert "volunteered at the Children's Museum" in focused.content
    assert "visited the History Museum" in focused.content


def test_agentic_high_recall_search_skips_llm_agent() -> None:
    class NoAgentLLM(FakeFlowLLM):
        def as_langchain_model(self):
            raise AssertionError("high-recall search must not create an LLM agent")

    documents = [
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_one", kind="memory"),
            content="# Events\n\n- User attended one art event.",
            key="doc/events/memory_one.md",
        ),
        MemoryDocument(
            metadata=DocumentMetadata(id="memory_two", kind="memory"),
            content="# Events\n\n- User volunteered at another art event.",
            key="doc/events/memory_two.md",
        ),
    ]
    retriever = AgenticRetriever(
        documents=documents,
        llm=NoAgentLLM(),
        config=RetrievalConfig(),
        limit=20,
    )

    selected = retriever.search("How many different art events did I attend?")

    assert {item.metadata.id for item in selected} == {
        "memory_one",
        "memory_two",
    }


def test_agentic_answer_rehydrates_partition_hit_to_complete_leaf() -> None:
    complete = MemoryDocument(
        metadata=DocumentMetadata(id="memory_care", kind="memory"),
        content=(
            "# Fitness\n\n- User teaches spinning classes.\n\n"
            "# Personal Care\n\n"
            "- User uses lavender shampoo from Trader Joe's."
        ),
        key="doc/care/memory_care.md",
    )
    partition = complete.model_copy(
        update={"content": "# Fitness\n\n- User teaches spinning classes."}
    )
    retriever = AgenticRetriever(
        documents=[complete],
        llm=FakeFlowLLM(),
        config=RetrievalConfig(),
        limit=20,
    )

    focused = retriever._focus_documents_for_answer(
        "What brand of shampoo do I currently use?", [partition]
    )

    assert "Trader Joe's" in focused[0].content


def test_agentic_focus_keeps_pack_and_wear_word_forms_for_ratio() -> None:
    document = MemoryDocument(
        metadata=DocumentMetadata(id="memory_shoes", kind="memory"),
        content=(
            "# Travel\n\n"
            "- On the last trip, the user packed 5 pairs of shoes.\n"
            + "\n".join(f"- Generic packing suggestion {index}." for index in range(50))
            + "\n- On the previous trip, the user wore only sneakers and sandals."
        ),
        key="doc/travel/memory_shoes.md",
    )

    focused = AgenticRetriever._focus_document_for_answer(
        "What percentage of packed shoes did I wear on my last trip?",
        document,
        max_lines=3,
    )

    assert "packed 5 pairs" in focused.content
    assert "wore only sneakers and sandals" in focused.content
    assert focused.content.index("packed 5 pairs") < focused.content.index("wore only")


def _config() -> MemFlowConfig:
    return MemFlowConfig(
        s3=S3Config(
            endpoint_url=os.getenv("MEM_FLOW_S3_ENDPOINT", "http://127.0.0.1:9000"),
            bucket=os.getenv("MEM_FLOW_S3_BUCKET", "infini-memory-test"),
            access_key=os.getenv("MEM_FLOW_S3_ACCESS_KEY", ""),
            secret_key=os.getenv("MEM_FLOW_S3_SECRET_KEY", ""),
            fixed_prefix=os.getenv("MEM_FLOW_S3_PREFIX", "inf_mem_test"),
            ensure_bucket=os.getenv("MEM_FLOW_S3_ENSURE_BUCKET", "false").lower()
            == "true",
        ),
        llm=LLMConfig(
            api_key=os.getenv("MEM_FLOW_LLM_API_KEY", ""),
            base_url=os.getenv("MEM_FLOW_LLM_BASE_URL", ""),
            model=os.getenv("MEM_FLOW_LLM_MODEL", "deepseek-v4-flash-0731"),
            retry_attempts=int(os.getenv("MEM_FLOW_LLM_RETRY_ATTEMPTS", "3")),
        ),
        metrics=MetricsConfig(namespace=f"mem_flow_test_{uuid4().hex[:8]}"),
    )


def _create_flow(
    config: MemFlowConfig,
    *,
    store_id: str = "test",
    user_id: str = "mem_flow_test",
    **kwargs,
) -> MemFlow:
    return MemFlow.create(
        config,
        store_id=store_id,
        user_id=user_id,
        **kwargs,
    )


def test_mem_flow_requires_scope_and_rejects_cross_scope_keys() -> None:
    config = _config()
    store = InMemoryObjectStore()
    kwargs = {
        "store": store,
        "llm": FakeFlowLLM(),
        "metrics": FlowMetrics.create(enabled=False),
    }

    with pytest.raises(TypeError):
        MemFlow.create(config, **kwargs)

    alice = MemFlow.create(config, store_id="private", user_id="alice", **kwargs)
    bob = MemFlow.create(config, store_id="private", user_id="bob", **kwargs)
    result = alice.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="Alice likes tea.")],
            infer=False,
        )
    )

    assert result.current_key.startswith("current/")
    assert "private" not in result.current_key
    assert "alice" not in result.current_key
    assert len(alice.extractor.repository.list_current()) == 1
    assert set(alice.extractor.repository.store.list_keys("")) == {
        result.current_key,
        result.evidence_key,
    }
    assert bob.extractor.repository.list_current() == []
    with pytest.raises(ValueError, match="unsafe scoped object key"):
        alice.extractor.repository.store.get_text("../USER_bob/current/secret.md")


def test_direct_source_tagged_extraction_writes_lossless_turn_records() -> None:
    config = _config()
    flow = _create_flow(
        config,
        store=InMemoryObjectStore(),
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )

    result = flow.extract(
        ExtractionRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Session date: 2026/08/01\nI paid $20.",
                    source_id="session-a",
                ),
                ChatMessage(
                    role="assistant",
                    content="The workshop costs $500.",
                    source_id="session-a",
                ),
            ],
            infer=False,
            sequence=7,
        )
    )

    assert "# Conversation turns" in result.extracted_content
    assert result.extracted_content.count("- <seq=7") == 2
    assert "observed=2026-08-01" in result.extracted_content
    assert ",source=AI> The assistant said:" in result.extracted_content
    assert result.extracted_content.count("origin=EVIDENCE_") == 2


def test_deterministic_maintenance_uses_no_llm_calls() -> None:
    config = _config().model_copy(
        update={
            "maintenance": MaintenanceConfig(
                rewrite_batch_max_tokens=12000,
                document_summary_tokens=100,
                deterministic=True,
            )
        }
    )
    llm = FakeFlowLLM()
    flow = _create_flow(
        config,
        store=InMemoryObjectStore(),
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
    )
    flow.extract(
        ExtractionRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Session date: 2026/08/01\nI paid $20.",
                    source_id="session-a",
                )
            ],
            infer=False,
            sequence=1,
        )
    )

    result = flow.maintain(MaintenanceRequest())

    assert result.memory_documents_created == 1
    assert flow.maintainer.repository.list_memory_documents()
    assert llm.calls == []


def test_doc_body_read_and_line_update_interfaces() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    repository = flow.documents.repository
    key = flow.maintainer.layout.memory_key("dir_drinks", "memory_drinks")
    repository.write(
        key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id="memory_drinks",
                kind="memory",
                directory_id="dir_drinks",
                title="饮品偏好",
                summary="用户喜欢无糖拿铁。",
            ),
            content=(
                "# 饮品偏好\n\n"
                "- <seq=1785398401> 用户喜欢无糖拿铁。\n"
                "- <seq=1785398402> 用户偶尔喝绿茶。"
            ),
        ),
    )
    flow.maintain(MaintenanceRequest())
    topic_key = flow.maintainer.layout.directory_topic_key("dir_drinks")
    topic_digest_before = repository.read(topic_key).metadata.content_digest
    persisted_before = repository.store.get_text(key)
    front_matter_before = yaml.safe_load(persisted_before[4:].split("\n---\n", 1)[0])

    read = flow.get_doc(DocReadRequest(document_id="memory_drinks"))
    assert set(read.model_dump()) == {"document_id", "content", "line_count"}
    assert read.document_id == "memory_drinks"
    assert read.line_count == 4
    assert not read.content.startswith("---")
    assert "store_id" not in read.content

    update = flow.update_doc_lines(
        DocLineUpdateRequest(
            document_id="memory_drinks",
            start_line=3,
            end_line=4,
            replacement=(
                "- <seq=1785398401> 用户喜欢燕麦拿铁。\n"
                "- <seq=1785398403> 用户也喜欢乌龙茶。"
            ),
        )
    )
    assert update.changed is True
    assert update.replacement_line_count == 2
    assert set(update.document.model_dump()) == {
        "document_id",
        "content",
        "line_count",
    }
    assert "燕麦拿铁" in update.document.content
    assert "无糖拿铁" not in update.document.content
    assert flow.get_doc(DocReadRequest(document_id="memory_drinks")) == update.document

    persisted_after = repository.store.get_text(key)
    front_matter_after = yaml.safe_load(persisted_after[4:].split("\n---\n", 1)[0])
    for field in (
        "id",
        "kind",
        "summary",
        "title",
        "directory_id",
        "store_id",
        "user_id",
        "created_at",
        "updated_at",
        "source_ids",
    ):
        assert front_matter_after[field] == front_matter_before[field]
    assert (
        persisted_after[4:].split("\n---\n", 1)[0]
        == persisted_before[4:].split("\n---\n", 1)[0]
    )
    assert strip_yaml_front_matter(persisted_after) == update.document.content
    assert repository.read(topic_key).metadata.content_digest != topic_digest_before

    heading_update = flow.update_doc_lines(
        DocLineUpdateRequest(
            document_id="memory_drinks",
            start_line=1,
            end_line=1,
            replacement="# 热饮偏好",
        )
    )
    assert heading_update.changed is True
    assert repository.read(topic_key).content == "# 热饮偏好"

    with pytest.raises(ValueError, match="line range exceeds"):
        flow.update_doc_lines(
            DocLineUpdateRequest(
                document_id="memory_drinks",
                start_line=99,
                end_line=99,
                replacement="out of range",
            )
        )
    with pytest.raises(LookupError, match="not found"):
        flow.get_doc(DocReadRequest(document_id="missing"))
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        DocLineUpdateRequest(
            document_id="memory_drinks",
            start_line=1,
            end_line=1,
            replacement="body",
            summary="must not be accepted",
        )

    other_user = MemFlow.create(
        config,
        store_id="test",
        user_id="other_user",
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    with pytest.raises(LookupError, match="not found"):
        other_user.get_doc(DocReadRequest(document_id="memory_drinks"))


def test_delete_doc_refreshes_topic_and_removes_empty_directory_metadata() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    repository = flow.documents.repository
    for document_id, title in (
        ("memory_coffee", "咖啡偏好"),
        ("memory_tea", "茶饮偏好"),
    ):
        key = flow.maintainer.layout.memory_key("dir_drinks", document_id)
        repository.write(
            key,
            MemoryDocument(
                metadata=DocumentMetadata(
                    id=document_id,
                    kind="memory",
                    directory_id="dir_drinks",
                    title=title,
                ),
                content=f"# {title}\n\n- <seq=1785398401> 用户喜欢这类饮品。",
            ),
        )
    flow.maintain(MaintenanceRequest())
    topic_key = flow.maintainer.layout.directory_topic_key("dir_drinks")

    assert flow.delete_doc(DocReadRequest(document_id="memory_coffee")) is None
    with pytest.raises(LookupError, match="not found"):
        flow.get_doc(DocReadRequest(document_id="memory_coffee"))
    assert flow.get_doc(DocReadRequest(document_id="memory_tea")).document_id == (
        "memory_tea"
    )
    topic = repository.read(topic_key)
    assert topic.metadata.document_count == 1
    assert topic.content == "# 茶饮偏好"

    flow.delete_doc(DocReadRequest(document_id="memory_tea"))
    assert repository.list_memory_documents("dir_drinks") == []
    assert not repository.store.exists(topic_key)
    assert not repository.store.exists(
        flow.maintainer.layout.legacy_directory_summary_key("dir_drinks")
    )
    with pytest.raises(LookupError, match="not found"):
        flow.delete_doc(DocReadRequest(document_id="memory_tea"))


def test_delete_all_clears_only_bound_store_user_scope() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    other_user = MemFlow.create(
        config,
        store_id="test",
        user_id="other_user",
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="Alice likes tea.")],
            infer=False,
        )
    )
    flow.documents.repository.store.put_text("raw/pending.md", "raw")
    flow.documents.repository.store.put_text("rewrite/ROUTE_pending.json", "{}")
    flow.documents.repository.store.put_text("doc/dir/TOPIC.md", "topic")
    other_result = other_user.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="Bob likes coffee.")],
            infer=False,
        )
    )

    assert flow.delete_all() == 5
    assert flow.documents.repository.store.list_keys("") == []
    assert set(other_user.documents.repository.store.list_keys("")) == {
        other_result.current_key,
        other_result.evidence_key,
    }
    assert flow.delete_all() == 0


def test_all_mem_flow_modules(pytestconfig) -> None:
    use_real_llm = pytestconfig.getoption("--mem-flow-real-llm")
    use_real_s3 = pytestconfig.getoption("--mem-flow-real-s3")
    config = _config()
    if use_real_llm and not config.llm.api_key.get_secret_value():
        pytest.fail("--mem-flow-real-llm requires MEM_FLOW_LLM_API_KEY")
    if use_real_s3 and not os.getenv("MEM_FLOW_S3_ENDPOINT"):
        pytest.fail("--mem-flow-real-s3 requires MEM_FLOW_S3_ENDPOINT")

    registry = CollectorRegistry()
    metrics = FlowMetrics.create(
        namespace=config.metrics.namespace,
        registry=registry,
    )
    store = (
        S3ObjectStore.create(config.s3, metrics)
        if use_real_s3
        else InMemoryObjectStore()
    )
    llm = OpenAIFlowLLM.create(config.llm, metrics) if use_real_llm else FakeFlowLLM()
    flow = _create_flow(config, store=store, llm=llm, metrics=metrics)
    second_flow = _create_flow(
        config,
        store=store,
        llm=llm,
        metrics=metrics,
        instance_id="worker-east-1",
    )
    scope = MemoryScope(store_id="test", user_id="mem_flow_test")
    # Keep this run's final docs in real S3 for inspection. At the beginning of the
    # next run, remove the fixed test scope and legacy UUID-suffixed test scopes.
    test_store_root = f"{config.s3.fixed_prefix}/STORE_test"
    fixed_scope_prefix = f"{test_store_root}/USER_{scope.user_id}/"
    if use_real_s3:
        for old_key in store.list_keys(test_store_root):
            if (
                old_key.startswith(fixed_scope_prefix)
                or "/USER_mem_flow_test_" in old_key
            ):
                store.delete(old_key)

    started_at = int(time.time())
    first = flow.extract(
        ExtractionRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "My name is Alice. I like apples and jazz. Label these "
                        "preferences as leisure."
                    ),
                ),
                ChatMessage(
                    role="assistant", content="Your preferred language is Chinese."
                ),
            ],
        )
    )
    appended = flow.extract(
        ExtractionRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "I will visit Kyoto on 2026-10-12. Label it travel. "
                        "I am allergic to peanuts."
                    ),
                )
            ],
        )
    )
    third = second_flow.extract(
        ExtractionRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "I joined Acme in March 2024. Bob is my emergency contact. "
                        "Hello, can you also explain what HTTP means?"
                    ),
                )
            ],
        )
    )
    finished_at = int(time.time())
    assert str(UUID(flow.instance_id)) == flow.instance_id
    assert flow.instance_id == flow.extractor.instance_id == first.instance_id
    assert second_flow.instance_id == third.instance_id
    assert second_flow.instance_id == "worker-east-1"
    assert flow.instance_id != second_flow.instance_id
    assert not first.current_key.startswith("/")
    assert first.current_key == f"current/CURRENT_{flow.instance_id}.md"
    assert third.current_key == f"current/CURRENT_{second_flow.instance_id}.md"
    assert scope.store_id not in first.current_key
    assert scope.user_id not in first.current_key
    assert appended.appended is True
    assert third.extracted_content
    for extraction in (first, appended, third):
        assert started_at <= extraction.sequence_timestamp <= finished_at
        assert f"<seq={extraction.sequence_timestamp}" in extraction.extracted_content
        assert ",origin=EVIDENCE_" in extraction.extracted_content
        assert "time=empty" not in extraction.extracted_content
    assert {item.key for item in flow.extractor.repository.list_evidence()} == {
        first.evidence_key,
        appended.evidence_key,
        third.evidence_key,
    }

    fresh = flow.search(SearchRequest(query="What do I like?", answer=True))
    assert fresh.hits

    progress_events: list[tuple[str, dict[str, object]]] = []
    maintained = flow.maintain(
        MaintenanceRequest(merge_topics=True),
        progress=lambda stage, details: progress_events.append((stage, details)),
    )
    assert maintained.current_documents == 2
    assert maintained.raw_documents == 2
    assert maintained.rewrite_documents == 1
    if isinstance(llm, FakeFlowLLM):
        assert len(llm.rewrite_sources) == 1
        assert len(llm.rewrite_sources[0]) == 2
    assert maintained.directories_created >= 1
    assert maintained.memory_documents_created >= 1
    assert maintained.directory_topics_updated >= 1
    assert len(maintained.deleted_current_keys) == 2
    assert len(maintained.deleted_raw_keys) == 2
    assert len(maintained.deleted_rewrite_keys) == 1
    assert len(maintained.deleted_route_keys) == 1
    assert {stage for stage, _ in progress_events} == {
        "scan_current",
        "archive_raw",
        "rewrite_current",
        "route_directories",
        "write_documents",
        "refresh_topics",
        "publish_index",
        "cleanup_intermediates",
    }
    assert progress_events[0] == ("scan_current", {"status": "running"})
    assert progress_events[-1][0] == "cleanup_intermediates"
    assert progress_events[-1][1]["status"] == "completed"
    assert flow.extractor.repository.list_current() == []
    raw_documents = flow.extractor.repository.list_raw()
    rewrite_documents = flow.extractor.repository.list_rewrites()
    memory_documents = flow.extractor.repository.list_memory_documents()
    directory_topics = flow.extractor.repository.list_directory_topics()
    assert raw_documents == []
    assert rewrite_documents == []
    assert len(flow.extractor.repository.list_evidence()) == 3
    assert memory_documents
    assert directory_topics
    assert all(
        summary.metadata.document_count
        == len(flow.extractor.repository.list_memory_documents(summary.metadata.id))
        for summary in directory_topics
    )
    assert all(document.key.endswith("/TOPIC.md") for document in directory_topics)
    assert all(
        document.metadata.kind == "directory_topic" for document in directory_topics
    )
    for topic in directory_topics:
        memories = flow.extractor.repository.list_memory_documents(topic.metadata.id)
        expected_headings: list[str] = []
        for memory in memories:
            for heading in extract_h1_headings(memory.content):
                if heading.casefold() not in {
                    item.casefold() for item in expected_headings
                }:
                    expected_headings.append(heading)
        assert extract_h1_headings(topic.content) == expected_headings
        assert topic.content.splitlines() == [
            f"# {heading}" for heading in expected_headings
        ]
    for instance_id in (flow.instance_id, second_flow.instance_id):
        assert any(
            f"raw/RAW_CURRENT_{instance_id}_" in key
            for key in maintained.deleted_raw_keys
        )
    assert all(
        re.search(r"_\d{8}T\d{12}\+0800_[0-9a-f]{8}\.md$", key)
        for key in maintained.deleted_raw_keys
    )
    assert all(
        re.fullmatch(r"dir_\d{8}T\d{6}\+0800_[0-9a-f]{12}", topic.metadata.id)
        for topic in directory_topics
    )
    all_public_keys = [
        *(document.key for document in memory_documents),
        *(document.key for document in directory_topics),
        *maintained.deleted_current_keys,
        *maintained.deleted_raw_keys,
        *maintained.deleted_rewrite_keys,
        *maintained.deleted_route_keys,
    ]
    assert all(
        key.startswith(("current/", "raw/", "rewrite/", "doc/"))
        for key in all_public_keys
    )
    assert all("STORE_" not in key and "USER_" not in key for key in all_public_keys)
    assert all(
        document.metadata.store_id == scope.store_id
        and document.metadata.user_id == scope.user_id
        for document in [*memory_documents, *directory_topics]
    )
    assert all(
        timestamp.tzinfo == BEIJING_TIMEZONE
        and timestamp.utcoffset() == timedelta(hours=8)
        for document in [*memory_documents, *directory_topics]
        for timestamp in (document.metadata.created_at, document.metadata.updated_at)
    )
    assert (
        maintained.deleted_rewrite_keys[0]
        .rsplit("/", 1)[-1]
        .startswith("REWRITE_CURRENT_")
    )

    # The final layer must keep extraction metadata; no later LLM stage may turn
    # an absent time into the former `time=empty` sentinel.
    for documents in (memory_documents,):
        persisted = "\n".join(document.content for document in documents)
        assert "time=empty" not in persisted
        assert re.search(r"<seq=\d{10}(?:,|>)", persisted)
    if not use_real_llm:
        inspected_layers = [memory_documents]
        if isinstance(store, InMemoryObjectStore):
            scoped_store = flow.extractor.repository.store

            def deleted_text(key: str) -> str:
                return store.deleted_objects[f"{scoped_store.root_prefix}/{key}"]

            inspected_layers.extend(
                [
                    [
                        decode_document(deleted_text(key), key=key)
                        for key in maintained.deleted_raw_keys
                    ],
                    [
                        decode_document(deleted_text(key), key=key)
                        for key in maintained.deleted_rewrite_keys
                    ],
                ]
            )
            rewrite_document = decode_document(
                deleted_text(maintained.deleted_rewrite_keys[0]),
                key=maintained.deleted_rewrite_keys[0],
            )
            assert rewrite_document.metadata.source_ids == llm.rewrite_sources[0]
            transient_keys = [
                *maintained.deleted_current_keys,
                *maintained.deleted_raw_keys,
                *maintained.deleted_rewrite_keys,
            ]
            for key in transient_keys:
                front_matter = yaml.safe_load(
                    deleted_text(key)[4:].split("\n---\n", 1)[0]
                )
                assert "summary" not in front_matter
                assert front_matter["store_id"] == scope.store_id
                assert front_matter["user_id"] == scope.user_id
                assert not deleted_text(key)[4:].startswith("{")
        for documents in inspected_layers:
            persisted = "\n".join(document.content for document in documents)
            assert "time=2026-10-12" in persisted
            assert "time=2024-03" in persisted
            assert "label=leisure" in persisted
            assert "label=travel" in persisted
            assert "source=AI" in persisted
            assert "allergic to peanuts" in persisted
            assert "emergency contact" in persisted
        # Regression: merge output is Markdown, so legitimate backslashes are not
        # interpreted as JSON escapes.
        assert r"C:\Users\Alice\memory" in "\n".join(
            document.content for document in memory_documents
        )

    result = flow.search(SearchRequest(query="What do I like?", answer=True))
    assert result.hits
    assert result.answer

    metric_text = generate_latest(registry).decode()
    assert "flow_total" in metric_text
    assert "llm_calls_total" in metric_text
    assert "documents_total" in metric_text


def test_maintainer_batches_current_documents_by_configured_token_total() -> None:
    config = _config()
    assert MaintenanceConfig().rewrite_batch_max_tokens == 12000
    config.maintenance = MaintenanceConfig(rewrite_batch_max_tokens=4)
    store = InMemoryObjectStore()
    llm = FakeFlowLLM()
    flow = _create_flow(
        config,
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
    )
    contents = ["one two three four five", "alpha beta", "gamma delta"]
    base_created_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
    created_at_offsets = [timedelta(minutes=2), timedelta(), timedelta(minutes=1)]
    current_keys: list[str] = []
    for index, content in enumerate(contents, start=1):
        key = flow.extractor.layout.current_key(f"worker-{index}")
        current_keys.append(key)
        flow.extractor.repository.write(
            key,
            MemoryDocument(
                metadata=DocumentMetadata(
                    id=f"CURRENT_worker-{index}",
                    kind="current",
                    created_at=base_created_at + created_at_offsets[index - 1],
                ),
                content=content,
                key=key,
            ),
        )

    result = flow.maintain(MaintenanceRequest(merge_topics=False))

    assert result.current_documents == 3
    assert result.raw_documents == 3
    assert result.rewrite_documents == 2
    assert [len(source_ids) for source_ids in llm.rewrite_sources] == [2, 1]
    assert [
        [source_id.split("_")[2] for source_id in source_ids]
        for source_ids in llm.rewrite_sources
    ] == [["worker-2", "worker-3"], ["worker-1"]]
    assert result.deleted_current_keys == [
        current_keys[1],
        current_keys[2],
        current_keys[0],
    ]
    archived_created_at = {
        document.metadata.source_ids[0]: document.metadata.created_at
        for key in result.deleted_raw_keys
        for document in [
            decode_document(
                store.deleted_objects[
                    f"{flow.extractor.repository.store.root_prefix}/{key}"
                ],
                key=key,
            )
        ]
    }
    assert archived_created_at == {
        f"CURRENT_worker-{index}": base_created_at + created_at_offsets[index - 1]
        for index in range(1, 4)
    }
    assert len(result.deleted_rewrite_keys) == 2


def test_extractor_append_only_recreates_current_after_deletion() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
        instance_id="append-worker",
    )

    def extract(content: str):
        return flow.extract(
            ExtractionRequest(
                messages=[ChatMessage(role="user", content=content)],
            )
        )

    first = extract("first fact")
    second = extract("second fact")
    appended = flow.extractor.repository.read(first.current_key)
    assert second.appended is True
    assert appended.content == (
        f"{first.extracted_content}\n\n{second.extracted_content}"
    )

    flow.extractor.repository.store.delete(first.current_key)
    third = extract("third fact")
    recreated = flow.extractor.repository.read(third.current_key)

    assert third.appended is False
    assert recreated.content == third.extracted_content
    assert first.extracted_content not in recreated.content
    assert second.extracted_content not in recreated.content


def test_extractor_can_append_direct_memory_without_llm_inference() -> None:
    class UnexpectedLLM:
        def complete(self, request: LLMRequest) -> str:
            raise AssertionError(f"LLM must not be called: {request.operation}")

    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=UnexpectedLLM(),
        metrics=FlowMetrics.create(enabled=False),
        instance_id="direct-worker",
    )

    first = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="用户喜欢手冲咖啡。")],
            infer=False,
        )
    )
    second = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="用户的紧急联系人是周宁。")],
            infer=False,
        )
    )

    assert first.extracted_content.startswith(f"- <seq={first.sequence_timestamp},")
    assert first.extracted_content.endswith(",source=add_memory> 用户喜欢手冲咖啡。")
    assert second.extracted_content.startswith(f"- <seq={second.sequence_timestamp},")
    assert second.extracted_content.endswith(
        ",source=add_memory> 用户的紧急联系人是周宁。"
    )
    assert first.appended is False
    assert second.appended is True
    current = flow.extractor.repository.read(first.current_key)
    assert current.content == (
        f"{first.extracted_content}\n\n{second.extracted_content}"
    )


def test_maintainer_deletes_current_after_raw_and_preserves_new_current() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
        instance_id="handoff-worker",
    )
    first = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="first fact")],
        )
    )
    replacement = None

    def capture_handoff(stage: str, details: dict[str, object]) -> None:
        nonlocal replacement
        if stage != "archive_raw" or details.get("status") != "completed":
            return
        assert details["current_key"] == first.current_key
        assert details["deleted_current"] == 1
        assert not flow.extractor.repository.store.exists(first.current_key)
        assert len(flow.extractor.repository.list_raw()) == 1
        replacement = flow.extract(
            ExtractionRequest(
                messages=[ChatMessage(role="user", content="second fact")],
            )
        )

    result = flow.maintain(
        MaintenanceRequest(merge_topics=False),
        progress=capture_handoff,
    )

    assert replacement is not None
    assert replacement.appended is False
    assert result.deleted_current_keys == [first.current_key]
    recreated = flow.extractor.repository.read(first.current_key)
    assert recreated.content == replacement.extracted_content
    assert first.extracted_content not in recreated.content


def test_maintainer_retries_from_raw_after_rewrite_failure() -> None:
    class RewriteFailureLLM(FakeFlowLLM):
        def complete(self, request: LLMRequest) -> str:
            if request.operation == "rewrite_current":
                # RuntimeError is a recoverable exhausted-transport condition
                # and now uses the deterministic conservation fallback. Keep
                # this test focused on an unexpected programmer failure.
                raise TypeError("rewrite failed")
            return super().complete(request)

    config = _config()
    store = InMemoryObjectStore()
    failed_flow = _create_flow(
        config,
        store=store,
        llm=RewriteFailureLLM(),
        metrics=FlowMetrics.create(enabled=False),
        instance_id="retry-worker",
    )
    extracted = failed_flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="remember this")],
        )
    )

    with pytest.raises(TypeError, match="rewrite failed"):
        failed_flow.maintain(MaintenanceRequest(merge_topics=False))

    assert not failed_flow.extractor.repository.store.exists(extracted.current_key)
    assert len(failed_flow.extractor.repository.list_raw()) == 1

    retry_flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
        instance_id="retry-worker",
    )
    result = retry_flow.maintain(MaintenanceRequest(merge_topics=False))

    assert result.current_documents == 0
    assert result.raw_documents == 1
    assert result.rewrite_documents == 1
    assert result.memory_documents_created == 1
    assert retry_flow.extractor.repository.list_raw() == []
    assert retry_flow.extractor.repository.list_memory_documents()


def test_maintainer_recovers_existing_rewrite_without_raw() -> None:
    config = _config()
    store = InMemoryObjectStore()
    llm = FakeFlowLLM()
    flow = _create_flow(
        config,
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
    )
    rewrite_id = "REWRITE_CURRENT_recoverable"
    rewrite_key = f"{flow.maintainer.layout.rewrite_prefix()}/{rewrite_id}.md"
    flow.maintainer.repository.write(
        rewrite_key,
        MemoryDocument(
            metadata=DocumentMetadata(id=rewrite_id, kind="rewrite"),
            content="# Recovery\n\n- <seq=1700000000> recover me",
            key=rewrite_key,
        ),
    )
    progress_events: list[tuple[str, dict[str, object]]] = []

    result = flow.maintain(
        MaintenanceRequest(),
        progress=lambda stage, details: progress_events.append((stage, details)),
    )

    assert "route_directories" in llm.calls
    assert result.current_documents == 0
    assert result.deleted_raw_keys == []
    assert result.deleted_rewrite_keys == [rewrite_key]
    assert result.memory_documents_created == 1
    assert result.deleted_current_keys == []
    assert (
        flow.maintainer.repository.store.list_keys(
            flow.maintainer.layout.rewrite_prefix()
        )
        == []
    )
    assert progress_events[-1][0] == "cleanup_intermediates"
    assert progress_events[-1][1]["deleted_rewrite"] == 1


def test_maintainer_replays_legacy_route_without_summary_llm_call() -> None:
    class UnexpectedLLM:
        def complete(self, request: LLMRequest) -> str:
            raise AssertionError(f"unexpected LLM operation: {request.operation}")

    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=UnexpectedLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    rewrite_id = "REWRITE_CURRENT_legacy_route"
    rewrite_key = f"{flow.maintainer.layout.rewrite_prefix()}/{rewrite_id}.md"
    flow.maintainer.repository.write(
        rewrite_key,
        MemoryDocument(
            metadata=DocumentMetadata(id=rewrite_id, kind="rewrite"),
            content="# Recovery\n\n- <seq=1700000000> recover me",
            key=rewrite_key,
        ),
    )
    route_key = flow.maintainer.layout.route_key(rewrite_id)
    flow.maintainer.repository.store.put_text(
        route_key,
        json.dumps(
            {
                "rewrite_id": rewrite_id,
                "assignments": [
                    {
                        "directory_id": "dir_recovery",
                        "directory_title": "Recovery",
                        "document_id": "memory_recovery",
                        "fact_ids": ["f0001"],
                    }
                ],
            }
        ),
    )

    result = flow.maintain(MaintenanceRequest())
    memory = flow.maintainer.repository.list_memory_documents()[0]

    assert result.memory_documents_created == 1
    assert "recover me" in memory.metadata.summary
    assert "seq=" not in memory.metadata.summary


def test_maintainer_deduplicates_across_current_and_preserves_markdown() -> None:
    class DuplicateAwareLLM(FakeFlowLLM):
        def complete(self, request: LLMRequest) -> str:
            if request.operation != "rewrite_current":
                return super().complete(request)
            self.calls.append(request.operation)
            facts = self._facts(request.messages[-1].content)
            assert [fact["id"] for fact in facts] == [
                "f0001",
                "f0002",
                "f0003",
                "f0004",
                "f0005",
                "f0006",
                "f0007",
            ]
            return json.dumps(
                {
                    "topics": [
                        {
                            "title": "用户身份与偏好",
                            "fact_ids": ["f0001", "f0002", "f0003", "f0007"],
                        }
                    ],
                    "duplicates": [
                        {"discarded_id": "f0004", "retained_id": "f0001"},
                        {"discarded_id": "f0005", "retained_id": "f0002"},
                        {"discarded_id": "f0006", "retained_id": "f0003"},
                    ],
                },
                ensure_ascii=False,
            )

    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=DuplicateAwareLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    first = MemoryDocument(
        metadata=DocumentMetadata(id="CURRENT_a", kind="current"),
        content=(
            "# 用户身份与偏好\n\n"
            "- <seq=1785233061> 用户名叫 Jay\n"
            "- <seq=1785233061> 用户喜欢喝可口可乐\n"
            "- <seq=1785233061> 用户喜欢吃 **草莓**\n"
            "  - 偏好新鲜草莓"
        ),
    )
    second = MemoryDocument(
        metadata=DocumentMetadata(id="CURRENT_b", kind="current"),
        content=(
            "# 用户身份与偏好\n\n"
            "- <seq=1785233061> 用户名叫Jay\n"
            "- <seq=1785233061> 喜欢喝可口可乐\n"
            "- <seq=1785233061> 喜欢草莓\n"
            "- <seq=1785233164> 用户喜欢吃蓝莓"
        ),
    )
    for instance_id, document in (("a", first), ("b", second)):
        key = flow.maintainer.layout.current_key(instance_id)
        document.key = key
        flow.maintainer.repository.write(key, document)

    result = flow.maintain(MaintenanceRequest(merge_topics=False))

    memories = flow.maintainer.repository.list_memory_documents()
    assert result.current_documents == 2
    assert result.rewrite_documents == 1
    assert len(memories) == 1
    assert memories[0].content == (
        "# 用户身份与偏好\n\n"
        "- <seq=1785233061> 用户名叫 Jay\n"
        "- <seq=1785233061> 用户喜欢喝可口可乐\n"
        "- <seq=1785233061> 用户喜欢吃 **草莓**\n"
        "  - 偏好新鲜草莓\n"
        "- <seq=1785233164> 用户喜欢吃蓝莓"
    )
    assert "用户名叫Jay" not in memories[0].content
    assert "喜欢草莓\n" not in memories[0].content
    assert "seq=" not in memories[0].metadata.summary


def test_maintainer_splits_legacy_unbulleted_facts() -> None:
    class LegacyFactLLM(FakeFlowLLM):
        def complete(self, request: LLMRequest) -> str:
            if request.operation == "rewrite_current":
                facts = self._facts(request.messages[-1].content)
                assert [fact["markdown"] for fact in facts] == [
                    "<seq=1700000000> Likes apples.",
                    "<seq=1700000001> Likes pears.",
                ]
            return super().complete(request)

    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=LegacyFactLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    key = flow.maintainer.layout.current_key("legacy-worker")
    flow.maintainer.repository.write(
        key,
        MemoryDocument(
            metadata=DocumentMetadata(id="CURRENT_legacy-worker", kind="current"),
            content=(
                "# Preferences\n\n"
                "<seq=1700000000> Likes apples.\n"
                "<seq=1700000001> Likes pears."
            ),
            key=key,
        ),
    )

    result = flow.maintain(MaintenanceRequest(merge_topics=False))

    assert result.memory_documents_created == 1
    assert (
        "Likes apples" in flow.maintainer.repository.list_memory_documents()[0].content
    )


def test_maintainer_appends_new_document_to_existing_directory() -> None:
    config = _config()
    store = InMemoryObjectStore()
    llm = FakeFlowLLM()
    flow = _create_flow(
        config,
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
    )

    def write_current(instance_id: str, fact: str) -> None:
        key = flow.maintainer.layout.current_key(instance_id)
        flow.maintainer.repository.write(
            key,
            MemoryDocument(
                metadata=DocumentMetadata(id=f"CURRENT_{instance_id}", kind="current"),
                content=f"# Preferences\n\n- <seq=1700000000> {fact}",
                key=key,
            ),
        )

    write_current("one", "Likes apples.")
    first_result = flow.maintain(MaintenanceRequest())
    first_memories = flow.maintainer.repository.list_memory_documents()
    assert first_result.directories_created == 1
    assert len(first_memories) == 1
    original_key = first_memories[0].key
    original_bytes = flow.maintainer.repository.store.get_text(original_key)
    directory_id = first_memories[0].metadata.directory_id

    write_current("two", "Likes pears.")
    second_result = flow.maintain(MaintenanceRequest())
    memories = flow.maintainer.repository.list_memory_documents(directory_id)
    summaries = flow.maintainer.repository.list_directory_topics()

    assert second_result.directories_created == 0
    assert second_result.memory_documents_created == 1
    assert len(memories) == 2
    assert flow.maintainer.repository.store.get_text(original_key) == original_bytes
    assert len(summaries) == 1
    assert summaries[0].metadata.document_count == 2
    assert not (
        {"split_plan", "update_topic", "merge_plan", "merge_topic"} & set(llm.calls)
    )


def test_maintainer_reuses_route_and_leaf_after_topic_write_failure() -> None:
    class TopicFailureStore(InMemoryObjectStore):
        fail_topic_write: bool = True

        def put_text(self, key: str, content: str) -> None:
            if self.fail_topic_write and key.endswith("/TOPIC.md"):
                self.fail_topic_write = False
                raise RuntimeError("topic write failed")
            super().put_text(key, content)

    config = _config()
    store = TopicFailureStore()
    failed = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
        instance_id="resume-worker",
    )
    failed.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="I like apples.")],
        )
    )

    with pytest.raises(RuntimeError, match="topic write failed"):
        failed.maintain(MaintenanceRequest())

    before = failed.maintainer.repository.list_memory_documents()
    assert len(before) == 1
    assert len(failed.maintainer.repository.list_raw()) == 1
    assert len(failed.maintainer.repository.list_rewrites()) == 1
    assert any(key.endswith(".json") for key in store.objects)

    retry_llm = FakeFlowLLM()
    retry = _create_flow(
        config,
        store=store,
        llm=retry_llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="resume-worker",
    )
    result = retry.maintain(MaintenanceRequest())
    after = retry.maintainer.repository.list_memory_documents()

    assert result.memory_documents_created == 0
    assert [item.key for item in after] == [item.key for item in before]
    assert "route_directories" not in retry_llm.calls
    assert "document_summary" not in retry_llm.calls
    assert retry.maintainer.repository.list_directory_topics()
    assert retry.maintainer.repository.list_raw() == []
    assert retry.maintainer.repository.list_rewrites() == []


def test_retriever_searches_current_raw_rewrite_and_doc() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    layout = flow.retriever.repository.layout
    documents = [
        (
            layout.current_key("worker"),
            DocumentMetadata(id="CURRENT_worker", kind="current"),
            "current needle",
        ),
        (
            layout.raw_key("RAW_worker"),
            DocumentMetadata(id="RAW_worker", kind="raw"),
            "raw needle",
        ),
        (
            f"{layout.rewrite_prefix()}/REWRITE_worker.md",
            DocumentMetadata(id="REWRITE_worker", kind="rewrite"),
            "rewrite needle",
        ),
        (
            layout.evidence_key("EVIDENCE_worker"),
            DocumentMetadata(id="EVIDENCE_worker", kind="evidence"),
            "evidence needle",
        ),
        (
            layout.memory_key("dir_test", "memory_test"),
            DocumentMetadata(
                id="memory_test",
                kind="memory",
                directory_id="dir_test",
                title="Test",
                summary="Test",
            ),
            "doc needle",
        ),
        (
            layout.directory_topic_key("dir_test"),
            DocumentMetadata(
                id="dir_test",
                kind="directory_topic",
                title="Test",
                summary="doc needle",
                document_count=1,
                content_digest="digest",
            ),
            "# Test",
        ),
    ]
    for key, metadata, content in documents:
        flow.retriever.repository.write(
            key, MemoryDocument(metadata=metadata, content=content, key=key)
        )

    result = flow.search(SearchRequest(query="needle", limit=10, answer=False))
    assert {hit.kind for hit in result.hits} == {
        "current",
        "raw",
        "rewrite",
        "evidence",
        "memory",
    }
    assert all("needle" in hit.content for hit in result.hits)

    raw_only = flow.search(
        SearchRequest(
            query="needle",
            limit=10,
            sources={SearchSource.RAW},
            answer=False,
        )
    )
    assert [hit.kind for hit in raw_only.hits] == ["raw"]

    catalog_result = flow.search(
        SearchRequest(
            query="所有记忆内容",
            limit=10,
            strategy=SearchStrategy.AGENTIC,
            answer=False,
        )
    )
    assert {hit.kind for hit in catalog_result.hits} == {
        "current",
        "raw",
        "rewrite",
        "evidence",
        "memory",
    }
    assert len(catalog_result.hits) == 5

    for strategy in SearchStrategy:
        strategy_result = flow.search(
            SearchRequest(
                query="doc needle",
                limit=10,
                strategy=strategy,
                answer=False,
            )
        )
        assert strategy_result.strategy == strategy
        assert strategy_result.hits, strategy
        assert any("needle" in hit.content for hit in strategy_result.hits), strategy


def test_retriever_uses_configured_directory_limit_unless_request_overrides() -> None:
    config = _config()
    config.retrieval.directory_limit = 1
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    layout = flow.retriever.repository.layout
    for index in range(2):
        directory_id = f"dir_{index}"
        memory_id = f"memory_{index}"
        memory_key = layout.memory_key(directory_id, memory_id)
        flow.retriever.repository.write(
            memory_key,
            MemoryDocument(
                metadata=DocumentMetadata(
                    id=memory_id,
                    kind="memory",
                    directory_id=directory_id,
                    title=f"Topic {index}",
                    summary=f"Topic {index}",
                ),
                content=f"shared needle {index}",
                key=memory_key,
            ),
        )
        summary_key = layout.directory_topic_key(directory_id)
        flow.retriever.repository.write(
            summary_key,
            MemoryDocument(
                metadata=DocumentMetadata(
                    id=directory_id,
                    kind="directory_topic",
                    title=f"Topic {index}",
                    summary="shared needle",
                    document_count=1,
                    content_digest=f"digest-{index}",
                ),
                content=f"# Topic {index}",
                key=summary_key,
            ),
        )

    configured = flow.search(
        SearchRequest(query="shared needle", limit=10, answer=False)
    )
    overridden = flow.search(
        SearchRequest(
            query="shared needle",
            limit=10,
            directory_limit=2,
            answer=False,
        )
    )

    assert len(configured.hits) == 1
    assert len(overridden.hits) == 2


def test_hierarchical_retriever_skips_llm_selection_for_single_leaf() -> None:
    config = _config()
    store = InMemoryObjectStore()
    llm = FakeFlowLLM()
    flow = _create_flow(
        config,
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
    )
    directory_id = "dir_preferences"
    memory_id = "memory_preferences"
    layout = flow.retriever.repository.layout
    memory_key = layout.memory_key(directory_id, memory_id)
    flow.retriever.repository.write(
        memory_key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id=memory_id,
                kind="memory",
                directory_id=directory_id,
                title="Preferences",
                summary="Likes tea.",
            ),
            content="# Preferences\n\n- <seq=1> Likes tea.",
            key=memory_key,
        ),
    )
    topic_key = layout.directory_topic_key(directory_id)
    flow.retriever.repository.write(
        topic_key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id=directory_id,
                kind="directory_topic",
                title="Preferences",
                summary="Preferences",
                document_count=1,
            ),
            content="# Preferences",
            key=topic_key,
        ),
    )

    result = flow.search(SearchRequest(query="What does the user like?", answer=False))

    assert [hit.id for hit in result.hits] == [memory_id]
    assert "search_scope" in llm.calls
    assert "search_documents" not in llm.calls


def test_hierarchical_key_layout_rejects_unsafe_or_ambiguous_paths() -> None:
    layout = KeyLayout()
    key = layout.memory_key("dir_safe", "memory_safe")

    parsed = layout.parse_doc_key(key)
    assert (parsed.kind, parsed.directory_id, parsed.filename) == (
        "memory",
        "dir_safe",
        "memory_safe.md",
    )
    with pytest.raises(ValueError, match="reserved"):
        layout.memory_key("dir_safe", "SUMMARY")
    with pytest.raises(ValueError, match="reserved"):
        layout.memory_key("dir_safe", "TOPIC")
    with pytest.raises(ValueError, match="unsafe"):
        layout.memory_key("../escape", "memory_safe")
    with pytest.raises(ValueError, match="invalid hierarchical"):
        layout.parse_doc_key(f"{layout.doc_prefix()}/one/two/memory.md")


def test_maintainer_migrates_legacy_summary_to_heading_topic() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    directory_id = "dir_preferences"
    memory_key = flow.maintainer.layout.memory_key(directory_id, "memory_preferences")
    flow.maintainer.repository.write(
        memory_key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id="memory_preferences",
                kind="memory",
                directory_id=directory_id,
                title="Preferences",
                summary="Likes apples.",
            ),
            content="# Preferences\n\n- <seq=1700000000> Likes apples.",
            key=memory_key,
        ),
    )
    legacy_key = flow.maintainer.layout.legacy_directory_summary_key(directory_id)
    flow.maintainer.repository.write(
        legacy_key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id=directory_id,
                kind="directory_summary",
                title="Preferences",
                summary="The user likes apples.",
                document_count=1,
                content_digest="legacy-digest",
            ),
            content="# Preferences\n\nThe user likes apples.",
            key=legacy_key,
        ),
    )

    before = flow.maintainer.repository.list_directory_topics()
    assert [item.key for item in before] == [legacy_key]

    result = flow.maintain(MaintenanceRequest())
    topic_key = flow.maintainer.layout.directory_topic_key(directory_id)
    topic = flow.maintainer.repository.read(topic_key)

    assert result.directory_topics_updated == 1
    assert not flow.maintainer.repository.store.exists(legacy_key)
    assert topic.metadata.kind == "directory_topic"
    assert topic.metadata.summary == "Preferences"
    assert topic.content == "# Preferences"
    assert "apples" not in encode_document(topic).lower()


def test_legacy_topic_migration_supports_dry_run_and_verified_delete() -> None:
    config = _config()
    store = InMemoryObjectStore()
    flow = _create_flow(
        config,
        store=store,
        llm=FakeFlowLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    legacy_key = flow.maintainer.layout.topic_key("topic_legacy")
    flow.maintainer.repository.write(
        legacy_key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id="topic_legacy",
                kind="topic",
                title="Preferences",
                summary="Likes apples.",
            ),
            content="# Preferences\n\n- <seq=1700000000> Likes apples.",
            key=legacy_key,
        ),
    )

    planned = flow.migrate_legacy_topics(dry_run=True)
    assert planned.scanned == 1
    assert planned.copied == 0
    assert planned.items[0].action == "planned"
    assert not flow.maintainer.repository.store.exists(planned.items[0].target_key)

    copied = flow.migrate_legacy_topics(dry_run=False)
    assert copied.copied == 1
    assert flow.maintainer.repository.store.exists(legacy_key)
    target = flow.maintainer.repository.read(copied.items[0].target_key)
    assert target.metadata.kind == "memory"
    assert target.content.endswith("Likes apples.")
    assert flow.maintainer.repository.list_directory_topics()

    deleted = flow.migrate_legacy_topics(dry_run=False, delete_source=True)
    assert deleted.copied == 0
    assert deleted.items[0].action == "existing"
    assert deleted.deleted_sources == 1
    assert not flow.maintainer.repository.store.exists(legacy_key)


@pytest.mark.parametrize(
    "kind",
    [
        "current",
        "raw",
        "rewrite",
        "evidence",
        "memory",
        "directory_topic",
        "directory_summary",
        "topic",
    ],
)
def test_document_codec_writes_yaml_front_matter(kind: str) -> None:
    document = MemoryDocument(
        metadata=DocumentMetadata(
            id=f"{kind}_document",
            kind=kind,
            title="用户: 偏好",
            summary="A useful summary",
            directory_id="dir_preferences" if kind == "memory" else "",
            document_count=2 if kind in {"directory_topic", "directory_summary"} else 0,
            content_digest=(
                "abc123" if kind in {"directory_topic", "directory_summary"} else ""
            ),
            source_ids=["source_a", "source_b"],
        ),
        content="# 偏好\n\n- <seq=1700000000> 喜欢苹果。",
    )

    encoded = encode_document(document)
    front_matter_text = encoded[4:].split("\n---\n", 1)[0]
    front_matter = yaml.safe_load(front_matter_text)

    assert not front_matter_text.startswith("{")
    assert front_matter["id"] == f"{kind}_document"
    assert front_matter["created_at"].endswith("+08:00")
    assert front_matter["updated_at"].endswith("+08:00")
    if kind not in {"evidence", "directory_topic", "directory_summary"}:
        assert front_matter["source_ids"] == ["source_a", "source_b"]
    if kind in {
        "evidence",
        "topic",
        "memory",
        "directory_topic",
        "directory_summary",
    }:
        assert front_matter["summary"] == "A useful summary"
    else:
        assert "summary" not in front_matter
    decoded = decode_document(encoded)
    assert decoded.content == document.content
    assert decoded.metadata.summary == (
        "A useful summary"
        if kind
        in {"evidence", "topic", "memory", "directory_topic", "directory_summary"}
        else ""
    )


def test_document_codec_reads_legacy_json_front_matter() -> None:
    legacy = (
        "---\n"
        + json.dumps(
            {
                "id": "CURRENT_legacy",
                "kind": "current",
                "summary": "legacy summary",
                "title": "",
                "created_at": "2026-07-29T00:00:00Z",
                "updated_at": "2026-07-29T00:00:00Z",
                "source_ids": [],
            }
        )
        + "\n---\n# Legacy\n\n- <seq=1700000000> fact\n"
    )

    decoded = decode_document(legacy, key="legacy.md")

    assert decoded.metadata.id == "CURRENT_legacy"
    assert decoded.metadata.summary == "legacy summary"
    assert decoded.metadata.created_at.isoformat() == "2026-07-29T08:00:00+08:00"
    assert decoded.metadata.updated_at.isoformat() == "2026-07-29T08:00:00+08:00"
    assert "fact" in decoded.content
    migrated = encode_document(decoded)
    migrated_front_matter = migrated[4:].split("\n---\n", 1)[0]
    assert not migrated_front_matter.startswith("{")
    migrated_metadata = yaml.safe_load(migrated_front_matter)
    assert "summary" not in migrated_metadata
    assert migrated_metadata["created_at"] == "2026-07-29T08:00:00+08:00"
    assert migrated_metadata["updated_at"] == "2026-07-29T08:00:00+08:00"


def test_document_metadata_normalizes_recorded_times_to_beijing() -> None:
    metadata = DocumentMetadata(
        id="CURRENT_timezone",
        kind="current",
        created_at=datetime(2026, 7, 29, 1, 2, 3, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 29, 9, 2, 3),
    )

    assert metadata.created_at.isoformat() == "2026-07-29T09:02:03+08:00"
    assert metadata.updated_at.isoformat() == "2026-07-29T09:02:03+08:00"
    assert metadata.created_at.tzinfo == BEIJING_TIMEZONE
    assert metadata.updated_at.tzinfo == BEIJING_TIMEZONE

    metadata.updated_at = datetime(2026, 7, 29, 2, 2, 3, tzinfo=timezone.utc)
    assert metadata.updated_at.isoformat() == "2026-07-29T10:02:03+08:00"
    assert re.fullmatch(r"\d{8}T\d{6}\+0800", compact_beijing_timestamp())
    assert re.fullmatch(
        r"\d{8}T\d{12}\+0800",
        compact_beijing_timestamp(microseconds=True),
    )


def test_strip_yaml_front_matter_preserves_only_document_body() -> None:
    persisted = (
        "---\n"
        "id: hidden-id\n"
        "kind: current\n"
        "frontmatter_secret: never-send\n"
        "---\n"
        "# Preferences\n\n"
        "- <seq=1700000000> Likes apples.\n"
    )

    assert strip_yaml_front_matter(persisted) == (
        "# Preferences\n\n- <seq=1700000000> Likes apples."
    )
    assert strip_yaml_front_matter("---\nnot a mapping\n---\nbody") == (
        "---\nnot a mapping\n---\nbody"
    )


def test_all_document_prompts_exclude_yaml_front_matter() -> None:
    """Every LLM flow receives body text, even for a nested persisted envelope."""

    class PromptAuditLLM(FakeFlowLLM):
        def __init__(self) -> None:
            super().__init__()
            self.prompts: list[tuple[str, str]] = []

        def complete(self, request: LLMRequest) -> str:
            self.prompts.append((request.operation, request.messages[-1].content))
            return super().complete(request)

    config = _config()
    store = InMemoryObjectStore()
    llm = PromptAuditLLM()
    flow = _create_flow(
        config,
        store=store,
        llm=llm,
        metrics=FlowMetrics.create(enabled=False),
        instance_id="prompt-audit",
    )
    key = flow.maintainer.layout.current_key("worker")
    nested_persisted_document = (
        "---\n"
        "id: INNER_FRONTMATTER_SENTINEL\n"
        "kind: current\n"
        "created_at: 1999-01-01T00:00:00Z\n"
        "updated_at: 1999-01-01T00:00:00Z\n"
        "frontmatter_secret: never-send\n"
        "---\n"
        "# Preferences\n\n"
        "- <seq=1700000000> Likes apples."
    )
    flow.maintainer.repository.write(
        key,
        MemoryDocument(
            metadata=DocumentMetadata(id="CURRENT_worker", kind="current"),
            content=nested_persisted_document,
            key=key,
        ),
    )

    fresh = flow.search(SearchRequest(query="apples", answer=True))
    maintained = flow.maintain(MaintenanceRequest())
    organized = flow.search(SearchRequest(query="apples", answer=True))

    assert fresh.hits and fresh.answer
    assert maintained.memory_documents_created == 1
    assert organized.hits and organized.answer
    expected_operations = {
        "rewrite_current",
        "route_directories",
        "search_scope",
        "search_answer",
    }
    assert expected_operations <= {operation for operation, _ in llm.prompts}
    for operation, prompt in llm.prompts:
        if operation in expected_operations:
            assert "INNER_FRONTMATTER_SENTINEL" not in prompt
            assert "frontmatter_secret" not in prompt
            assert "1999-01-01T00:00:00Z" not in prompt
    assert "directory_topic" not in {operation for operation, _ in llm.prompts}


def test_extractor_rotates_current_after_token_threshold() -> None:
    config = _config()
    first_extraction = (
        "# Identity and preferences\n\n"
        "- <seq=1700000000> The user's name is Alice.\n"
        "- <seq=1700000000,label=leisure> The user likes apples and jazz.\n"
        "- <seq=1700000000,source=AI> The user's preferred language is Chinese."
    )
    one_fact_tokens = estimate_tokens(first_extraction)
    config.extraction = ExtractionConfig(current_max_tokens=one_fact_tokens + 1)
    registry = CollectorRegistry()
    metrics = FlowMetrics.create(
        namespace=f"mem_flow_rotation_{uuid4().hex[:8]}", registry=registry
    )
    store = InMemoryObjectStore()
    flow = _create_flow(config, store=store, llm=FakeFlowLLM(), metrics=metrics)

    def request(number: int) -> ExtractionRequest:
        return ExtractionRequest(
            messages=[ChatMessage(role="user", content=f"fact {number}")],
        )

    first = flow.extract(request(1))
    second = flow.extract(request(2))

    assert first.rotated_key is None
    assert second.rotated_key is not None
    assert second.rotated_key.endswith(f"CURRENT_{flow.instance_id}_full_1.md")
    assert not flow.extractor.repository.store.exists(second.current_key)
    assert flow.extractor.repository.store.exists(second.rotated_key)
    archived = flow.extractor.repository.read(second.rotated_key)
    assert archived.metadata.id == f"CURRENT_{flow.instance_id}_full_1"
    assert "The user's name is Alice" in archived.content
    assert "The user will visit Kyoto" in archived.content


def test_openai_flow_llm_retries_empty_content() -> None:
    class EmptyThenSuccessCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            content = "" if self.calls == 1 else "remembered"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    completions = EmptyThenSuccessCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    metrics = FlowMetrics.create(enabled=False)
    llm = OpenAIFlowLLM.create(
        LLMConfig(retry_attempts=2, retry_initial_seconds=0),
        metrics,
        client=client,
    )

    result = llm.complete(
        LLMRequest(
            operation="extract",
            messages=[ChatMessage(role="user", content="remember this")],
        )
    )

    assert result == "remembered"
    assert completions.calls == 2


def test_extractor_accepts_no_memory_sentinel() -> None:
    class NoMemoryLLM:
        def complete(self, _request: LLMRequest) -> str:
            return "NO_MEMORY"

    config = _config()
    store = InMemoryObjectStore()
    metrics = FlowMetrics.create(enabled=False)
    flow = _create_flow(config, store=store, llm=NoMemoryLLM(), metrics=metrics)

    result = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert result.extracted_content == ""
    assert result.bytes_written == 0
    assert not flow.extractor.repository.store.exists(result.current_key)
    evidence = flow.extractor.repository.read(result.evidence_key)
    assert evidence.metadata.kind == "evidence"
    assert "[USER]\nhello" in evidence.content


def test_extractor_accepts_explicit_logical_sequence_without_changing_default() -> None:
    class ExplicitSequenceLLM:
        def __init__(self) -> None:
            self.prompt = ""

        def complete(self, request: LLMRequest) -> str:
            self.prompt = request.messages[-1].content
            return "# Memory\n\n- <seq=0> remember this"

    config = _config()
    store = InMemoryObjectStore()
    metrics = FlowMetrics.create(enabled=False)
    llm = ExplicitSequenceLLM()
    flow = _create_flow(config, store=store, llm=llm, metrics=metrics)

    result = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="remember this")],
            sequence=7,
        )
    )

    assert result.sequence_timestamp == 7
    assert "`- <seq=7> fact`" in llm.prompt
    assert "<seq=7,origin=EVIDENCE_" in result.extracted_content
    assert "> remember this" in result.extracted_content


def test_extractor_normalizes_model_owned_sequence_and_empty_time() -> None:
    class InvalidMetadataLLM:
        def complete(self, _request: LLMRequest) -> str:
            return (
                "# Preference\n\n"
                "<seq=0,time=empty,label=food> Likes apples.\n"
                "<seq=0> Likes pears."
            )

    config = _config()
    store = InMemoryObjectStore()
    metrics = FlowMetrics.create(enabled=False)
    flow = _create_flow(config, store=store, llm=InvalidMetadataLLM(), metrics=metrics)

    result = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="I like apples.")],
        )
    )

    assert (
        f"<seq={result.sequence_timestamp},origin=EVIDENCE_" in result.extracted_content
    )
    assert ",label=food>" in result.extracted_content
    fact_lines = [
        line for line in result.extracted_content.splitlines() if "<seq=" in line
    ]
    assert len(fact_lines) == 2
    assert all(line.startswith("- <seq=") for line in fact_lines)
    assert "seq=0" not in result.extracted_content
    assert "time=empty" not in result.extracted_content
    persisted = flow.extractor.repository.read(result.current_key)
    assert persisted.content == result.extracted_content


def test_extractor_promotes_nested_model_headings_to_h1() -> None:
    class NestedHeadingLLM:
        def complete(self, _request: LLMRequest) -> str:
            return (
                "# Relationships\n"
                "- <seq=0> The user has a partner.\n"
                "## Relationships\n"
                "- <seq=0> The user has a friend named Alex.\n"
                "### Knowledge\n"
                "- <seq=0> Alex shared a secret BBQ recipe.\n"
                "###### Goals\n"
                "- <seq=0> The user plans to ask Alex for the recipe.\n"
                "```markdown\n"
                "## Keep this fenced example unchanged\n"
                "```"
            )

    flow = _create_flow(
        _config(),
        store=InMemoryObjectStore(),
        llm=NestedHeadingLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )

    result = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="Remember these facts.")],
        )
    )

    assert extract_h1_headings(result.extracted_content) == [
        "Relationships",
        "Knowledge",
        "Goals",
    ]
    assert "## Relationships" not in result.extracted_content
    assert "### Knowledge" not in result.extracted_content
    assert "###### Goals" not in result.extracted_content
    assert "## Keep this fenced example unchanged" in result.extracted_content


def test_maintainer_promotes_nested_headings_and_indexes_all_leaf_h1s() -> None:
    class SingleDirectoryLLM(FakeFlowLLM):
        def complete(self, request: LLMRequest) -> str:
            if request.operation != "route_directories":
                return super().complete(request)
            self.calls.append(request.operation)
            facts = self._facts(request.messages[-1].content)
            return json.dumps(
                {
                    "assignments": [
                        {
                            "directory_id": None,
                            "new_directory_title": "People and plans",
                            "fact_ids": [fact["id"] for fact in facts],
                            "summary": " ".join(
                                re.sub(r"^[-*+]\s+<[^>]+>\s*", "", fact["markdown"])
                                for fact in facts
                            ),
                        }
                    ]
                }
            )

    flow = _create_flow(
        _config(),
        store=InMemoryObjectStore(),
        llm=SingleDirectoryLLM(),
        metrics=FlowMetrics.create(enabled=False),
        instance_id="mixed-heading-worker",
    )
    current_key = flow.maintainer.layout.current_key(flow.instance_id)
    flow.maintainer.repository.write(
        current_key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id=f"CURRENT_{flow.instance_id}",
                kind="current",
            ),
            content=(
                "# Relationships\n"
                "- <seq=1785467051> User's sister recommended a true crime "
                "podcast to the user.\n"
                "- <seq=1785467154> The user has a partner.\n"
                "- <seq=1785467154> The user plans to look into a small luxury "
                "ship for a romantic getaway with their partner.\n"
                "- <seq=1785467196> The user attended a barbecue party at "
                "Alex's house three weeks ago where they had slow-cooked ribs.\n"
                "## Relationships\n"
                "- <seq=1785467196> The user has a friend named Alex.\n"
                "## Knowledge\n"
                "- <seq=1785467196> The user remembers Alex telling them that "
                "his grandfather's secret recipe involved a dry rub marinated "
                "for 24 hours before grilling.\n"
                "## Goals\n"
                "- <seq=1785467196> The user has been meaning to ask Alex for "
                "his secret BBQ recipe.\n"
                "- <seq=1785467280> User watched a World Series game with their dad."
            ),
            key=current_key,
        ),
    )

    result = flow.maintain(MaintenanceRequest())

    assert result.memory_documents_created == 1
    memories = flow.maintainer.repository.list_memory_documents()
    topics = flow.maintainer.repository.list_directory_topics()
    assert len(memories) == len(topics) == 1
    expected_headings = ["Relationships", "Knowledge", "Goals"]
    assert extract_h1_headings(memories[0].content) == expected_headings
    assert extract_h1_headings(topics[0].content) == expected_headings
    assert topics[0].content == "\n".join(
        f"# {heading}" for heading in expected_headings
    )
    assert all(
        not re.search(r"^#{2,6}\s+", memory.content, re.MULTILINE)
        for memory in memories
    )
    for topic in topics:
        leaf_headings = {
            heading
            for memory in flow.maintainer.repository.list_memory_documents(
                topic.metadata.id
            )
            for heading in extract_h1_headings(memory.content)
        }
        assert set(extract_h1_headings(topic.content)) == leaf_headings
