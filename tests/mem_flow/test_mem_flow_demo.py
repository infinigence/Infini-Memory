"""Opt-in real-service demo for the complete multi-instance mem_flow lifecycle."""

from __future__ import annotations

import json
import logging
import os
import tomllib
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from mem_flow import (
    ChatMessage,
    DocLineUpdateRequest,
    DocReadRequest,
    ExtractionRequest,
    FlowMetrics,
    LLMConfig,
    MaintenanceConfig,
    MaintenanceRequest,
    MemFlow,
    MemFlowConfig,
    MemoryScope,
    RetrievalConfig,
    S3Config,
    SearchRequest,
    SearchStrategy,
)
from mem_flow.hierarchy import extract_h1_headings
from mem_flow.llm import OpenAIFlowLLM
from mem_flow.models import DocumentMetadata, MemoryDocument
from mem_flow.storage import S3ObjectStore
from mem_flow.utils.codec import strip_yaml_front_matter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _assert_relative_key(key: str) -> None:
    assert key.startswith(("current/", "raw/", "rewrite/", "doc/"))
    assert not key.startswith("/")
    assert "STORE_" not in key
    assert "USER_" not in key


def _demo_config_path(pytestconfig: pytest.Config) -> Path:
    configured = pytestconfig.getoption("--mem-flow-demo-config")
    configured = configured or os.getenv("AI_DEMO_CONFIG_PATH", "")
    return (
        Path(configured).expanduser()
        if configured
        else (PROJECT_ROOT / "config" / "ai_demo_config.toml")
    )


def _load_ai_demo_config(path: Path) -> MemFlowConfig:
    if not path.is_file():
        pytest.fail(f"AI Demo config does not exist: {path}")

    with path.open("rb") as config_file:
        data = tomllib.load(config_file)

    llm_config = data.get("llm", {})
    ai_config = data.get("ai", {})
    default_model_id = str(ai_config.get("default_model", "") or "")
    selected_model = next(
        (
            model
            for model in ai_config.get("models", [])
            if model.get("id") == default_model_id
        ),
        {},
    )
    api_key = selected_model.get("api_key") or llm_config.get("openai_api_key", "")
    base_url = selected_model.get("base_url") or llm_config.get("openai_base_url", "")
    model = selected_model.get("model") or llm_config.get("model", "deepseek-v4-flash-0731")

    storage_config = data.get("storage", {})
    endpoint = str(storage_config.get("s3_endpoint", "") or "").strip()
    missing = [
        name
        for name, value in (
            ("llm.openai_api_key", api_key),
            ("llm.openai_base_url", base_url),
            ("llm.model", model),
            ("storage.s3_endpoint", endpoint),
        )
        if not value
    ]
    if missing:
        pytest.fail(
            "AI Demo config is missing required real-service values: "
            + ", ".join(missing)
        )

    memory_config = data.get("memory", {})
    try:
        configured_strategy = SearchStrategy(
            memory_config.get("search_strategy", SearchStrategy.HIERARCHICAL.value)
        )
    except ValueError:
        configured_strategy = SearchStrategy.HIERARCHICAL
    return MemFlowConfig(
        s3=S3Config(
            endpoint_url=endpoint,
            bucket=storage_config.get("s3_bucket", "infini-memory") or "infini-memory",
            access_key=storage_config.get("s3_access_key", ""),
            secret_key=storage_config.get("s3_secret_key", ""),
            fixed_prefix=storage_config.get("s3_prefix", "inf_mem") or "inf_mem",
        ),
        llm=LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
        ),
        maintenance=MaintenanceConfig(
            rewrite_batch_max_tokens=memory_config.get(
                "rewrite_batch_max_tokens", 12000
            )
        ),
        retrieval=RetrievalConfig(
            strategy=configured_strategy,
            search_folder=memory_config.get("search_folder", "doc") or "doc",
        ),
    )


def _delete_scope(
    store: S3ObjectStore, config: MemFlowConfig, scope: MemoryScope
) -> None:
    scope_prefix = (
        f"{config.s3.fixed_prefix}/STORE_{scope.store_id}/USER_{scope.user_id}/"
    )
    for key in store.list_keys(scope_prefix):
        store.delete(key)


def test_real_ai_demo_multi_instance_flow(
    pytestconfig: pytest.Config, caplog: pytest.LogCaptureFixture
) -> None:
    """Extract, maintain, and retrieve facts written by three instance IDs."""
    if not pytestconfig.getoption("--mem-flow-real-demo"):
        pytest.skip("pass --mem-flow-real-demo to use the AI Demo LLM and S3")

    for logger_name in (
        "boto3",
        "botocore",
        "httpcore",
        "httpx",
        "openai",
        "urllib3",
    ):
        caplog.set_level(logging.WARNING, logger=logger_name)
    for logger_name in ("mem_flow.repository", "mem_flow.storage"):
        caplog.set_level(logging.INFO, logger=logger_name)

    config_path = _demo_config_path(pytestconfig)
    config = _load_ai_demo_config(config_path)
    metrics = FlowMetrics.create(enabled=False)
    store = S3ObjectStore.create(config.s3, metrics)
    llm = OpenAIFlowLLM.create(config.llm, metrics)
    run_id = uuid4().hex
    scope = MemoryScope(store_id="mem_flow_demo_test", user_id="multi_instance")
    instance_ids = [f"demo-worker-{index}-{run_id[:8]}" for index in range(1, 4)]
    flows = [
        MemFlow.create(
            config,
            store_id=scope.store_id,
            user_id=scope.user_id,
            store=store,
            llm=llm,
            metrics=metrics,
            instance_id=instance_id,
        )
        for instance_id in instance_ids
    ]
    conversations = [
        "我叫林岚，最喜欢蓝莓松饼，平时更喜欢喝无糖拿铁。",
        "我的宠物是一只橘猫，名字叫豆包；我每周六早上带它去宠物公园。",
        "我计划在2026年10月15日去京都旅行，并预订靠近京都站的酒店。",
    ]

    test_store_root = f"{config.s3.fixed_prefix}/STORE_mem_flow_demo_test"
    fixed_scope_prefix = f"{test_store_root}/USER_{scope.user_id}/"
    for old_key in store.list_keys(test_store_root):
        if old_key.startswith(fixed_scope_prefix) or "/USER_multi_instance_" in old_key:
            store.delete(old_key)
    try:
        extractions = [
            flow.extract(
                ExtractionRequest(
                    messages=[ChatMessage(role="user", content=conversation)],
                )
            )
            for flow, conversation in zip(flows, conversations, strict=True)
        ]
        direct_memory = flows[0].extract(
            ExtractionRequest(
                messages=[ChatMessage(role="user", content="用户的紧急联系人是周宁。")],
                infer=False,
            )
        )

        assert {result.instance_id for result in extractions} == set(instance_ids)
        assert all(result.extracted_content for result in extractions)
        for result in [*extractions, direct_memory]:
            _assert_relative_key(result.current_key)
            if result.rotated_key:
                _assert_relative_key(result.rotated_key)
        assert direct_memory.appended is True
        assert direct_memory.extracted_content == (
            f"- <seq={direct_memory.sequence_timestamp},source=add_memory> "
            "用户的紧急联系人是周宁。"
        )
        current_documents = flows[0].extractor.repository.list_current()
        assert len(current_documents) == 3
        assert {document.metadata.id for document in current_documents} == {
            f"CURRENT_{instance_id}" for instance_id in instance_ids
        }
        current_content = "\n".join(document.content for document in current_documents)
        assert all(term in current_content for term in ("蓝莓", "豆包", "京都", "周宁"))
        assert "source=add_memory" in current_content
        for document in current_documents:
            fact_lines = [
                line for line in document.content.splitlines() if "<seq=" in line
            ]
            assert fact_lines
            assert all(line.startswith("- <seq=") for line in fact_lines)
            persisted = flows[0].extractor.repository.store.get_text(document.key)
            front_matter_text = persisted[4:].split("\n---\n", 1)[0]
            assert not front_matter_text.startswith("{")
            front_matter = yaml.safe_load(front_matter_text)
            assert "summary" not in front_matter
            assert front_matter["store_id"] == scope.store_id
            assert front_matter["user_id"] == scope.user_id

        fresh_search = flows[0].search(
            SearchRequest(
                query="我的宠物叫什么名字，它是什么动物？",
                answer=False,
            )
        )
        assert fresh_search.hits
        assert any(
            hit.kind == "current" and "豆包" in hit.content for hit in fresh_search.hits
        )

        maintenance = flows[0].maintain(MaintenanceRequest())
        assert maintenance.current_documents == 3
        assert maintenance.raw_documents == 3
        assert maintenance.rewrite_documents >= 1
        assert maintenance.directories_created >= 1
        assert maintenance.memory_documents_created >= 1
        assert maintenance.directory_topics_updated >= 1
        assert len(maintenance.deleted_current_keys) == 3
        for key in [
            *maintenance.deleted_current_keys,
            *maintenance.deleted_raw_keys,
            *maintenance.deleted_rewrite_keys,
            *maintenance.deleted_route_keys,
        ]:
            _assert_relative_key(key)
        assert flows[0].extractor.repository.list_current() == []
        memories = flows[0].extractor.repository.list_memory_documents()
        topics = flows[0].extractor.repository.list_directory_topics()
        assert memories
        assert topics
        for document in [*memories, *topics]:
            front_matter_text = (
                flows[0]
                .extractor.repository.store.get_text(document.key)[4:]
                .split("\n---\n", 1)[0]
            )
            assert not front_matter_text.startswith("{")
            front_matter = yaml.safe_load(front_matter_text)
            assert front_matter["summary"]
            assert front_matter["store_id"] == scope.store_id
            assert front_matter["user_id"] == scope.user_id
            _assert_relative_key(document.key)
        for topic in topics:
            assert topic.key.endswith("/TOPIC.md")
            assert topic.metadata.kind == "directory_topic"
            assert topic.metadata.summary == topic.metadata.title
            directory_memories = flows[0].extractor.repository.list_memory_documents(
                topic.metadata.id
            )
            expected_headings: list[str] = []
            for memory in directory_memories:
                for heading in extract_h1_headings(memory.content):
                    if heading.casefold() not in {
                        item.casefold() for item in expected_headings
                    }:
                        expected_headings.append(heading)
            assert extract_h1_headings(topic.content) == expected_headings
            assert topic.content.splitlines() == [
                f"# {heading}" for heading in expected_headings
            ]
            assert "---" not in topic.content
            assert "<seq=" not in topic.content
        memory_content = "\n".join(document.content for document in memories)
        assert all(term in memory_content for term in ("蓝莓", "豆包", "京都", "周宁"))
        assert "source=add_memory" in memory_content
        assert all(
            topic.metadata.document_count
            == len(
                flows[0].extractor.repository.list_memory_documents(topic.metadata.id)
            )
            for topic in topics
        )

        maintained_search = flows[0].search(
            SearchRequest(
                query="我的宠物叫什么名字，它是什么动物？",
                answer=True,
            )
        )
        assert maintained_search.hits
        assert all(not hit.key.startswith("/") for hit in maintained_search.hits)
        assert all("STORE_" not in hit.key for hit in maintained_search.hits)
        assert all("USER_" not in hit.key for hit in maintained_search.hits)
        assert any(hit.kind == "memory" for hit in maintained_search.hits)
        assert maintained_search.answer
        assert "豆包" in maintained_search.answer

        strategy_searches = {}
        for strategy in SearchStrategy:
            strategy_search = flows[0].search(
                SearchRequest(
                    query="我的宠物叫什么名字，它是什么动物？",
                    strategy=strategy,
                    answer=False,
                )
            )
            assert strategy_search.strategy == strategy
            assert strategy_search.hits, strategy
            for hit in strategy_search.hits:
                _assert_relative_key(hit.key)
            assert any("豆包" in hit.content for hit in strategy_search.hits), strategy
            strategy_searches[strategy.value] = [
                {"id": hit.id, "kind": hit.kind} for hit in strategy_search.hits
            ]

        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "scope": scope.model_dump(),
                    "instance_ids": instance_ids,
                    "current_documents": maintenance.current_documents,
                    "rewrite_documents": maintenance.rewrite_documents,
                    "directories_created": maintenance.directories_created,
                    "memory_documents_created": maintenance.memory_documents_created,
                    "direct_memory": direct_memory.model_dump(),
                    "directory_titles": [doc.metadata.title for doc in topics],
                    "topic_headings": {
                        doc.metadata.id: extract_h1_headings(doc.content)
                        for doc in topics
                    },
                    "retrieval_answer": maintained_search.answer,
                    "retrievals_by_strategy": strategy_searches,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        # Keep this run for inspection. The next real demo removes this fixed scope.
        pass


def test_real_ai_demo_topic_collects_multiple_headings(
    pytestconfig: pytest.Config, caplog: pytest.LogCaptureFixture
) -> None:
    """Repair a multi-heading TOPIC in S3 and retrieve through it with the real LLM."""

    if not pytestconfig.getoption("--mem-flow-real-demo"):
        pytest.skip("pass --mem-flow-real-demo to use the AI Demo LLM and S3")

    for logger_name in (
        "boto3",
        "botocore",
        "httpcore",
        "httpx",
        "openai",
        "urllib3",
    ):
        caplog.set_level(logging.WARNING, logger=logger_name)

    config_path = _demo_config_path(pytestconfig)
    config = _load_ai_demo_config(config_path)
    metrics = FlowMetrics.create(enabled=False)
    store = S3ObjectStore.create(config.s3, metrics)
    llm = OpenAIFlowLLM.create(config.llm, metrics)
    flow = MemFlow.create(
        config,
        store_id="mem_flow_demo_topic_test",
        user_id="multi_topic",
        store=store,
        llm=llm,
        metrics=metrics,
        instance_id="demo-topic-worker",
    )
    scope = MemoryScope(store_id="mem_flow_demo_topic_test", user_id="multi_topic")
    _delete_scope(store, config, scope)

    repository = flow.maintainer.repository
    layout = flow.maintainer.layout
    directory_id = "dir_multi_topic"
    leaves = [
        MemoryDocument(
            metadata=DocumentMetadata(
                id="memory_preferences_a",
                kind="memory",
                directory_id=directory_id,
                title="用户偏好",
                summary="用户喜欢蓝莓松饼和无糖拿铁。",
                source_ids=["REWRITE_TOPIC_DEMO_A"],
            ),
            content=(
                "# 饮食偏好\n\n"
                "- <seq=1785312001> 用户喜欢蓝莓松饼。\n\n"
                "# 饮品偏好\n\n"
                "- <seq=1785312002> 用户喜欢无糖拿铁。"
            ),
        ),
        MemoryDocument(
            metadata=DocumentMetadata(
                id="memory_preferences_b",
                kind="memory",
                directory_id=directory_id,
                title="用户偏好",
                summary="用户喜欢爵士乐和绿茶。",
                source_ids=["REWRITE_TOPIC_DEMO_B"],
            ),
            content=(
                "# 音乐偏好\n\n"
                "- <seq=1785312003> 用户喜欢爵士乐。\n\n"
                "# 饮品偏好\n\n"
                "- <seq=1785312004> 用户也喜欢绿茶。"
            ),
        ),
    ]
    leaf_bytes: dict[str, str] = {}
    for leaf in leaves:
        key = layout.memory_key(directory_id, leaf.metadata.id)
        leaf.key = key
        repository.write(key, leaf)
        leaf_bytes[key] = repository.store.get_text(key)

    topic_key = layout.directory_topic_key(directory_id)
    repository.write(
        topic_key,
        MemoryDocument(
            metadata=DocumentMetadata(
                id=directory_id,
                kind="directory_topic",
                title="用户偏好",
                summary="用户偏好",
                document_count=1,
                content_digest="stale-before-multi-heading-repair",
            ),
            content="# 饮食偏好",
            key=topic_key,
        ),
    )

    maintenance = flow.maintain(MaintenanceRequest())
    topic = repository.read(topic_key)
    expected_headings = ["饮食偏好", "饮品偏好", "音乐偏好"]

    assert maintenance.current_documents == 0
    assert maintenance.memory_documents_created == 0
    assert maintenance.directory_topics_updated == 1
    assert topic.content == "\n".join(f"# {heading}" for heading in expected_headings)
    assert extract_h1_headings(topic.content) == expected_headings
    assert topic.metadata.document_count == 2
    assert all(
        repository.store.get_text(key) == content for key, content in leaf_bytes.items()
    )
    persisted_topic = repository.store.get_text(topic_key)
    topic_front_matter = yaml.safe_load(persisted_topic[4:].split("\n---\n", 1)[0])
    assert topic_front_matter["kind"] == "directory_topic"
    assert topic_front_matter["store_id"] == scope.store_id
    assert topic_front_matter["user_id"] == scope.user_id
    _assert_relative_key(topic_key)
    for key in leaf_bytes:
        _assert_relative_key(key)
    assert strip_yaml_front_matter(persisted_topic) == topic.content

    doc_before = flow.get_doc(DocReadRequest(document_id="memory_preferences_b"))
    assert set(doc_before.model_dump()) == {
        "document_id",
        "content",
        "line_count",
    }
    assert "绿茶" in doc_before.content
    assert not doc_before.content.startswith("---")
    green_tea_line = next(
        index
        for index, line in enumerate(doc_before.content.splitlines(), start=1)
        if "绿茶" in line
    )
    topic_digest_before_edit = topic.metadata.content_digest
    doc_update = flow.update_doc_lines(
        DocLineUpdateRequest(
            document_id="memory_preferences_b",
            start_line=green_tea_line,
            end_line=green_tea_line,
            replacement="- <seq=1785312004> 用户也喜欢乌龙茶。",
        )
    )
    doc_after = flow.get_doc(DocReadRequest(document_id="memory_preferences_b"))
    assert doc_update.changed is True
    assert doc_update.document == doc_after
    assert "乌龙茶" in doc_after.content
    assert "绿茶" not in doc_after.content
    assert not doc_after.content.startswith("---")
    edited_key = layout.memory_key(directory_id, "memory_preferences_b")
    front_matter_before_edit = yaml.safe_load(
        leaf_bytes[edited_key][4:].split("\n---\n", 1)[0]
    )
    front_matter_after_edit = yaml.safe_load(
        repository.store.get_text(edited_key)[4:].split("\n---\n", 1)[0]
    )
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
        assert front_matter_after_edit[field] == front_matter_before_edit[field]
    assert (
        repository.store.get_text(edited_key)[4:].split("\n---\n", 1)[0]
        == leaf_bytes[edited_key][4:].split("\n---\n", 1)[0]
    )
    topic = repository.read(topic_key)
    assert topic.metadata.content_digest != topic_digest_before_edit
    assert extract_h1_headings(topic.content) == expected_headings

    retrieval = flow.search(
        SearchRequest(
            query="用户喜欢喝什么？",
            strategy=SearchStrategy.HIERARCHICAL,
            answer=True,
        )
    )
    assert retrieval.hits
    for hit in retrieval.hits:
        _assert_relative_key(hit.key)
    assert any(
        "无糖拿铁" in hit.content or "乌龙茶" in hit.content for hit in retrieval.hits
    )
    assert retrieval.answer
    assert "无糖拿铁" in retrieval.answer or "乌龙茶" in retrieval.answer

    catalog_retrieval = flow.search(
        SearchRequest(
            query="所有记忆内容",
            strategy=SearchStrategy.AGENTIC,
            limit=10,
            answer=True,
        )
    )
    assert {hit.id for hit in catalog_retrieval.hits} == {
        leaf.metadata.id for leaf in leaves
    }
    for hit in catalog_retrieval.hits:
        _assert_relative_key(hit.key)
    catalog_content = "\n".join(hit.content for hit in catalog_retrieval.hits)
    assert all(
        fact in catalog_content
        for fact in ("蓝莓松饼", "无糖拿铁", "爵士乐", "乌龙茶")
    )
    assert catalog_retrieval.answer

    print(
        json.dumps(
            {
                "config": str(config_path),
                "scope": scope.model_dump(),
                "model": config.llm.model,
                "topic_key": topic_key,
                "topic_headings": extract_h1_headings(topic.content),
                "topic_document_count": topic.metadata.document_count,
                "doc_read": doc_before.model_dump(),
                "doc_update": doc_update.model_dump(),
                "memory_documents": [
                    {
                        "id": leaf.metadata.id,
                        "headings": extract_h1_headings(leaf.content),
                    }
                    for leaf in leaves
                ],
                "retrieval_hits": [
                    {"id": hit.id, "kind": hit.kind} for hit in retrieval.hits
                ],
                "retrieval_answer": retrieval.answer,
                "catalog_retrieval_hits": [
                    {"id": hit.id, "kind": hit.kind} for hit in catalog_retrieval.hits
                ],
                "catalog_retrieval_answer": catalog_retrieval.answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
