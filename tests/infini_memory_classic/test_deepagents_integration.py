"""Tests for the deepagents integration module.

Tests are skipped if the ``deepagents`` package is not installed.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _project_root()
sys.path.insert(0, str(ROOT / "src"))

deepagents_mod = pytest.importorskip("deepagents")


from infini_memory_classic.deepagents_integration import (  # noqa: E402
    create_memory_agent,
    create_memory_tools,
)


def _mock_memory() -> MagicMock:
    m = MagicMock()
    m.add.return_value = 1
    m.search.return_value = {"query": "test", "results": []}
    m.get.return_value = {"id": "doc1", "summary": "hello", "content": "world"}
    m.list.return_value = [{"id": "doc1", "summary": "hello"}]
    m.update.return_value = {"id": "doc1", "summary": "updated", "content": "new"}
    m.delete.return_value = None
    m.stats.return_value = {"total_docs": 1, "avg_tokens": 50}
    return m


# --- create_memory_tools tests ---


def test_create_memory_tools_default():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice")
    names = {t.name for t in tools}
    assert names == {
        "add_memory",
        "search_memory",
        "get_memory",
        "list_memories",
        "update_memory",
        "delete_memory",
        "generate_skills",
        "list_skills",
        "get_skill",
    }


def test_create_memory_tools_no_crud():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice", include_crud=False)
    names = {t.name for t in tools}
    assert names == {
        "add_memory",
        "search_memory",
        "generate_skills",
        "list_skills",
        "get_skill",
    }


def test_create_memory_tools_with_admin():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice", include_admin=True)
    names = {t.name for t in tools}
    assert "memory_stats" in names
    assert len(tools) == 10


# --- Tool invocation tests ---


def test_tool_add_memory():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice")
    add_tool = next(t for t in tools if t.name == "add_memory")

    result = add_tool.invoke({"content": "I love sushi"})

    mem.add.assert_called_once_with("I love sushi", store="test_store", user_id="alice")
    assert "Stored successfully" in result


def test_tool_search_memory():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice")
    search_tool = next(t for t in tools if t.name == "search_memory")

    result = search_tool.invoke({"query": "food", "limit": 3})

    mem.search.assert_called_once_with("food", store="test_store", user_id="alice", limit=3)
    parsed = json.loads(result)
    assert "query" in parsed


def test_tool_get_memory():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice")
    get_tool = next(t for t in tools if t.name == "get_memory")

    result = get_tool.invoke({"doc_id": "doc1"})

    mem.get.assert_called_once_with("doc1", store="test_store", user_id="alice")
    parsed = json.loads(result)
    assert parsed["id"] == "doc1"


def test_tool_get_memory_not_found():
    mem = _mock_memory()
    mem.get.return_value = None
    tools = create_memory_tools(mem, "test_store", "alice")
    get_tool = next(t for t in tools if t.name == "get_memory")

    result = get_tool.invoke({"doc_id": "nonexistent"})

    parsed = json.loads(result)
    assert "error" in parsed


def test_tool_list_memories():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice")
    list_tool = next(t for t in tools if t.name == "list_memories")

    result = list_tool.invoke({})

    mem.list.assert_called_once_with(store="test_store", user_id="alice")
    parsed = json.loads(result)
    assert isinstance(parsed, list)


def test_tool_update_memory():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice")
    update_tool = next(t for t in tools if t.name == "update_memory")

    result = update_tool.invoke(
        {"doc_id": "doc1", "content": "new content", "summary": "new summary"}
    )

    mem.update.assert_called_once_with("doc1", "new content", "new summary", store="test_store", user_id="alice")
    parsed = json.loads(result)
    assert parsed["id"] == "doc1"


def test_tool_update_memory_not_found():
    mem = _mock_memory()
    mem.update.side_effect = ValueError("Document not found")
    tools = create_memory_tools(mem, "test_store", "alice")
    update_tool = next(t for t in tools if t.name == "update_memory")

    result = update_tool.invoke(
        {"doc_id": "bad", "content": "x", "summary": "y"}
    )

    parsed = json.loads(result)
    assert "error" in parsed


def test_tool_delete_memory():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice")
    delete_tool = next(t for t in tools if t.name == "delete_memory")

    result = delete_tool.invoke({"doc_id": "doc1"})

    mem.delete.assert_called_once_with("doc1", store="test_store", user_id="alice")
    assert "deleted" in result


def test_tool_delete_memory_not_found():
    mem = _mock_memory()
    mem.delete.side_effect = ValueError("Document not found")
    tools = create_memory_tools(mem, "test_store", "alice")
    delete_tool = next(t for t in tools if t.name == "delete_memory")

    result = delete_tool.invoke({"doc_id": "bad"})

    parsed = json.loads(result)
    assert "error" in parsed


def test_tool_memory_stats():
    mem = _mock_memory()
    tools = create_memory_tools(mem, "test_store", "alice", include_admin=True)
    stats_tool = next(t for t in tools if t.name == "memory_stats")

    result = stats_tool.invoke({})

    mem.stats.assert_called_once_with(store="test_store", user_id="alice")
    parsed = json.loads(result)
    assert parsed["total_docs"] == 1


# --- create_memory_agent tests ---


def test_create_memory_agent_returns_graph():
    agent = create_memory_agent(
        memory=_mock_memory(),
        store="test_store",
        user_id="test_user",
        model="openai:gpt-5-mini",
    )
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")


def test_create_memory_agent_with_extra_tools():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    agent = create_memory_agent(
        memory=_mock_memory(),
        store="test_store",
        user_id="test_user",
        model="openai:gpt-5-mini",
        extra_tools=[greet],
    )
    assert hasattr(agent, "invoke")


def test_create_memory_agent_auto_creates_memory(tmp_path):
    agent = create_memory_agent(
        model="openai:gpt-5-mini",
        store="test_store",
        user_id="auto_user",
        api_key="test-key",
        data_root=str(tmp_path),
    )
    assert hasattr(agent, "invoke")
