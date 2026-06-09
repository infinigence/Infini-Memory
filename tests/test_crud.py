"""Tests for document CRUD and user management APIs."""

import sys
import tempfile
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _project_root()
sys.path.insert(0, str(ROOT / "src"))

from infini_memory import Memory  # noqa: E402
from infini_memory.manager import MemoryManager, count_tokens  # noqa: E402


def _make_memory(tmp_path: Path) -> Memory:
    return Memory(
        api_key="test-key",
        data_root=str(tmp_path),
        root=str(tmp_path.parent),
    )


def _seed_doc(tmp_path: Path, user_id: str, content: str, summary: str) -> str:
    """Add a document directly via MemoryManager, return doc_id."""
    mm = MemoryManager(
        root=tmp_path.parent,
        data_root=tmp_path.name,
        doc_dir="doc",
        meta_dir="metadata",
        index_file="index.json",
        user_id=user_id,
    )
    meta = mm.add_doc(content, summary, count_tokens(content))
    return meta.id


def test_list_empty(tmp_path):
    m = _make_memory(tmp_path / "data")
    assert m.list(user_id="no_such_user") == []


def test_list_after_seed(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    doc_id = _seed_doc(data_root, "alice", "# Notes\nAlice likes cats.", "Alice's pet preferences")

    docs = m.list(user_id="alice")
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id
    assert docs[0]["summary"] == "Alice's pet preferences"
    assert "content" not in docs[0]


def test_get_existing(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    content = "# Travel\nGoing to Tokyo in spring."
    doc_id = _seed_doc(data_root, "bob", content, "Bob's travel plans")

    doc = m.get(doc_id, user_id="bob")
    assert doc is not None
    assert doc["id"] == doc_id
    assert doc["content"] == content
    assert doc["summary"] == "Bob's travel plans"


def test_get_nonexistent(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    _seed_doc(data_root, "bob", "some content", "summary")

    assert m.get("nonexistent-id", user_id="bob") is None


def test_get_wrong_user(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    doc_id = _seed_doc(data_root, "alice", "content", "summary")

    assert m.get(doc_id, user_id="bob") is None


def test_update(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    doc_id = _seed_doc(data_root, "alice", "old content", "old summary")

    updated = m.update(doc_id, "new content", "new summary", user_id="alice")
    assert updated["id"] == doc_id
    assert updated["content"] == "new content"
    assert updated["summary"] == "new summary"
    assert updated["update_count"] == 1

    fetched = m.get(doc_id, user_id="alice")
    assert fetched["content"] == "new content"


def test_update_nonexistent(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    _seed_doc(data_root, "alice", "content", "summary")

    try:
        m.update("nonexistent-id", "content", "summary", user_id="alice")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_delete(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    doc_id = _seed_doc(data_root, "alice", "content to delete", "summary")

    assert len(m.list(user_id="alice")) == 1
    m.delete(doc_id, user_id="alice")
    assert len(m.list(user_id="alice")) == 0
    assert m.get(doc_id, user_id="alice") is None


def test_delete_nonexistent(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    _seed_doc(data_root, "alice", "content", "summary")

    try:
        m.delete("nonexistent-id", user_id="alice")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_list_users(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    assert m.list_users() == []

    _seed_doc(data_root, "alice", "content", "summary")
    _seed_doc(data_root, "bob", "content", "summary")

    users = m.list_users()
    assert users == ["alice", "bob"]


def test_delete_user(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    _seed_doc(data_root, "alice", "content", "summary")
    _seed_doc(data_root, "bob", "content", "summary")
    assert "alice" in m.list_users()

    m.delete_user("alice")
    assert m.list_users() == ["bob"]
    assert m.list(user_id="alice") == []


def test_delete_user_nonexistent(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    m.delete_user("nonexistent")


def test_get_all(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    assert m.get_all(user_id="alice") == []

    _seed_doc(data_root, "alice", "# Doc1\nContent one.", "First doc")
    _seed_doc(data_root, "alice", "# Doc2\nContent two.", "Second doc")

    docs = m.get_all(user_id="alice")
    assert len(docs) == 2
    assert all("content" in d for d in docs)
    summaries = {d["summary"] for d in docs}
    assert summaries == {"First doc", "Second doc"}


def test_count(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    assert m.count(user_id="alice") == 0

    _seed_doc(data_root, "alice", "content1", "summary1")
    _seed_doc(data_root, "alice", "content2", "summary2")
    assert m.count(user_id="alice") == 2


def test_stats(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    _seed_doc(data_root, "alice", "hello world", "greeting")
    _seed_doc(data_root, "alice", "another doc", "another")

    s = m.stats(user_id="alice")
    assert s["total_docs"] == 2
    assert "avg_tokens" in s
    assert "by_update_count" in s


def test_history_empty(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    assert m.history(user_id="no_such_user") == []


def test_history_after_operations(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    mm = MemoryManager(
        root=data_root.parent,
        data_root=data_root.name,
        doc_dir="doc",
        meta_dir="metadata",
        index_file="index.json",
        user_id="alice",
    )
    mm._ensure_dirs()
    mm.log_event({"event": "test_event", "detail": "hello"})

    events = m.history(user_id="alice")
    assert len(events) == 1
    assert events[0]["event"] == "test_event"


def test_delete_all(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    _seed_doc(data_root, "alice", "content1", "summary1")
    _seed_doc(data_root, "alice", "content2", "summary2")
    _seed_doc(data_root, "alice", "content3", "summary3")
    assert m.count(user_id="alice") == 3

    deleted = m.delete_all(user_id="alice")
    assert deleted == 3
    assert m.count(user_id="alice") == 0
    assert m.list(user_id="alice") == []
    # User directory still exists
    assert "alice" in m.list_users()


def test_delete_all_empty(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    assert m.delete_all(user_id="no_such_user") == 0


def test_reset(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    _seed_doc(data_root, "alice", "content", "summary")
    _seed_doc(data_root, "bob", "content", "summary")
    _seed_doc(data_root, "charlie", "content", "summary")
    assert len(m.list_users()) == 3

    m.reset()
    assert m.list_users() == []


def test_reset_empty(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    m.reset()
    assert m.list_users() == []


def test_search_limit_parameter(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    # Disable memory so search returns early without LLM calls
    m._cfg.memory.enabled = False

    orig_limit = m._cfg.memory.search_limit
    result = m.search("test", user_id="no_user", limit=5)
    # Verify cfg is restored after search with limit
    assert m._cfg.memory.search_limit == orig_limit

    result = m.search("test", user_id="no_user")
    assert m._cfg.memory.search_limit == orig_limit
