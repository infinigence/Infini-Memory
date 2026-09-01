"""Tests for memory store management APIs."""

import sys
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _project_root()
sys.path.insert(0, str(ROOT / "src"))

from infini_memory_classic import Memory  # noqa: E402
from infini_memory_classic.manager import MemoryManager, count_tokens  # noqa: E402


def _make_memory(tmp_path: Path) -> Memory:
    return Memory(
        api_key="test-key",
        data_root=str(tmp_path),
        root=str(tmp_path.parent),
    )


def _seed_doc(tmp_path: Path, store: str, user_id: str, content: str, summary: str) -> str:
    mm = MemoryManager(
        root=tmp_path.parent,
        data_root=tmp_path.name,
        doc_dir="doc",
        meta_dir="metadata",
        index_file="index.json",
        store=store,
        user_id=user_id,
    )
    meta = mm.add_doc(content, summary, count_tokens(content))
    return meta.id


def test_list_stores_empty(tmp_path):
    m = _make_memory(tmp_path / "data")
    assert m.list_stores() == []


def test_create_and_list_stores(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    m.create_store("store_a")
    m.create_store("store_b")

    stores = m.list_stores()
    assert stores == ["store_a", "store_b"]


def test_get_store_empty(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)
    m.create_store("my_store")

    info = m.get_store("my_store")
    assert info["name"] == "my_store"
    assert info["user_count"] == 0
    assert info["users"] == []


def test_get_store_with_users(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    _seed_doc(data_root, "my_store", "alice", "content", "summary")
    _seed_doc(data_root, "my_store", "bob", "content", "summary")

    info = m.get_store("my_store")
    assert info["name"] == "my_store"
    assert info["user_count"] == 2
    assert set(info["users"]) == {"alice", "bob"}


def test_delete_store(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    _seed_doc(data_root, "to_delete", "alice", "content", "summary")
    assert "to_delete" in m.list_stores()

    m.delete_store("to_delete")
    assert "to_delete" not in m.list_stores()


def test_store_isolation(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    doc_id_a = _seed_doc(data_root, "store_a", "alice", "content A", "summary A")
    doc_id_b = _seed_doc(data_root, "store_b", "alice", "content B", "summary B")

    doc_a = m.get(doc_id_a, store="store_a", user_id="alice")
    assert doc_a is not None
    assert doc_a["content"] == "content A"

    doc_b = m.get(doc_id_b, store="store_b", user_id="alice")
    assert doc_b is not None
    assert doc_b["content"] == "content B"

    # Cross-store access should fail
    assert m.get(doc_id_a, store="store_b", user_id="alice") is None
    assert m.get(doc_id_b, store="store_a", user_id="alice") is None


def test_list_users_scoped_to_store(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    _seed_doc(data_root, "store_a", "alice", "content", "summary")
    _seed_doc(data_root, "store_a", "bob", "content", "summary")
    _seed_doc(data_root, "store_b", "charlie", "content", "summary")

    assert m.list_users(store="store_a") == ["alice", "bob"]
    assert m.list_users(store="store_b") == ["charlie"]


def test_validate_store_name(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    import pytest
    with pytest.raises(ValueError):
        m.create_store("")
    with pytest.raises(ValueError):
        m.create_store("../escape")
    with pytest.raises(ValueError):
        m.create_store("a/b")


def test_create_store_idempotent(tmp_path):
    data_root = tmp_path / "data"
    m = _make_memory(data_root)

    m.create_store("my_store")
    m.create_store("my_store")  # Should not raise
    assert m.list_stores() == ["my_store"]
