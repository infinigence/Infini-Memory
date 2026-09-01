"""Tests for LocalStorage backend."""

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

from infini_memory_classic.storage import LocalStorage  # noqa: E402


def test_read_write_roundtrip(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write_text("a/b/test.txt", "hello world")
    assert storage.read_text("a/b/test.txt") == "hello world"


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write_text("deep/nested/dir/file.md", "content")
    assert (tmp_path / "deep/nested/dir/file.md").exists()


def test_delete(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write_text("to_delete.txt", "bye")
    assert storage.exists("to_delete.txt")
    storage.delete("to_delete.txt")
    assert not storage.exists("to_delete.txt")


def test_delete_missing_ok(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.delete("nonexistent.txt", missing_ok=True)


def test_exists_nonexistent(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    assert not storage.exists("no_such_file.txt")


def test_exists_directory(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.mkdir("mydir")
    assert storage.exists("mydir")


def test_mkdir_nested(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.mkdir("a/b/c")
    assert (tmp_path / "a/b/c").is_dir()


def test_glob_pattern(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write_text("docs/file1.md", "one")
    storage.write_text("docs/file2.md", "two")
    storage.write_text("docs/file3.txt", "three")

    md_files = sorted(storage.glob("docs", "*.md"))
    assert len(md_files) == 2
    assert all(f.endswith(".md") for f in md_files)


def test_glob_empty_dir(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.mkdir("empty")
    assert storage.glob("empty", "*.md") == []


def test_glob_nonexistent_dir(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    assert storage.glob("no_such_dir", "*.md") == []


def test_rmtree(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write_text("tree/a.txt", "a")
    storage.write_text("tree/sub/b.txt", "b")
    assert storage.exists("tree/a.txt")

    storage.rmtree("tree")
    assert not storage.exists("tree")
    assert not storage.exists("tree/a.txt")


def test_rmtree_nonexistent(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.rmtree("nonexistent")


def test_listdir(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write_text("parent/file.txt", "content")
    storage.mkdir("parent/subdir")

    entries = storage.listdir("parent")
    assert "file.txt" in entries
    assert "subdir" in entries


def test_listdir_empty(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    assert storage.listdir("nonexistent") == []


def test_is_dir(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.write_text("file.txt", "content")
    storage.mkdir("dir")

    assert not storage.is_dir("file.txt")
    assert storage.is_dir("dir")
    assert not storage.is_dir("nonexistent")


def test_absolute_path_support(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    abs_path = str(tmp_path / "abs_test.txt")
    storage.write_text(abs_path, "absolute")
    assert storage.read_text(abs_path) == "absolute"
