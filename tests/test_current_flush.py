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


from infini_memory.manager import MemoryManager  # noqa: E402


def test_flush_current_docs_to_raw_writes_and_deletes(tmp_path: Path) -> None:
    mm = MemoryManager(
        root=tmp_path,
        data_root="data",
        doc_dir="doc",
        meta_dir="metadata",
        index_file="index.json",
        user_id="u1",
    )

    # Create doc files.
    mm.doc_dir.mkdir(parents=True, exist_ok=True)
    (mm.doc_dir / "CURRENT.md").write_text("<seq=1> hi", encoding="utf-8")
    (mm.doc_dir / "CURRENT_THREAD_0.md").write_text("<seq=2> hello", encoding="utf-8")
    (mm.doc_dir / "OTHER.md").write_text("keep", encoding="utf-8")

    # Create metadata index with CURRENT docs.
    mm.meta_dir.mkdir(parents=True, exist_ok=True)
    mm.index_path.write_text(
        '{"docs": [{"id": "CURRENT", "path": "data/u1/doc/CURRENT.md", "created_at": "t", "updated_at": "t", "tokens": 1, "summary": ""},'
        ' {"id": "CURRENT_THREAD_0", "path": "data/u1/doc/CURRENT_THREAD_0.md", "created_at": "t", "updated_at": "t", "tokens": 1, "summary": ""},'
        ' {"id": "OTHER", "path": "data/u1/doc/OTHER.md", "created_at": "t", "updated_at": "t", "tokens": 1, "summary": ""}],'
        ' "common": {"total_docs": 3, "by_update_count": {}, "by_current_epoch": {}, "avg_tokens": 0}}',
        encoding="utf-8",
    )

    saved = mm.flush_current_docs_to_raw()
    assert len(saved) == 2
    assert (mm.doc_dir / "CURRENT.md").exists() is False
    assert (mm.doc_dir / "CURRENT_THREAD_0.md").exists() is False
    assert (mm.doc_dir / "OTHER.md").exists() is True

    raw_files = sorted(p.name for p in mm.raw_dir.glob("CURRENT_*.md"))
    assert raw_files == ["CURRENT_1.md", "CURRENT_2.md"]

    idx = mm._load_index()
    remaining = {d["id"] for d in idx.get("docs", [])}
    assert remaining == {"OTHER"}

