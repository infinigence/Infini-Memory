from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .storage import StorageBackend


@dataclass
class DocMeta:
    id: str
    path: str
    created_at: str
    updated_at: str
    tokens: int
    summary: str
    update_count: int = 0
    parent_id: Optional[str] = None
    current_epoch: int = 0
    merged_from: Optional[List[str]] = None
    history: Optional[List[Dict]] = None


def _validate_store_name(store: str) -> None:
    """Validate store name to prevent path traversal."""
    if not store:
        raise ValueError("store name must not be empty")
    if ".." in store or "/" in store or "\\" in store:
        raise ValueError(f"Invalid store name: {store!r}")


class MemoryManager:
    """Manage data/STORE_<store>/USER_<user_id>/doc and metadata."""

    STORE_PREFIX = "STORE_"
    USER_PREFIX = "USER_"

    def __init__(self, root: Path, data_root: str, doc_dir: str, meta_dir: str, index_file: str, store: str, user_id: str, storage: Optional["StorageBackend"] = None):
        _validate_store_name(store)
        self.logger = logging.getLogger("infini_memory_classic")
        self.root = Path(root)
        self.store = store
        self.user_id = user_id
        self.data_dir = self.root / data_root / f"{self.STORE_PREFIX}{store}" / f"{self.USER_PREFIX}{user_id}"
        self.doc_dir = self.data_dir / doc_dir
        self.meta_dir = self.data_dir / meta_dir
        self.raw_dir = self.data_dir / "raw"
        self.index_path = self.meta_dir / index_file
        self.event_path = self.meta_dir / "event.json"
        self._index_lock = threading.Lock()
        self._event_lock = threading.Lock()

        if storage is not None:
            self.storage = storage
        else:
            from .storage import LocalStorage
            self.storage = LocalStorage(self.root)

    def _rel(self, abs_path: Path) -> str:
        """Convert an absolute Path to a relative string for storage backend."""
        return str(abs_path.relative_to(self.root))

    def read_file(self, rel_path: str) -> str:
        """Read a file by its relative path (as stored in index)."""
        return self.storage.read_text(rel_path)

    def write_file(self, rel_path: str, content: str) -> None:
        """Write a file by its relative path."""
        self.storage.write_text(rel_path, content)

    def delete_file(self, rel_path: str, missing_ok: bool = True) -> None:
        """Delete a file by its relative path."""
        self.storage.delete(rel_path, missing_ok=missing_ok)

    def file_exists(self, rel_path: str) -> bool:
        """Check if a file exists by its relative path."""
        return self.storage.exists(rel_path)

    def glob_files(self, dir_rel_path: str, pattern: str) -> List[str]:
        """Glob for files in a directory. Returns relative paths."""
        return self.storage.glob(dir_rel_path, pattern)

    def _ensure_dirs(self) -> None:
        """Ensure directories and the index file exist (called only when a write is needed)."""
        self.storage.mkdir(self._rel(self.doc_dir))
        self.storage.mkdir(self._rel(self.meta_dir))

        if not self.storage.exists(self._rel(self.index_path)):
            self.storage.write_text(
                self._rel(self.index_path),
                json.dumps(
                    {"docs": [], "common": {"total_docs": 0, "by_update_count": {}, "by_current_epoch": {}, "avg_tokens": 0, "next_seq": 1}},
                    ensure_ascii=False,
                    indent=2,
                ),
            )

    def _load_events(self) -> List:
        """Load event history."""
        try:
            return json.loads(self.storage.read_text(self._rel(self.event_path)))
        except Exception:
            return []

    def _save_events(self, events: List) -> None:
        """Thread-safe save of event history."""
        with self._event_lock:
            self.storage.write_text(self._rel(self.event_path), json.dumps(events, ensure_ascii=False, indent=2))

    def _load_index(self) -> Dict:
        try:
            return json.loads(self.storage.read_text(self._rel(self.index_path)))
        except Exception:
            return {"docs": []}

    def _save_index(self, idx: Dict) -> None:
        self._ensure_dirs()
        self.storage.write_text(self._rel(self.index_path), json.dumps(idx, ensure_ascii=False, indent=2))

    def _load_index_locked(self) -> Dict:
        """Thread-safe index load (holding lock)."""
        with self._index_lock:
            return self._load_index()

    def _save_index_locked(self, idx: Dict) -> None:
        """Thread-safe index save (holding lock)."""
        with self._index_lock:
            self._save_index(idx)

    def _update_common_stats(self, idx: Dict) -> None:
        """Update the common field statistics."""
        docs = idx.get("docs", [])
        total_docs = len(docs)
        by_update_count: Dict[str, int] = {}
        by_current_epoch: Dict[str, int] = {}
        total_tokens = 0

        for doc in docs:
            uc = str(doc.get("update_count", 0))
            by_update_count[uc] = by_update_count.get(uc, 0) + 1

            ce = str(doc.get("current_epoch", 0))
            by_current_epoch[ce] = by_current_epoch.get(ce, 0) + 1

            total_tokens += doc.get("tokens", 0)

        avg_tokens = total_tokens / total_docs if total_docs > 0 else 0

        idx.setdefault("common", {})["total_docs"] = total_docs
        idx.setdefault("common", {})["by_update_count"] = by_update_count
        idx.setdefault("common", {})["by_current_epoch"] = by_current_epoch
        idx.setdefault("common", {})["avg_tokens"] = avg_tokens

    def log_event(self, event: Dict) -> None:
        """Log an operation to event.json."""
        events = self._load_events()
        events.append(event)
        self._save_events(events)

    def get_common_stats(self) -> Dict:
        """Get common statistics."""
        idx = self._load_index()
        return idx.get("common", {"total_docs": 0, "by_update_count": {}, "by_current_epoch": {}, "avg_tokens": 0, "next_seq": 1, "split_count": 0})

    def get_split_count(self) -> int:
        """Get the current split count."""
        idx = self._load_index()
        return idx.get("common", {}).get("split_count", 0)

    def increment_split_count(self) -> int:
        """Increment the split count and return the new value."""
        self._ensure_dirs()
        idx = self._load_index()
        current = idx.get("common", {}).get("split_count", 0)
        idx.setdefault("common", {})["split_count"] = current + 1
        self._save_index(idx)
        return current + 1

    def get_next_seq(self) -> int:
        """Get the next available sequence number."""
        idx = self._load_index()
        return idx.get("common", {}).get("next_seq", 1)

    def increment_seq(self) -> int:
        """Increment the sequence number and return the value before incrementing."""
        self._ensure_dirs()
        idx = self._load_index()
        current = idx.get("common", {}).get("next_seq", 1)
        idx.setdefault("common", {})["next_seq"] = current + 1
        self._save_index(idx)
        return current

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, supporting both absolute and relative paths.

        For LocalStorage, relative paths are resolved against root to get
        absolute paths. For non-local backends (e.g. S3), paths are kept
        as-is since the storage backend handles resolution.

        Args:
            path: Document path (may be absolute or relative to root)

        Returns:
            Resolved path
        """
        p = Path(path)
        if p.is_absolute():
            return p
        from .storage import LocalStorage
        if isinstance(self.storage, LocalStorage):
            return self.root / p
        return p

    def list_docs(self) -> List[DocMeta]:
        idx = self._load_index()
        result: List[DocMeta] = []
        for d in idx.get("docs", []):
            meta_dict = d.copy()
            if "path" in meta_dict:
                meta_dict["path"] = str(self._resolve_path(meta_dict["path"]))
            result.append(DocMeta(**meta_dict))
        return result

    def get_doc_by_id(self, doc_id: str) -> Optional[DocMeta]:
        """Get document metadata by id.

        Supports exact matching and fuzzy matching (via UUID suffix).
        If doc_id contains only the UUID portion (36 characters), attempts to match the suffix of the full ID.
        """
        idx = self._load_index()

        for d in idx.get("docs", []):
            if d.get("id") == doc_id:
                meta_dict = d.copy()
                if "path" in meta_dict:
                    meta_dict["path"] = str(self._resolve_path(meta_dict["path"]))
                return DocMeta(**meta_dict)

        if len(doc_id) == 36:
            for d in idx.get("docs", []):
                existing_id = d.get("id", "")
                if existing_id.endswith("_" + doc_id) or existing_id == doc_id:
                    self.logger.debug(
                        "[get_doc_by_id] Fuzzy match succeeded: input=%s, matched=%s", doc_id, existing_id
                    )
                    meta_dict = d.copy()
                    if "path" in meta_dict:
                        meta_dict["path"] = str(self._resolve_path(meta_dict["path"]))
                    return DocMeta(**meta_dict)

        return None

    def clear_doc(self, doc_id: str) -> DocMeta:
        """Clear the content of the specified document (keep metadata, set content to empty)."""
        self._ensure_dirs()
        with self._index_lock:
            idx = self._load_index()
            target = None
            for d in idx.get("docs", []):
                if d.get("id") == doc_id:
                    target = d
                    break
            if target is None:
                raise ValueError(f"Document does not exist: {doc_id}")

            self.storage.write_text(target["path"], "")

            target["updated_at"] = json_datetime_now()
            target["tokens"] = 0
            target["summary"] = ""
            self._update_common_stats(idx)
            self._save_index(idx)
            self.logger.info("Cleared document: %s", doc_id)
            return DocMeta(**target)

    def add_doc(self, content: str, summary: str, tokens: int) -> DocMeta:
        self._ensure_dirs()
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        doc_id = f"{ts}_{uuid.uuid4()}"
        filename = f"{doc_id}.md"
        rel_path = self._rel(self.doc_dir / filename)
        self.storage.write_text(rel_path, content)
        now = json_datetime_now()
        meta = DocMeta(
            id=doc_id,
            path=rel_path,
            created_at=now,
            updated_at=now,
            tokens=tokens,
            summary=summary,
        )
        self.add_doc_to_index(meta)
        self.logger.info("Added document: %s", filename)
        return meta

    def update_doc(self, doc_id: str, new_content: str, summary: str, tokens: int) -> DocMeta:
        self._ensure_dirs()
        with self._index_lock:
            idx = self._load_index()
            target = None
            for d in idx.get("docs", []):
                if d.get("id") == doc_id:
                    target = d
                    break
            if target is None:
                raise ValueError(f"Document does not exist: {doc_id}")
            self.storage.write_text(target["path"], new_content)
            target["updated_at"] = json_datetime_now()
            try:
                target["update_count"] = int(target.get("update_count", 0)) + 1
            except Exception:
                target["update_count"] = 1
            target["tokens"] = tokens
            target["summary"] = summary
            self._update_common_stats(idx)
            self._save_index(idx)
            self.logger.info("Updated document: %s", doc_id)
            return DocMeta(**target)

    def split_doc(self, doc_id: str, parts: List[Dict[str, str]]) -> List[DocMeta]:
        with self._index_lock:
            self._ensure_dirs()
            idx = self._load_index()
            target = None
            target_index = None
            for i, d in enumerate(idx.get("docs", [])):
                if d.get("id") == doc_id:
                    target = d
                    target_index = i
                    break
            if target is None:
                raise ValueError(f"Document does not exist: {doc_id}")
            old_rel_path = target["path"]
            metas: List[DocMeta] = []
            for p in parts:
                meta_id = f"{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
                filename = f"{meta_id}.md"
                rel_path = self._rel(self.doc_dir / filename)
                self.storage.write_text(rel_path, p.get("content", ""))
                now = json_datetime_now()
                meta = DocMeta(
                    id=meta_id,
                    path=rel_path,
                    created_at=now,
                    updated_at=now,
                    tokens=count_tokens(p.get("content", "")),
                    summary=p.get("title", ""),
                    parent_id=target["id"],
                )
                idx.setdefault("docs", []).append(meta.__dict__)
                metas.append(meta)
            try:
                if isinstance(target_index, int):
                    del idx["docs"][target_index]
                else:
                    idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") != doc_id]
            except Exception:
                idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") != doc_id]
            self._update_common_stats(idx)
            self._save_index(idx)
        try:
            self.storage.delete(old_rel_path)
        except Exception:
            ...
        self.logger.info("Split document: %s -> %d parts (old document deleted)", doc_id, len(parts))
        return metas

    def _ensure_raw_dir(self) -> None:
        """Ensure the raw directory exists."""
        self.storage.mkdir(self._rel(self.raw_dir))

    def save_to_raw(self, content: str, current_number: int = None) -> str:
        """Save content to a raw/CURRENT_<number>.md file.

        Args:
            content: The content to save
            current_number: The CURRENT document number

        Returns:
            Relative path of the saved file
        """
        import re

        self._ensure_raw_dir()

        if current_number is None:
            seq_matches = re.findall(r'<seq=(\d+)[,\s>]', content)
            if seq_matches:
                current_number = max(int(seq) for seq in seq_matches)
                self.logger.debug("Extracted max seq value from content: %d", current_number)
            else:
                current_number = 0
                self.logger.warning("No seq value found in content, using default value 0")

        filename = f"CURRENT_{current_number}.md"
        rel_path = self._rel(self.raw_dir / filename)
        self.storage.write_text(rel_path, content)
        self.logger.info("Saved to raw directory: %s", filename)
        return rel_path

    def flush_current_docs_to_raw(self) -> List[str]:
        """Write all CURRENT* documents from the doc directory to raw, and delete the corresponding doc files and index metadata.

        Returns:
            List of relative file paths written to raw.
        """
        self._ensure_dirs()

        doc_dir_rel = self._rel(self.doc_dir)
        all_md_files = self.storage.glob(doc_dir_rel, "*.md")
        current_files = [f for f in all_md_files if Path(f).name.startswith("CURRENT")]
        if not current_files:
            return []

        saved_paths: List[str] = []
        for rel_path in current_files:
            content = self.storage.read_text(rel_path)
            saved_paths.append(self.save_to_raw(content))

        for rel_path in current_files:
            self.storage.delete(rel_path)

        docs_to_remove = {"CURRENT"}
        idx = self._load_index()
        for d in idx.get("docs", []):
            doc_id = d.get("id", "")
            if isinstance(doc_id, str) and doc_id.startswith("CURRENT_THREAD_"):
                docs_to_remove.add(doc_id)
        idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") not in docs_to_remove]
        self._update_common_stats(idx)
        self._save_index(idx)

        return saved_paths

    def list_raw_files(self) -> List[str]:
        """List all raw/CURRENT_<number>.md files, sorted by number. Returns relative paths."""
        self._ensure_raw_dir()
        raw_dir_rel = self._rel(self.raw_dir)
        files = self.storage.glob(raw_dir_rel, "CURRENT_*.md")
        files.sort(key=lambda p: int(Path(p).stem.split("_")[1]))
        return files

    def load_raw_content(self, current_number: int) -> str:
        """Load the content of a raw/CURRENT_<number>.md file."""
        filename = f"CURRENT_{current_number}.md"
        rel_path = self._rel(self.raw_dir / filename)
        if not self.storage.exists(rel_path):
            raise FileNotFoundError(f"Raw file does not exist: {filename}")
        return self.storage.read_text(rel_path)

    def add_doc_to_index(self, meta: DocMeta) -> None:
        """Thread-safe addition of a document to the index (for multi-threaded environments)."""
        with self._index_lock:
            idx = self._load_index()
            idx.setdefault("docs", []).append(meta.__dict__)
            self._update_common_stats(idx)
            self._save_index(idx)

    def update_doc_in_index(self, doc_id: str, updater: callable) -> None:
        """Thread-safe update of a document in the index (for multi-threaded environments)."""
        with self._index_lock:
            idx = self._load_index()
            for d in idx.get("docs", []):
                if d.get("id") == doc_id:
                    updater(d)
                    break
            self._update_common_stats(idx)
            self._save_index(idx)

    def remove_docs_from_index(self, doc_ids: set) -> None:
        """Thread-safe removal of multiple documents from the index (for multi-threaded environments)."""
        with self._index_lock:
            idx = self._load_index()
            idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") not in doc_ids]
            self._update_common_stats(idx)
            self._save_index(idx)

    def delete_doc(self, doc_id: str) -> None:
        """Completely delete the specified document (including file and metadata)."""
        self._ensure_dirs()
        with self._index_lock:
            idx = self._load_index()

            target = None
            for d in idx.get("docs", []):
                if d.get("id") == doc_id:
                    target = d
                    break

            if target is None:
                raise ValueError(f"Document does not exist: {doc_id}")

            rel_path = target["path"]
            if self.storage.exists(rel_path):
                self.storage.delete(rel_path)
                self.logger.info("Deleted document file: %s", rel_path)

            idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") != doc_id]
            self._update_common_stats(idx)
            self._save_index(idx)
            self.logger.info("Deleted document: %s", doc_id)

    def get_doc_content(self, doc_id: str) -> Optional[str]:
        """Read document file content by id. Returns None if not found."""
        meta = self.get_doc_by_id(doc_id)
        if meta is None:
            return None
        idx = self._load_index()
        rel_path = None
        for d in idx.get("docs", []):
            if d.get("id") == doc_id:
                rel_path = d.get("path")
                break
        if rel_path is None:
            return None
        if not self.storage.exists(rel_path):
            return None
        return self.storage.read_text(rel_path)

    @staticmethod
    def list_stores(root: Path, data_root: str, storage: Optional["StorageBackend"] = None) -> List[str]:
        """List all store directories under data_root."""
        pfx = MemoryManager.STORE_PREFIX
        if storage is not None:
            rel = str(Path(data_root))
            if not storage.exists(rel):
                return []
            entries = storage.listdir(rel)
            return sorted(
                e[len(pfx):] for e in entries
                if e.startswith(pfx) and storage.is_dir(f"{rel}/{e}")
            )
        base = Path(root) / data_root
        if not base.exists():
            return []
        return sorted(
            d.name[len(pfx):] for d in base.iterdir()
            if d.is_dir() and d.name.startswith(pfx)
        )

    @staticmethod
    def create_store(root: Path, data_root: str, store: str, storage: Optional["StorageBackend"] = None) -> None:
        """Create a store directory."""
        _validate_store_name(store)
        dir_name = f"{MemoryManager.STORE_PREFIX}{store}"
        if storage is not None:
            rel = str(Path(data_root) / dir_name)
            storage.mkdir(rel)
            return
        store_dir = Path(root) / data_root / dir_name
        store_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def delete_store(root: Path, data_root: str, store: str, storage: Optional["StorageBackend"] = None) -> None:
        """Remove entire store directory and all its user data."""
        _validate_store_name(store)
        dir_name = f"{MemoryManager.STORE_PREFIX}{store}"
        if storage is not None:
            rel = str(Path(data_root) / dir_name)
            storage.rmtree(rel)
            return
        import shutil
        store_dir = Path(root) / data_root / dir_name
        if store_dir.exists():
            shutil.rmtree(store_dir)

    @staticmethod
    def get_store_info(root: Path, data_root: str, store: str, storage: Optional["StorageBackend"] = None) -> Dict:
        """Return info about a store: name, user_count, users."""
        _validate_store_name(store)
        users = MemoryManager.list_users(root, data_root, store, storage=storage)
        return {"name": store, "user_count": len(users), "users": users}

    @staticmethod
    def list_users(root: Path, data_root: str, store: str, storage: Optional["StorageBackend"] = None) -> List[str]:
        """List all user_id directories under data_root/STORE_<store>/."""
        _validate_store_name(store)
        store_dir_name = f"{MemoryManager.STORE_PREFIX}{store}"
        user_pfx = MemoryManager.USER_PREFIX
        if storage is not None:
            rel = str(Path(data_root) / store_dir_name)
            if not storage.exists(rel):
                return []
            entries = storage.listdir(rel)
            return sorted(
                e[len(user_pfx):] for e in entries
                if e.startswith(user_pfx) and storage.is_dir(f"{rel}/{e}")
            )
        base = Path(root) / data_root / store_dir_name
        if not base.exists():
            return []
        return sorted(
            d.name[len(user_pfx):] for d in base.iterdir()
            if d.is_dir() and d.name.startswith(user_pfx)
        )

    @staticmethod
    def delete_user_data(root: Path, data_root: str, store: str, user_id: str, storage: Optional["StorageBackend"] = None) -> None:
        """Remove entire data directory for a user within a store."""
        _validate_store_name(store)
        store_dir_name = f"{MemoryManager.STORE_PREFIX}{store}"
        user_dir_name = f"{MemoryManager.USER_PREFIX}{user_id}"
        if storage is not None:
            rel = str(Path(data_root) / store_dir_name / user_dir_name)
            storage.rmtree(rel)
            return
        import shutil
        user_dir = Path(root) / data_root / store_dir_name / user_dir_name
        if user_dir.exists():
            shutil.rmtree(user_dir)


def count_tokens(text: str) -> int:
    """Approximate token count for mixed Chinese/English text."""
    import re

    chinese_chars = len(re.findall(r'[一-鿿　-〿＀-￯]', text))
    english_words = len(re.findall(r'[a-zA-Z0-9]+', text))

    chinese_tokens = int(chinese_chars / 1.5)
    total_tokens = max(1, chinese_tokens + english_words)
    return total_tokens


def json_datetime_now() -> str:
    """Return Beijing time (Asia/Shanghai) as an ISO8601 string."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


__all__ = ["MemoryManager", "DocMeta", "count_tokens", "json_datetime_now"]
