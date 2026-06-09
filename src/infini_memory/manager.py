from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


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


class MemoryManager:
    """Manage data/<user_id>/doc and data/<user_id>/metadata."""

    def __init__(self, root: Path, data_root: str, doc_dir: str, meta_dir: str, index_file: str, user_id: str):
        self.logger = logging.getLogger("infini_memory")
        self.root = Path(root)
        self.user_id = user_id
        self.data_dir = self.root / data_root / user_id
        self.doc_dir = self.data_dir / doc_dir
        self.meta_dir = self.data_dir / meta_dir
        self.raw_dir = self.data_dir / "raw"
        self.index_path = self.meta_dir / index_file
        self.event_path = self.meta_dir / "event.json"  # Operation history file
        # Thread locks to protect read/write operations on the index and event files
        self._index_lock = threading.Lock()
        self._event_lock = threading.Lock()
        # Do not create directories and files during initialization; only create them when a write is needed

    def _ensure_dirs(self) -> None:
        """Ensure directories and the index file exist (called only when a write is needed)."""
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the index file
        if not self.index_path.exists():
            self.index_path.write_text(
                json.dumps(
                    {"docs": [], "common": {"total_docs": 0, "by_update_count": {}, "by_current_epoch": {}, "avg_tokens": 0, "next_seq": 1}},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _load_events(self) -> List:
        """Load event history."""
        try:
            return json.loads(self.event_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_events(self, events: List) -> None:
        """Thread-safe save of event history."""
        with self._event_lock:
            self.event_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index(self) -> Dict:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"docs": []}

    def _save_index(self, idx: Dict) -> None:
        self._ensure_dirs()
        self.index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

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
        # Backward compatible: if old index lacks the next_seq field, default to 1
        return idx.get("common", {}).get("next_seq", 1)

    def increment_seq(self) -> int:
        """Increment the sequence number and return the value before incrementing."""
        self._ensure_dirs()
        idx = self._load_index()
        # Backward compatible: if old index lacks the next_seq field, initialize to 1
        current = idx.get("common", {}).get("next_seq", 1)
        idx.setdefault("common", {})["next_seq"] = current + 1
        self._save_index(idx)
        return current  # Return the value before incrementing (i.e., the sequence number used this time)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, supporting both absolute and relative paths.

        Args:
            path: Document path (may be absolute or relative to root)

        Returns:
            Resolved absolute path
        """
        p = Path(path)
        if p.is_absolute():
            # Absolute path: use directly
            return p
        else:
            # Relative path: resolve relative to root
            return self.root / p

    def list_docs(self) -> List[DocMeta]:
        idx = self._load_index()
        result: List[DocMeta] = []
        for d in idx.get("docs", []):
            # Resolve path (supports both absolute and relative paths)
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

        # First try exact matching
        for d in idx.get("docs", []):
            if d.get("id") == doc_id:
                # Resolve path (supports both absolute and relative paths)
                meta_dict = d.copy()
                if "path" in meta_dict:
                    meta_dict["path"] = str(self._resolve_path(meta_dict["path"]))
                return DocMeta(**meta_dict)

        # If exact match fails and doc_id length is 36 (standard UUID), try suffix matching
        if len(doc_id) == 36:
            for d in idx.get("docs", []):
                existing_id = d.get("id", "")
                # Document ID format: <timestamp>_<uuid>, check if the suffix matches
                if existing_id.endswith("_" + doc_id) or existing_id == doc_id:
                    self.logger.debug(
                        "[get_doc_by_id] Fuzzy match succeeded: input=%s, matched=%s", doc_id, existing_id
                    )
                    # Resolve path (supports both absolute and relative paths)
                    meta_dict = d.copy()
                    if "path" in meta_dict:
                        meta_dict["path"] = str(self._resolve_path(meta_dict["path"]))
                    return DocMeta(**meta_dict)

        return None

    def clear_doc(self, doc_id: str) -> DocMeta:
        """Clear the content of the specified document (keep metadata, set content to empty)."""
        self._ensure_dirs()
        # Use thread lock to protect index updates
        with self._index_lock:
            idx = self._load_index()
            target = None
            for d in idx.get("docs", []):
                if d.get("id") == doc_id:
                    target = d
                    break
            if target is None:
                raise ValueError(f"Document does not exist: {doc_id}")

            # Clear file content (using path resolution to support both absolute and relative paths)
            path = self._resolve_path(target["path"])
            path.write_text("", encoding="utf-8")

            # Update metadata
            target["updated_at"] = json_datetime_now()
            target["tokens"] = 0
            target["summary"] = ""
            self._update_common_stats(idx)
            self._save_index(idx)
            self.logger.info("Cleared document: %s", doc_id)
            return DocMeta(**target)

    def add_doc(self, content: str, summary: str, tokens: int) -> DocMeta:
        self._ensure_dirs()
        # New document id format: <Beijing time>_uuid
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        doc_id = f"{ts}_{uuid.uuid4()}"
        filename = f"{doc_id}.md"
        path = self.doc_dir / filename
        path.write_text(content, encoding="utf-8")
        now = json_datetime_now()
        meta = DocMeta(
            id=doc_id,
            path=str(path.relative_to(self.root)),
            created_at=now,
            updated_at=now,
            tokens=tokens,
            summary=summary,
        )
        # Use thread-safe method to add to the index
        self.add_doc_to_index(meta)
        self.logger.info("Added document: %s", filename)
        return meta

    def update_doc(self, doc_id: str, new_content: str, summary: str, tokens: int) -> DocMeta:
        self._ensure_dirs()
        # Use thread lock to protect index updates
        with self._index_lock:
            idx = self._load_index()
            target = None
            for d in idx.get("docs", []):
                if d.get("id") == doc_id:
                    target = d
                    break
            if target is None:
                raise ValueError(f"Document does not exist: {doc_id}")
            # Use path resolution to support both absolute and relative paths
            path = self._resolve_path(target["path"])
            path.write_text(new_content, encoding="utf-8")
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
        # Use thread lock to protect the entire split operation
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
            # Record old document path for deletion (using path resolution to support both absolute and relative paths)
            old_path = self._resolve_path(target["path"])
            metas: List[DocMeta] = []
            for p in parts:
                meta_id = f"{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
                filename = f"{meta_id}.md"
                path = self.doc_dir / filename
                path.write_text(p.get("content", ""), encoding="utf-8")
                now = json_datetime_now()
                meta = DocMeta(
                    id=meta_id,
                    path=str(path.relative_to(self.root)),
                    created_at=now,
                    updated_at=now,
                    tokens=count_tokens(p.get("content", "")),
                    summary=p.get("title", ""),
                    parent_id=target["id"],
                )
                idx.setdefault("docs", []).append(meta.__dict__)
                metas.append(meta)
            # Remove old document from index and delete old file
            try:
                if isinstance(target_index, int):
                    del idx["docs"][target_index]
                else:
                    idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") != doc_id]
            except Exception:
                idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") != doc_id]
            # Update statistics and save index
            self._update_common_stats(idx)
            self._save_index(idx)
        # Delete old file (outside lock, since file operations are independent)
        try:
            old_path.unlink(missing_ok=True)
        except Exception:
            ...
        self.logger.info("Split document: %s -> %d parts (old document deleted)", doc_id, len(parts))
        return metas

    def _ensure_raw_dir(self) -> None:
        """Ensure the raw directory exists."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_to_raw(self, content: str, current_number: int = None) -> Path:
        """Save content to a raw/CURRENT_<number>.md file.

        Args:
            content: The content to save
            current_number: The CURRENT document number (number in the raw filename, distinct from the seq parameter of add())
                          If None, extracts the maximum seq value from the content to use as the filename

        Returns:
            Path of the saved file
        """
        import re

        self._ensure_raw_dir()

        # If current_number is not provided, extract the maximum seq value from the content
        if current_number is None:
            # Match format: seq=number, or seq=number> or seq=number space
            seq_matches = re.findall(r'<seq=(\d+)[,\s>]', content)
            if seq_matches:
                current_number = max(int(seq) for seq in seq_matches)
                self.logger.debug("Extracted max seq value from content: %d", current_number)
            else:
                # If no seq found, use default value 0
                current_number = 0
                self.logger.warning("No seq value found in content, using default value 0")

        filename = f"CURRENT_{current_number}.md"
        path = self.raw_dir / filename
        path.write_text(content, encoding="utf-8")
        self.logger.info("Saved to raw directory: %s", filename)
        return path

    def flush_current_docs_to_raw(self) -> List[Path]:
        """Write all CURRENT* documents from the doc directory to raw, and delete the corresponding doc files and index metadata.

        Used at the end of processing a sample in the CURRENT stage to persist accumulated content as raw files,
        preventing data from mixing across samples.

        Returns:
            List of file paths written to raw.
        """
        self._ensure_dirs()

        current_files = [p for p in self.doc_dir.glob("*.md") if p.name.startswith("CURRENT")]
        if not current_files:
            return []

        saved_paths: List[Path] = []
        for doc_file in current_files:
            content = doc_file.read_text(encoding="utf-8")
            saved_paths.append(self.save_to_raw(content))

        for doc_file in current_files:
            doc_file.unlink(missing_ok=True)

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

    def list_raw_files(self) -> List[Path]:
        """List all raw/CURRENT_<number>.md files, sorted by number."""
        self._ensure_raw_dir()
        files = []
        for f in self.raw_dir.glob("CURRENT_*.md"):
            files.append(f)
        # Sort by number
        files.sort(key=lambda p: int(p.stem.split("_")[1]))
        return files

    def load_raw_content(self, current_number: int) -> str:
        """Load the content of a raw/CURRENT_<number>.md file.

        Args:
            current_number: The CURRENT document number (number in the raw filename)

        Returns:
            File content

        Raises:
            FileNotFoundError: Raised when the file does not exist
        """
        filename = f"CURRENT_{current_number}.md"
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Raw file does not exist: {filename}")
        return path.read_text(encoding="utf-8")

    def add_doc_to_index(self, meta: DocMeta) -> None:
        """Thread-safe addition of a document to the index (for multi-threaded environments)."""
        with self._index_lock:
            idx = self._load_index()
            idx.setdefault("docs", []).append(meta.__dict__)
            self._update_common_stats(idx)
            self._save_index(idx)

    def update_doc_in_index(self, doc_id: str, updater: callable) -> None:
        """Thread-safe update of a document in the index (for multi-threaded environments).

        Args:
            doc_id: Document ID
            updater: Callback function that receives a document dict and modifies it in place
        """
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
        """Completely delete the specified document (including file and metadata).

        Args:
            doc_id: ID of the document to delete
        """
        self._ensure_dirs()
        # Use thread lock to protect the entire delete operation
        with self._index_lock:
            idx = self._load_index()

            # Find target document
            target = None
            for d in idx.get("docs", []):
                if d.get("id") == doc_id:
                    target = d
                    break

            if target is None:
                raise ValueError(f"Document does not exist: {doc_id}")

            # Delete file (using path resolution to support both absolute and relative paths)
            path = self._resolve_path(target["path"])
            if path.exists():
                path.unlink()
                self.logger.info("Deleted document file: %s", path)

            # Remove from index
            idx["docs"] = [d for d in idx.get("docs", []) if d.get("id") != doc_id]
            self._update_common_stats(idx)
            self._save_index(idx)
            self.logger.info("Deleted document: %s", doc_id)

    def get_doc_content(self, doc_id: str) -> Optional[str]:
        """Read document file content by id. Returns None if not found."""
        meta = self.get_doc_by_id(doc_id)
        if meta is None:
            return None
        path = Path(meta.path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @staticmethod
    def list_users(root: Path, data_root: str) -> List[str]:
        """List all user_id directories under data_root."""
        base = Path(root) / data_root
        if not base.exists():
            return []
        return sorted(
            d.name for d in base.iterdir() if d.is_dir()
        )

    @staticmethod
    def delete_user_data(root: Path, data_root: str, user_id: str) -> None:
        """Remove entire data directory for a user."""
        import shutil
        user_dir = Path(root) / data_root / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)


def count_tokens(text: str) -> int:
    """Approximate token count for mixed Chinese/English text.

    Strategy:
    - Chinese/full-width characters: ~1.5 characters = 1 token
    - English words: count of space-separated words
    - Sum of both as the approximation
    """
    import re

    # Match Chinese characters (including Chinese punctuation)
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
    # Match English words (non-Chinese strings separated by spaces)
    english_words = len(re.findall(r'[a-zA-Z0-9]+', text))

    # Chinese: ~1.5 characters = 1 token; English: ~1 word = 1 token
    chinese_tokens = int(chinese_chars / 1.5)
    total_tokens = max(1, chinese_tokens + english_words)
    return total_tokens


def json_datetime_now() -> str:
    """Return Beijing time (Asia/Shanghai) as an ISO8601 string.

    Note: Unifies metadata timestamps to Beijing time per requirements, including timezone offset (+08:00).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


__all__ = ["MemoryManager", "DocMeta", "count_tokens", "json_datetime_now"]
