from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any, Optional

from .config import InfiniMemoryConfig
from .llm import LLMClient
from .manager import MemoryManager, count_tokens
from .memory import InfiniMemory


class Memory:
    """Simple interface for Infini Memory.

    Usage::

        from infini_memory_classic import Memory

        m = Memory()  # uses OPENAI_API_KEY env var
        m.add("I love spicy food", store="default", user_id="alice")
        results = m.search("food preferences", store="default", user_id="alice")
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-5-mini",
        data_root: str = "data",
        config: Optional[InfiniMemoryConfig] = None,
        **kwargs: Any,
    ):
        if config is not None:
            self._cfg = config
        else:
            self._cfg = InfiniMemoryConfig.from_kwargs(
                api_key=api_key,
                base_url=base_url,
                model=model,
                data_root=data_root,
                **kwargs,
            )

        self._llm = LLMClient(
            api_key=self._cfg.llm.openai_api_key,
            base_url=self._cfg.llm.openai_base_url,
            retry_max_attempts=self._cfg.llm.retry_max_attempts,
            retry_initial_wait=self._cfg.llm.retry_initial_wait,
            retry_max_wait=self._cfg.llm.retry_max_wait,
            retry_jitter=self._cfg.llm.retry_jitter,
        )
        self._engine = InfiniMemory()

    def _create_manager(self, store: str, user_id: str) -> MemoryManager:
        return MemoryManager(
            root=self._cfg.root,
            data_root=self._cfg.memory.data_root,
            doc_dir=self._cfg.memory.doc_dir,
            meta_dir=self._cfg.memory.metadata_dir,
            index_file=self._cfg.memory.index_file,
            store=store,
            user_id=user_id,
            storage=self._cfg.get_storage_backend(),
        )

    @staticmethod
    def _meta_to_dict(meta, content: Optional[str] = None) -> dict:
        d = {
            "id": meta.id,
            "path": meta.path,
            "summary": meta.summary,
            "tokens": meta.tokens,
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
            "update_count": meta.update_count,
            "parent_id": meta.parent_id,
            "current_epoch": meta.current_epoch,
        }
        if content is not None:
            d["content"] = content
        return d

    # ---- Core (LLM-powered) ----

    def add(self, messages: Any, *, store: str, user_id: str, **kwargs: Any) -> int:
        """Store messages in memory.

        Args:
            messages: Text string or list of chat messages to store.
            store: Memory store name.
            user_id: User identifier for memory isolation.

        Returns:
            Number of successfully processed items.
        """
        return self._engine.add(messages, store=store, user_id=user_id, cfg=self._cfg, llm=self._llm, **kwargs)

    def search(self, query: str, *, store: str, user_id: str, limit: Optional[int] = None) -> Any:
        """Search memories.

        Args:
            query: Search query string.
            store: Memory store name.
            user_id: User identifier for memory isolation.
            limit: Max number of documents to return. Uses config default if not set.

        Returns:
            Dict with "query" and "results" keys, or empty list if disabled.
        """
        if limit is not None:
            orig = self._cfg.memory.search_limit
            self._cfg.memory.search_limit = limit
            try:
                return self._engine.search(query, store=store, user_id=user_id, cfg=self._cfg, llm=self._llm)
            finally:
                self._cfg.memory.search_limit = orig
        return self._engine.search(query, store=store, user_id=user_id, cfg=self._cfg, llm=self._llm)

    # ---- Document CRUD ----

    def get(self, doc_id: str, *, store: str, user_id: str) -> Optional[dict]:
        """Get a single document by ID.

        Returns:
            Dict with document metadata and content, or None if not found.
        """
        mm = self._create_manager(store, user_id)
        meta = mm.get_doc_by_id(doc_id)
        if meta is None:
            return None
        content = mm.get_doc_content(doc_id)
        return self._meta_to_dict(meta, content=content)

    def get_all(self, *, store: str, user_id: str) -> list[dict]:
        """List all documents for a user, including content.

        Returns:
            List of dicts with document metadata and content.
        """
        mm = self._create_manager(store, user_id)
        result = []
        for meta in mm.list_docs():
            content = mm.get_doc_content(meta.id)
            result.append(self._meta_to_dict(meta, content=content))
        return result

    def list(self, *, store: str, user_id: str) -> list[dict]:
        """List all documents for a user.

        Returns:
            List of dicts with document metadata (no content).
        """
        mm = self._create_manager(store, user_id)
        return [self._meta_to_dict(m) for m in mm.list_docs()]

    def count(self, *, store: str, user_id: str) -> int:
        """Return the number of documents for a user."""
        mm = self._create_manager(store, user_id)
        return mm.get_common_stats().get("total_docs", 0)

    def stats(self, *, store: str, user_id: str) -> dict:
        """Return document statistics for a user.

        Returns:
            Dict with total_docs, avg_tokens, by_update_count, by_current_epoch.
        """
        mm = self._create_manager(store, user_id)
        return mm.get_common_stats()

    def history(self, *, store: str, user_id: str) -> list[dict]:
        """Return operation history (event.json) for a user."""
        mm = self._create_manager(store, user_id)
        return mm._load_events()

    def update(self, doc_id: str, content: str, summary: str, *, store: str, user_id: str) -> dict:
        """Update a document's content and summary.

        Args:
            doc_id: Document ID to update.
            content: New document content.
            summary: New summary text.
            store: Memory store name.
            user_id: User identifier.

        Returns:
            Dict with updated document metadata and content.

        Raises:
            ValueError: If the document does not exist.
        """
        mm = self._create_manager(store, user_id)
        tokens = count_tokens(content)
        meta = mm.update_doc(doc_id, content, summary, tokens)
        return self._meta_to_dict(meta, content=content)

    def delete(self, doc_id: str, *, store: str, user_id: str) -> None:
        """Delete a document by ID.

        Raises:
            ValueError: If the document does not exist.
        """
        mm = self._create_manager(store, user_id)
        mm.delete_doc(doc_id)

    def delete_all(self, *, store: str, user_id: str) -> int:
        """Delete all documents for a user, keeping the user directory.

        Returns:
            Number of documents deleted.
        """
        mm = self._create_manager(store, user_id)
        docs = mm.list_docs()
        for doc in docs:
            mm.delete_doc(doc.id)
        return len(docs)

    # ---- Store management ----

    def list_stores(self) -> list[str]:
        """List all memory store names."""
        return MemoryManager.list_stores(self._cfg.root, self._cfg.memory.data_root, storage=self._cfg.get_storage_backend())

    def create_store(self, store: str) -> None:
        """Create a memory store."""
        MemoryManager.create_store(self._cfg.root, self._cfg.memory.data_root, store, storage=self._cfg.get_storage_backend())

    def delete_store(self, store: str) -> None:
        """Delete a memory store and all its user data."""
        MemoryManager.delete_store(self._cfg.root, self._cfg.memory.data_root, store, storage=self._cfg.get_storage_backend())

    def get_store(self, store: str) -> dict:
        """Get info about a memory store.

        Returns:
            Dict with name, user_count, users.
        """
        return MemoryManager.get_store_info(self._cfg.root, self._cfg.memory.data_root, store, storage=self._cfg.get_storage_backend())

    # ---- User management ----

    def list_users(self, *, store: str) -> list[str]:
        """List all user IDs that have stored data in a store."""
        return MemoryManager.list_users(self._cfg.root, self._cfg.memory.data_root, store, storage=self._cfg.get_storage_backend())

    def delete_user(self, user_id: str, *, store: str) -> None:
        """Delete all data for a user within a store."""
        MemoryManager.delete_user_data(self._cfg.root, self._cfg.memory.data_root, store, user_id, storage=self._cfg.get_storage_backend())

    def reset(self) -> None:
        """Delete all data for all stores and users."""
        for store in self.list_stores():
            self.delete_store(store)

    # ---- Skills ----

    def _create_skills_manager(self, store: str, user_id: str):
        from .skills import SkillsManager
        return SkillsManager(
            root=self._cfg.root,
            data_root=self._cfg.memory.data_root,
            store=store,
            user_id=user_id,
            storage=self._cfg.get_storage_backend(),
            skills_dir=self._cfg.memory.skills_dir,
        )

    def generate_skills(self, *, store: str, user_id: str) -> list[dict]:
        """Analyze accumulated memories and generate reusable skills.

        Returns:
            List of created/updated skill metadata dicts.
        """
        from .skills import generate_skills_from_memory, _skill_meta_to_dict

        mm = self._create_manager(store, user_id)
        sm = self._create_skills_manager(store, user_id)
        docs = mm.list_docs()
        existing_skills = sm.list_skills()
        created = generate_skills_from_memory(
            docs, existing_skills, self._cfg, self._llm, mm, sm,
        )
        return [_skill_meta_to_dict(s) for s in created]

    def list_skills(self, *, store: str, user_id: str) -> list[dict]:
        """List all skills for a user.

        Returns:
            List of skill metadata dicts.
        """
        from .skills import _skill_meta_to_dict
        sm = self._create_skills_manager(store, user_id)
        return [_skill_meta_to_dict(s) for s in sm.list_skills()]

    def get_skill(self, name: str, *, store: str, user_id: str) -> Optional[dict]:
        """Get a skill's full content and metadata.

        Returns:
            Dict with skill metadata and content, or None if not found.
        """
        from .skills import _skill_meta_to_dict
        sm = self._create_skills_manager(store, user_id)
        meta = sm.get_skill(name)
        if meta is None:
            return None
        content = sm.get_skill_content(name)
        result = _skill_meta_to_dict(meta)
        result["content"] = content
        return result

    def save_skill(self, name: str, content: str, *, store: str, user_id: str) -> dict:
        """Create or update a skill manually.

        Args:
            name: Skill name (kebab-case).
            content: Full SKILL.md content.
            store: Memory store name.
            user_id: User identifier.

        Returns:
            Skill metadata dict.
        """
        from .skills import _skill_meta_to_dict
        sm = self._create_skills_manager(store, user_id)
        meta = sm.save_skill(name, content)
        return _skill_meta_to_dict(meta)

    def delete_skill(self, name: str, *, store: str, user_id: str) -> None:
        """Delete a skill by name."""
        sm = self._create_skills_manager(store, user_id)
        sm.delete_skill(name)

    def get_skills_dir(self, *, store: str, user_id: str) -> "Path":
        """Return the absolute path to the per-user skills directory."""
        sm = self._create_skills_manager(store, user_id)
        return sm.get_skills_dir_abs()

    # ---- Storage browsing ----

    def _get_storage(self):
        return self._cfg.get_storage_backend()

    def storage_info(self) -> dict:
        """Return storage backend info.

        Returns:
            Dict with ``storage_type`` key (``"local"`` or ``"s3"``).
        """
        backend = self._get_storage()
        name = type(backend).__name__
        st = "s3" if "S3" in name else "local"
        return {"storage_type": st}

    def storage_list(self, path: str) -> list[dict]:
        """List directory contents.

        Args:
            path: Relative directory path.

        Returns:
            Sorted list of dicts with ``name``, ``path``, and ``is_dir`` keys.
            Directories come first, then files, both sorted by name.
        """
        backend = self._get_storage()
        names = backend.listdir(path)
        entries: list[dict] = []
        for name in names:
            entry_path = posixpath.join(path, name) if path else name
            try:
                is_directory = backend.is_dir(entry_path)
            except Exception:
                is_directory = False
            entries.append({"name": name, "path": entry_path, "is_dir": is_directory})
        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
        return entries

    def storage_read(self, path: str) -> str:
        """Read file content.

        Args:
            path: Relative file path.

        Returns:
            File content as string.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        return self._get_storage().read_text(path)

    def storage_write(self, path: str, content: str) -> None:
        """Write content to a file (creates parent directories as needed).

        Args:
            path: Relative file path.
            content: Text content to write.
        """
        self._get_storage().write_text(path, content)

    def storage_delete(self, path: str) -> None:
        """Delete a file.

        Args:
            path: Relative file path.
        """
        self._get_storage().delete(path)

    def storage_mkdir(self, path: str) -> None:
        """Create a directory (including parents).

        Args:
            path: Relative directory path.
        """
        self._get_storage().mkdir(path)

    def storage_rmtree(self, path: str) -> None:
        """Remove an entire directory tree.

        Args:
            path: Relative directory path.
        """
        self._get_storage().rmtree(path)

    def storage_exists(self, path: str) -> bool:
        """Check if a path exists.

        Args:
            path: Relative path.

        Returns:
            True if the path exists.
        """
        return self._get_storage().exists(path)

    def storage_is_dir(self, path: str) -> bool:
        """Check if a path is a directory.

        Args:
            path: Relative path.

        Returns:
            True if the path is a directory.
        """
        return self._get_storage().is_dir(path)
