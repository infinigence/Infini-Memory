from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .config import InfiniMemoryConfig
from .llm import LLMClient
from .manager import MemoryManager, count_tokens
from .memory import InfiniMemory


class Memory:
    """Simple interface for Infini Memory.

    Usage::

        from infini_memory import Memory

        m = Memory()  # uses OPENAI_API_KEY env var
        m.add("I love spicy food", user_id="alice")
        results = m.search("food preferences", user_id="alice")
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

    def _create_manager(self, user_id: str) -> MemoryManager:
        return MemoryManager(
            root=self._cfg.root,
            data_root=self._cfg.memory.data_root,
            doc_dir=self._cfg.memory.doc_dir,
            meta_dir=self._cfg.memory.metadata_dir,
            index_file=self._cfg.memory.index_file,
            user_id=user_id,
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

    def add(self, messages: Any, *, user_id: str, **kwargs: Any) -> int:
        """Store messages in memory.

        Args:
            messages: Text string or list of chat messages to store.
            user_id: User identifier for memory isolation.

        Returns:
            Number of successfully processed items.
        """
        return self._engine.add(messages, user_id=user_id, cfg=self._cfg, llm=self._llm, **kwargs)

    def search(self, query: str, *, user_id: str, limit: Optional[int] = None) -> Any:
        """Search memories.

        Args:
            query: Search query string.
            user_id: User identifier for memory isolation.
            limit: Max number of documents to return. Uses config default if not set.

        Returns:
            Dict with "query" and "results" keys, or empty list if disabled.
        """
        if limit is not None:
            orig = self._cfg.memory.search_limit
            self._cfg.memory.search_limit = limit
            try:
                return self._engine.search(query, user_id=user_id, cfg=self._cfg, llm=self._llm)
            finally:
                self._cfg.memory.search_limit = orig
        return self._engine.search(query, user_id=user_id, cfg=self._cfg, llm=self._llm)

    # ---- Document CRUD ----

    def get(self, doc_id: str, *, user_id: str) -> Optional[dict]:
        """Get a single document by ID.

        Returns:
            Dict with document metadata and content, or None if not found.
        """
        mm = self._create_manager(user_id)
        meta = mm.get_doc_by_id(doc_id)
        if meta is None:
            return None
        content = mm.get_doc_content(doc_id)
        return self._meta_to_dict(meta, content=content)

    def get_all(self, *, user_id: str) -> list[dict]:
        """List all documents for a user, including content.

        Returns:
            List of dicts with document metadata and content.
        """
        mm = self._create_manager(user_id)
        result = []
        for meta in mm.list_docs():
            content = mm.get_doc_content(meta.id)
            result.append(self._meta_to_dict(meta, content=content))
        return result

    def list(self, *, user_id: str) -> list[dict]:
        """List all documents for a user.

        Returns:
            List of dicts with document metadata (no content).
        """
        mm = self._create_manager(user_id)
        return [self._meta_to_dict(m) for m in mm.list_docs()]

    def count(self, *, user_id: str) -> int:
        """Return the number of documents for a user."""
        mm = self._create_manager(user_id)
        return mm.get_common_stats().get("total_docs", 0)

    def stats(self, *, user_id: str) -> dict:
        """Return document statistics for a user.

        Returns:
            Dict with total_docs, avg_tokens, by_update_count, by_current_epoch.
        """
        mm = self._create_manager(user_id)
        return mm.get_common_stats()

    def history(self, *, user_id: str) -> list[dict]:
        """Return operation history (event.json) for a user."""
        mm = self._create_manager(user_id)
        return mm._load_events()

    def update(self, doc_id: str, content: str, summary: str, *, user_id: str) -> dict:
        """Update a document's content and summary.

        Args:
            doc_id: Document ID to update.
            content: New document content.
            summary: New summary text.
            user_id: User identifier.

        Returns:
            Dict with updated document metadata and content.

        Raises:
            ValueError: If the document does not exist.
        """
        mm = self._create_manager(user_id)
        tokens = count_tokens(content)
        meta = mm.update_doc(doc_id, content, summary, tokens)
        return self._meta_to_dict(meta, content=content)

    def delete(self, doc_id: str, *, user_id: str) -> None:
        """Delete a document by ID.

        Raises:
            ValueError: If the document does not exist.
        """
        mm = self._create_manager(user_id)
        mm.delete_doc(doc_id)

    def delete_all(self, *, user_id: str) -> int:
        """Delete all documents for a user, keeping the user directory.

        Returns:
            Number of documents deleted.
        """
        mm = self._create_manager(user_id)
        docs = mm.list_docs()
        for doc in docs:
            mm.delete_doc(doc.id)
        return len(docs)

    # ---- User management ----

    def list_users(self) -> list[str]:
        """List all user IDs that have stored data."""
        return MemoryManager.list_users(self._cfg.root, self._cfg.memory.data_root)

    def delete_user(self, user_id: str) -> None:
        """Delete all data for a user."""
        MemoryManager.delete_user_data(self._cfg.root, self._cfg.memory.data_root, user_id)

    def reset(self) -> None:
        """Delete all data for all users."""
        for user_id in self.list_users():
            self.delete_user(user_id)
