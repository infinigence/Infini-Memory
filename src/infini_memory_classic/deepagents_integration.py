"""DeepAgents integration for Infini Memory.

Provides LangChain tools and a ``create_deep_agent``-powered factory so that
a deep agent can read, write, and manage persistent memory out of the box.

Requires the optional ``deepagents`` extra::

    pip install infini-memory[deepagents]

Usage::

    from infini_memory_classic.deepagents_integration import create_memory_agent

    agent = create_memory_agent(
        model="openai:gpt-5-mini",
        user_id="alice",
        api_key="sk-...",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Remember that I love sushi"}]}
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

try:
    from deepagents import create_deep_agent
    from langchain_core.tools import tool
except ImportError as _exc:
    raise ImportError(
        "deepagents integration requires the 'deepagents' package. "
        "Install with: pip install infini-memory[deepagents]"
    ) from _exc

from .config import InfiniMemoryConfig
from .convenience import Memory

logger = logging.getLogger("infini_memory_classic.deepagents")

MEMORY_SYSTEM_PROMPT = """\
You have access to a persistent memory system. Use it to:
- Store important facts, preferences, and information using add_memory
- Retrieve relevant information using search_memory before answering questions
- Manage stored memories using get_memory, list_memories, update_memory, delete_memory when needed

Always search memory first when the user asks about something that might have been stored before.\
"""


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def create_memory_tools(
    memory: Memory,
    store: str,
    user_id: str,
    *,
    include_crud: bool = True,
    include_admin: bool = False,
    include_skills: bool = True,
) -> list:
    """Build LangChain tools backed by a :class:`~infini_memory_classic.Memory` instance.

    Args:
        memory: An initialized ``Memory`` instance.
        store: Memory store name.
        user_id: User identifier for memory isolation.
        include_crud: Include document CRUD tools (get, list, update, delete).
        include_admin: Include admin tools (stats).
        include_skills: Include skill management tools (generate, list, get).

    Returns:
        A list of LangChain ``@tool``-decorated functions.
    """

    @tool
    def add_memory(content: str) -> str:
        """Store information in persistent memory.

        Args:
            content: The text content to memorize.
        """
        count = memory.add(content, store=store, user_id=user_id)
        return f"Stored successfully. Items processed: {count}"

    @tool
    def search_memory(query: str, limit: int = 5) -> str:
        """Search memories for relevant information.

        Args:
            query: Natural language search query.
            limit: Maximum number of documents to return (default 5).
        """
        results = memory.search(query, store=store, user_id=user_id, limit=limit)
        return json.dumps(results, ensure_ascii=False, default=_json_default)

    tools: list = [add_memory, search_memory]

    if include_crud:

        @tool
        def get_memory(doc_id: str) -> str:
            """Retrieve a specific memory document by its ID.

            Args:
                doc_id: The document ID to retrieve.
            """
            doc = memory.get(doc_id, store=store, user_id=user_id)
            if doc is None:
                return json.dumps({"error": f"Document '{doc_id}' not found."})
            return json.dumps(doc, ensure_ascii=False, default=_json_default)

        @tool
        def list_memories() -> str:
            """List all memory documents (metadata only, no content)."""
            docs = memory.list(store=store, user_id=user_id)
            return json.dumps(docs, ensure_ascii=False, default=_json_default)

        @tool
        def update_memory(doc_id: str, content: str, summary: str) -> str:
            """Update an existing memory document.

            Args:
                doc_id: The document ID to update.
                content: New document content.
                summary: New document summary.
            """
            try:
                updated = memory.update(doc_id, content, summary, store=store, user_id=user_id)
                return json.dumps(updated, ensure_ascii=False, default=_json_default)
            except ValueError as exc:
                return json.dumps({"error": str(exc)})

        @tool
        def delete_memory(doc_id: str) -> str:
            """Delete a memory document.

            Args:
                doc_id: The document ID to delete.
            """
            try:
                memory.delete(doc_id, store=store, user_id=user_id)
                return f"Document '{doc_id}' deleted."
            except ValueError as exc:
                return json.dumps({"error": str(exc)})

        tools.extend([get_memory, list_memories, update_memory, delete_memory])

    if include_admin:

        @tool
        def memory_stats() -> str:
            """Get memory statistics (total docs, average tokens, etc.)."""
            stats = memory.stats(store=store, user_id=user_id)
            return json.dumps(stats, ensure_ascii=False, default=_json_default)

        tools.append(memory_stats)

    if include_skills:

        @tool
        def generate_skills() -> str:
            """Analyze accumulated memories and generate reusable skills.

            Skills capture recurring patterns, preferences, and domain expertise
            from the user's memory documents.
            """
            results = memory.generate_skills(store=store, user_id=user_id)
            return json.dumps(results, ensure_ascii=False, default=_json_default)

        @tool
        def list_skills() -> str:
            """List all generated skills for this user."""
            skills = memory.list_skills(store=store, user_id=user_id)
            return json.dumps(skills, ensure_ascii=False, default=_json_default)

        @tool
        def get_skill(name: str) -> str:
            """Get a skill's full content by name.

            Args:
                name: The skill name (kebab-case).
            """
            skill = memory.get_skill(name, store=store, user_id=user_id)
            if skill is None:
                return json.dumps({"error": f"Skill '{name}' not found."})
            return json.dumps(skill, ensure_ascii=False, default=_json_default)

        tools.extend([generate_skills, list_skills, get_skill])

    return tools


def create_memory_agent(
    *,
    model: str = "openai:gpt-5-mini",
    memory: Optional[Memory] = None,
    store: str = "default",
    user_id: str = "default",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    data_root: str = "data",
    config: Optional[InfiniMemoryConfig] = None,
    system_prompt: str = "",
    extra_tools: Optional[Sequence[Any]] = None,
    include_crud: bool = True,
    include_admin: bool = False,
    include_skills: bool = True,
    **agent_kwargs: Any,
) -> Any:
    """Create a deep agent with built-in memory tools.

    Provide either a pre-built ``memory`` instance **or** credentials
    (``api_key``/``base_url``/``config``) to construct one automatically.

    When ``include_skills`` is ``True`` (default), per-user skills are
    automatically discovered and wired into the agent via DeepAgents'
    ``SkillsMiddleware``.

    Args:
        model: Model identifier for the agent's reasoning LLM
            (e.g. ``"openai:gpt-5-mini"``, ``"anthropic:claude-sonnet-4-20250514"``).
        memory: An existing :class:`~infini_memory_classic.Memory` instance. If ``None``,
            one is created from the credential arguments.
        user_id: User identifier for memory isolation.
        api_key: OpenAI API key (used only when ``memory`` is ``None``).
        base_url: OpenAI-compatible base URL (used only when ``memory`` is ``None``).
        data_root: Data directory (used only when ``memory`` is ``None``).
        config: Full :class:`~infini_memory_classic.InfiniMemoryConfig` (used only when
            ``memory`` is ``None``).
        system_prompt: Additional system instructions prepended to the default
            memory prompt.
        extra_tools: Extra LangChain tools to include alongside memory tools.
        include_crud: Include document CRUD tools (default ``True``).
        include_admin: Include admin tools like stats (default ``False``).
        include_skills: Include skill management tools and auto-discover
            per-user skills for the agent (default ``True``).
        **agent_kwargs: Forwarded to ``deepagents.create_deep_agent()``.

    Returns:
        A compiled LangGraph agent ready to ``.invoke()`` or ``.stream()``.
    """
    if memory is None:
        if config is not None:
            memory = Memory(config=config)
        else:
            kwargs: dict[str, Any] = {"data_root": data_root}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            memory = Memory(**kwargs)

    tools = create_memory_tools(
        memory,
        store,
        user_id,
        include_crud=include_crud,
        include_admin=include_admin,
        include_skills=include_skills,
    )
    if extra_tools:
        tools = list(tools) + list(extra_tools)

    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    prompt_parts.append(MEMORY_SYSTEM_PROMPT)
    combined_prompt = "\n\n".join(prompt_parts)

    # Auto-discover per-user skills for DeepAgents SkillsMiddleware
    da_skills = agent_kwargs.pop("skills", None)
    da_backend = agent_kwargs.pop("backend", None)

    if include_skills and da_skills is None:
        try:
            from .skills import SkillsManager
            sm = memory._create_skills_manager(store, user_id)
            skills_dir = sm.get_skills_dir_abs()
            if skills_dir.exists() and any(skills_dir.iterdir()):
                from deepagents.backends import FilesystemBackend
                da_skills = ["/"]
                da_backend = FilesystemBackend(
                    root_dir=skills_dir, virtual_mode=True,
                )
                logger.info(
                    "Auto-discovered skills for user %s at %s",
                    user_id, skills_dir,
                )
        except ImportError:
            logger.debug("FilesystemBackend not available, skipping skills auto-discovery")
        except Exception as e:
            logger.debug("Skills auto-discovery failed: %s", e)

    logger.info(
        "Creating memory agent: model=%s, user_id=%s, tools=%d, skills=%s",
        model,
        user_id,
        len(tools),
        "yes" if da_skills else "no",
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=combined_prompt,
        skills=da_skills,
        backend=da_backend,
        **agent_kwargs,
    )
