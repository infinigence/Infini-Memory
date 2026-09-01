from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Dict, Tuple, Callable, Literal

from pydantic import BaseModel, Field
from .config import InfiniMemoryConfig
from . import term_color
from .term_color import format_git_diff, yellow
from .prompts import (
    EXTRACT_MEMORY_PROMPT,
    REWRITE_CURRENT_PROMPT,
    PLAN_UPDATE_PROMPT,
    REWRITE_DOC_PROMPT,
    SELECT_MERGE_GROUPS_PROMPT,
    MERGE_DOCS_PROMPT,
    SEARCH_MEMORY_PROMPT,
    AGENTIC_TOOL_AGENT_PROMPT,
)
from .llm import LLMClient
from .manager import MemoryManager, count_tokens, json_datetime_now, DocMeta


def _llm_chat_with_retry(
    llm: LLMClient,
    messages: List[Dict[str, str]],
    model: str,
    cfg: InfiniMemoryConfig,
    context: str = "",
) -> str:
    """LLM call wrapper with retry mechanism (now handled uniformly by LLMClient)."""
    logger = logging.getLogger("infini_memory_classic")
    try:
        return llm.chat(messages, model=model, temperature=cfg.llm.temperature)
    except Exception as e:
        error_msg = f"{context}LLM call failed: {e}" if context else f"LLM call failed: {e}"
        logger.error(term_color.red(error_msg, enabled=True))
        raise


class MemoryItem(BaseModel):
    id: int
    text: str


class SearchResult(BaseModel):
    id: int
    text: str
    score: float = Field(ge=0.0, le=1.0)


class InfiniMemory(BaseModel):
    """File-system based memory implementation: persists to disk on every add or update.

    Removes in-memory database, relies directly on file system storage and indexing (managed by MemoryManager).
    """

    model_config = dict(arbitrary_types_allowed=True)

    def _to_text(self, m: Any) -> Optional[str]:
        """Convert any object to a text string."""
        if m is None:
            return None
        if isinstance(m, str):
            return m
        try:
            return str(m)
        except Exception:  # noqa: BLE001
            return None

    def _extract_markdown(self, user_text: str, cfg: InfiniMemoryConfig, llm: LLMClient, mm: MemoryManager, seq: int = 1) -> str:
        """Use LLM to extract structured Markdown from user text.

        Args:
            user_text: User input text
            cfg: Configuration object
            llm: LLMClient instance
            mm: MemoryManager instance
            seq: Current sequence number (passed in by caller)

        Returns the extracted Markdown content (with @@SEQ@@ placeholders replaced).
        """
        logger = logging.getLogger("infini_memory_classic")

        # Use the passed-in seq parameter
        current_seq = seq

        # Use EXTRACT_MEMORY_PROMPT to extract Markdown (with retry)
        extract_sys = EXTRACT_MEMORY_PROMPT.format(
            summary_length=cfg.memory.summary_length,
        )

        # Skills-guided extraction: prepend domain hints if available
        if cfg.memory.skills_enabled and cfg.memory.skills_extraction_hints:
            try:
                from .skills import SkillsManager
                from .prompts import EXTRACTION_HINTS_CONTEXT
                sm = SkillsManager(
                    root=cfg.root,
                    data_root=cfg.memory.data_root,
                    store=mm.store,
                    user_id=mm.user_id,
                    storage=cfg.get_storage_backend(),
                    skills_dir=cfg.memory.skills_dir,
                )
                if hints:
                    hints_block = EXTRACTION_HINTS_CONTEXT.format(
                        hints="\n".join(f"- {h}" for h in hints)
                    )
                    extract_sys = hints_block + "\n\n" + extract_sys
                    logger.debug("[Extract] Injected %d extraction hints from skills", len(hints))
            except Exception as e:
                logger.debug("[Extract] Failed to load skills extraction hints: %s", e)

        extract_md = _llm_chat_with_retry(
            llm,
            [
                {"role": "system", "content": extract_sys},
                {"role": "user", "content": user_text},
            ],
            cfg.llm.model,
            cfg,
            context="[Extract]",
        )

        # Replace @@SEQ@@ placeholder with actual sequence number
        extract_md = extract_md.replace("@@SEQ@@", str(current_seq))

        # Note: seq is managed by the caller, no longer calling increment_seq() here

        return extract_md

    def _create_current_doc(
        self,
        extract_md: str,
        current_epoch: int,
        cfg: InfiniMemoryConfig,
        mm: MemoryManager,
        doc_id: str = "CURRENT",
    ) -> None:
        """Create a new CURRENT document.

        Args:
            extract_md: Extracted Markdown content
            current_epoch: Current epoch number
            cfg: Configuration object
            mm: MemoryManager instance
            doc_id: Document ID (defaults to "CURRENT")
        """
        logger = logging.getLogger("infini_memory_classic")

        # Extract body (remove YAML Frontmatter)
        _, extract_body = extract_summary_from_markdown(extract_md)

        # Only count tokens for the body
        ex_tokens = count_tokens(extract_body)

        # Add directly as CURRENT document
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import uuid
        ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{doc_id}.md"
        path = mm.doc_dir / filename
        now = json_datetime_now()
        from .manager import DocMeta
        meta = DocMeta(
            id=doc_id,
            path=str(path.relative_to(mm.root)),
            created_at=now,
            updated_at=now,
            tokens=ex_tokens,
            summary="",  # CURRENT document does not need a summary
            current_epoch=current_epoch,  # Initially 0
        )
        # Ensure directories exist
        mm._ensure_dirs()

        # CURRENT document: write body content only (without summary from YAML Frontmatter)
        # Use simplified YAML Frontmatter without summary
        simple_content = f"""---
id: {meta.id}
created_at: {meta.created_at}
updated_at: {meta.updated_at}
tokens: {meta.tokens}
current_epoch: {meta.current_epoch}
---
{extract_body}"""
        mm.write_file(str(path.relative_to(mm.root)), simple_content)

        # Use thread-safe method to add to index
        mm.add_doc_to_index(meta)
        # Record operation history
        mm.log_event({
            "event": "created_current",
            "doc_id": meta.id,
            "timestamp": meta.created_at,
        })

        try:
            logger.info(
                term_color.yellow(
                    f"Created CURRENT document | tokens: {ex_tokens}",
                    enabled=True,
                )
            )
        except Exception:  # noqa: BLE001
            logger.info(f"Created CURRENT document | tokens: {ex_tokens}")

        # Output new document diff
        try:
            diff_output = format_git_diff("", simple_content, enabled=True)
            logging.getLogger("infini_memory_classic").info(
                term_color.cyan(
                    f"[New document diff]\n{diff_output}",
                    enabled=True,
                )
            )
        except Exception:  # noqa: BLE001
            pass

    def _append_to_current(
        self,
        extract_md: str,
        current_doc: DocMeta,
        cfg: InfiniMemoryConfig,
        mm: MemoryManager,
    ) -> DocMeta:
        """Append new content to CURRENT document.

        Returns the updated metadata.
        """
        logger = logging.getLogger("infini_memory_classic")

        # Read old content
        old_content = mm.read_file(current_doc.path)

        # Extract body of old content (remove YAML Frontmatter)
        _, old_body = extract_summary_from_markdown(old_content)
        # Extract body of new content
        _, extract_body = extract_summary_from_markdown(extract_md)

        # Directly append new content to old content
        new_body = old_body + "\n\n" + extract_body
        new_tokens = count_tokens(new_body)

        # Update metadata and timestamp (do not update summary, CURRENT document has no summary)
        now = json_datetime_now()
        meta = mm.update_doc(current_doc.id, new_body, summary="", tokens=new_tokens)
        meta.updated_at = now

        # Update YAML Frontmatter
        final_content = update_yaml_frontmatter(new_body, meta)
        mm.write_file(current_doc.path, final_content)
        # Use thread-safe method to update metadata in index
        mm.update_doc_in_index(current_doc.id, lambda d: d.update(meta.__dict__))
        # Record operation history
        mm.log_event({
            "event": "updated_current",
            "doc_id": current_doc.id,
            "timestamp": now,
        })

        # Output update info
        try:
            delta = new_tokens - current_doc.tokens
            delta_text = f"({delta:+d})"
            logging.getLogger("infini_memory_classic").info(
                term_color.cyan(
                    f"Updated CURRENT document | tokens: {current_doc.tokens} -> {new_tokens} {delta_text}",
                    enabled=True,
                )
            )
        except Exception:  # noqa: BLE001
            logging.getLogger("infini_memory_classic").info(f"Updated CURRENT document | tokens: {new_tokens}")

        # Output document diff
        try:
            diff_output = format_git_diff(old_content, final_content, enabled=True)
            logging.getLogger("infini_memory_classic").info(
                term_color.cyan(
                    f"[Document change diff]\n{diff_output}",
                    enabled=True,
                )
            )
        except Exception:  # noqa: BLE001
            pass  # Diff output failure does not affect the main flow

        return meta

    def _get_split_plan(
        self,
        content_body: str,
        existing_docs_list: List[Dict[str, str]],
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
        splitting_depth: int = 0,
    ) -> Dict:
        """Call PLAN_UPDATE_PROMPT to get the split plan.

        Two splitting scenarios:
        1. Preventive splitting: Before calling LLM, recursively split content until all parts are
           within plan_split_threshold, then call PLAN_UPDATE_PROMPT for each part and merge results
        2. Fallback splitting: When LLM call succeeds but the response cannot be parsed as JSON,
           split content by newlines into two sub-documents

        Args:
            content_body: Content to split
            existing_docs_list: List of existing documents
            cfg: Configuration object
            llm: LLMClient instance
            splitting_depth: Current splitting depth (to avoid infinite recursion)

        Returns the parsed plan dictionary (containing new_docs and updates).
        """
        logger = logging.getLogger("infini_memory_classic")

        max_call_retries = 3  # LLM call retry count
        max_splitting_depth = 10  # Maximum splitting recursion depth

        # Scenario 1: Preventive splitting - complete all splits first, then call LLM

        # Try splitting by H1 headings first (if content is large)
        content_tokens = count_tokens(content_body)
        if content_tokens > cfg.memory.plan_split_threshold:
            # If content exceeds threshold, try splitting by H1 headings first
            logger.info(
                f"[Split] Content tokens ({content_tokens}) exceed threshold ({cfg.memory.plan_split_threshold}), trying to split by H1 headings"
            )
            heading_chunks = self._split_content_by_heading(
                content_body, cfg.memory.markdown_length
            )
            # If heading split produced multiple chunks, use them
            if len(heading_chunks) > 1:
                logger.info(
                    term_color.cyan(
                        f"[Split] H1 heading split succeeded, got {len(heading_chunks)} chunks",
                        enabled=True,
                    )
                )
                content_chunks = heading_chunks
            else:
                # Heading split failed (only 1 chunk or split failed), fall back to recursive splitting
                logger.info(
                    "[Split] H1 heading split did not produce multiple chunks, falling back to recursive splitting"
                )
                content_chunks = self._split_content_to_chunks(
                    content_body, cfg.memory.plan_split_threshold, max_splitting_depth
                )
        else:
            # Content does not exceed threshold, use recursive splitting
            content_chunks = self._split_content_to_chunks(
                content_body, cfg.memory.plan_split_threshold, max_splitting_depth
            )

        # If split into multiple chunks, call LLM for each and merge
        if len(content_chunks) > 1:
            logger.info(
                term_color.cyan(
                    f"[Split] Content split into {len(content_chunks)} chunks, using {cfg.memory.plan_split_threads} threads to call PLAN_UPDATE_PROMPT separately",
                    enabled=True,
                )
            )

            # Multi-threaded parallel processing
            all_plans = [None] * len(content_chunks)  # Pre-allocate result array to maintain order

            with ThreadPoolExecutor(
                max_workers=cfg.memory.plan_split_threads,
                thread_name_prefix="PlanSplit"
            ) as executor:
                # Submit all tasks
                futures = {}
                for idx, chunk in enumerate(content_chunks):
                    logger.info(
                        term_color.cyan(
                            f"[Split] Submitting chunk {idx + 1}/{len(content_chunks)} ({count_tokens(chunk)} tokens)",
                            enabled=True,
                        )
                    )
                    future = executor.submit(
                        self._call_plan_update_prompt_for_chunk,
                        idx, chunk, existing_docs_list, cfg, llm, max_call_retries
                    )
                    futures[future] = idx

                # Collect results
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        plan = future.result()
                        all_plans[idx] = plan
                        logger.info(
                            term_color.green(
                                f"[Split] Chunk {idx + 1}/{len(content_chunks)} processing completed",
                                enabled=True,
                            )
                        )
                    except Exception as e:
                        logger.error(
                            term_color.red(
                                f"[Split] Chunk {idx + 1}/{len(content_chunks)} processing failed: {e}",
                                enabled=True,
                            )
                        )
                        # Failed chunks use empty plan
                        all_plans[idx] = {"new_docs": [], "updates": []}

            # Merge all plans (all chunks use the same existing_docs_list, processed independently then merged)
            merged_plan = {"new_docs": [], "updates": []}
            for plan in all_plans:
                if plan:
                    merged_plan = self._merge_split_plans(merged_plan, plan)

            logger.info(
                term_color.green(
                    f"[Split] Preventive splitting completed: total {len(merged_plan['new_docs'])} new documents, {len(merged_plan['updates'])} updates",
                    enabled=True,
                )
            )

            return merged_plan

        # Content was not split (single chunk), call LLM directly (may trigger fallback splitting)
        return self._call_plan_update_prompt_with_fallback(
            content_body, existing_docs_list, cfg, llm, max_call_retries, splitting_depth, max_splitting_depth
        )

    def _call_plan_update_prompt_for_chunk(
        self,
        idx: int,
        chunk: str,
        existing_docs_list: List[Dict[str, str]],
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
        max_retries: int,
    ) -> Dict:
        """Call PLAN_UPDATE_PROMPT for a single chunk (used in multi-threading).

        Args:
            idx: Chunk index
            chunk: Content chunk
            existing_docs_list: List of existing documents
            cfg: Configuration object
            llm: LLMClient instance
            max_retries: Maximum retry count

        Returns the parsed plan dictionary.
        """
        logger = logging.getLogger("infini_memory_classic")
        logger.info(
            term_color.cyan(
                f"[Split][Thread-{idx}] Starting chunk processing ({count_tokens(chunk)} tokens)",
                enabled=True,
            )
        )

        plan = self._call_plan_update_prompt(
            chunk, existing_docs_list, cfg, llm, max_retries
        )

        logger.info(
            term_color.green(
                f"[Split][Thread-{idx}] Chunk processing completed: {len(plan.get('new_docs', []))} new documents, {len(plan.get('updates', []))} updates",
                enabled=True,
            )
        )

        return plan

    def _split_content_by_heading(
        self, content: str, markdown_length: int
    ) -> List[str]:
        """Split content by Markdown H1 headings (#).

        Splitting rules:
        1. Each H1 heading and its content form a chunk
        2. Merge heading chunks one by one until total tokens approach but do not exceed markdown_length
        3. If a single heading chunk itself exceeds markdown_length, do not further split within that heading

        Args:
            content: Content to split
            markdown_length: Token threshold

        Returns the list of split content chunks.
        """
        logger = logging.getLogger("infini_memory_classic")

        # Split content by H1 headings
        import re
        # Match H1 headings (# Title), capture heading and content
        # Use split to preserve separators for reconstruction
        parts = re.split(r'^(# [^\n]*\n)', content, flags=re.MULTILINE)

        # parts[0] may be preamble content (before the first heading)
        # From parts[1], odd indices are heading lines, even indices are corresponding content
        heading_blocks = []
        if parts[0].strip():
            # If there is preamble content, treat it as a special chunk
            heading_blocks.append({"title": "[Preamble]", "content": parts[0].strip()})

        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                title_line = parts[i].strip()
                content_part = parts[i + 1].strip() if (i + 1) < len(parts) else ""
                heading_blocks.append({"title": title_line, "content": content_part})

        if not heading_blocks:
            return [content]

        # Merge heading blocks into final chunks
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for block in heading_blocks:
            block_text = block["title"] + "\n" + block["content"]
            block_tokens = count_tokens(block_text)

            # If adding the new block would exceed threshold, save current chunk first
            if current_tokens > 0 and current_tokens + block_tokens > markdown_length:
                chunks.append(current_chunk)
                current_chunk = block_text
                current_tokens = block_tokens
            else:
                # Otherwise add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + block_text
                else:
                    current_chunk = block_text
                current_tokens += block_tokens

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Print final split results (including H1 heading count)
        heading_count_in_chunks = sum(1 for chunk in chunks for heading in heading_blocks if heading["title"] in chunk)
        logger.info(
            term_color.cyan(
                f"[Split] H1 heading split completed: {len(chunks)} chunks, containing {len(heading_blocks)} H1 headings",
                enabled=True,
            )
        )
        for idx, chunk in enumerate(chunks, 1):
            chunk_tokens = count_tokens(chunk)
            logger.info(
                term_color.cyan(
                    f"[Split]   Chunk {idx}/{len(chunks)}: {chunk_tokens} tokens",
                    enabled=True,
                )
            )

        return chunks

    def _split_content_to_chunks(
        self, content: str, threshold: int, max_depth: int, current_depth: int = 0
    ) -> List[str]:
        """Recursively split content until all chunks are within the threshold.

        Args:
            content: Content to split
            threshold: Token threshold
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns the list of split content chunks.
        """
        logger = logging.getLogger("infini_memory_classic")
        content_tokens = count_tokens(content)

        # If content is within threshold or max depth reached, return single chunk
        if content_tokens <= threshold or current_depth >= max_depth:
            return [content]

        # Split into two parts
        part1, part2 = split_content_by_newlines(content)

        if not part1 or not part2:
            # Cannot split, return single chunk
            logger.warning(
                term_color.yellow(
                    f"[Split] Content cannot be split ({content_tokens} tokens), returning single chunk",
                    enabled=True,
                )
            )
            return [content]

        # Recursively split both parts
        logger.debug(
            f"[Split] Split depth {current_depth + 1}/{max_depth}: {content_tokens} tokens -> "
            f"part1={count_tokens(part1)} tokens, part2={count_tokens(part2)} tokens"
        )

        chunks1 = self._split_content_to_chunks(part1, threshold, max_depth, current_depth + 1)
        chunks2 = self._split_content_to_chunks(part2, threshold, max_depth, current_depth + 1)

        return chunks1 + chunks2

    def _call_plan_update_prompt(
        self,
        content_body: str,
        existing_docs_list: List[Dict[str, str]],
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
        max_retries: int,
    ) -> Dict:
        """Call PLAN_UPDATE_PROMPT to get the split plan.

        Args:
            content_body: Content to split
            existing_docs_list: List of existing documents
            cfg: Configuration object
            llm: LLMClient instance
            max_retries: Maximum retry count

        Returns the parsed plan dictionary (containing new_docs and updates).
        """
        logger = logging.getLogger("infini_memory_classic")

        # Format PLAN_UPDATE_PROMPT
        plan_prompt = PLAN_UPDATE_PROMPT.format(
            markdown_length=cfg.memory.markdown_length
        )

        content_tokens = count_tokens(content_body)

        # Call LLM, retry if empty response
        plan_json = None
        for call_attempt in range(1, max_retries + 1):
            # Log input info for debugging
            logger.info(
                term_color.cyan(
                    f"[Split] PLAN_UPDATE_PROMPT input: new_content tokens={content_tokens:,}, existing_docs={len(existing_docs_list)}",
                    enabled=True,
                )
            )

            # Call LLM (do not pass max_tokens)
            plan_json = _llm_chat_with_retry(
                llm,
                [
                    {"role": "system", "content": plan_prompt},
                    {"role": "user", "content": json_dumps_safe({
                        "new_content": content_body,
                        "docs": existing_docs_list,
                    })},
                ],
                cfg.llm.model,
                cfg,
                context="[Split][Plan]",
            )
            logger.debug("[Split] PLAN_UPDATE_PROMPT response: %s", plan_json)

            # If LLM returns empty content, retry
            if not plan_json or not plan_json.strip():
                logger.warning(
                    term_color.yellow(
                        f"[Split] PLAN_UPDATE_PROMPT returned empty string (call attempt {call_attempt}/{max_retries})",
                        enabled=True,
                    )
                )
                if call_attempt < max_retries:
                    time.sleep(call_attempt * cfg.memory.retry_initial_wait)
                    continue
                else:
                    # All call retries failed, raise error
                    logger.error(
                        term_color.red(
                            f"[Split] PLAN_UPDATE_PROMPT LLM returned empty string (retried {max_retries} times)",
                            enabled=True,
                        )
                    )
                    raise RuntimeError(f"[Split] PLAN_UPDATE_PROMPT LLM returned empty string (retried {max_retries} times)")
            else:
                # LLM call succeeded with response content
                break

        # Try to parse the returned JSON
        plan = parse_plan_updates(plan_json)

        # Check if parsed result is valid (has at least new docs or updates)
        if plan.get("new_docs") or plan.get("updates"):
            # Parsing succeeded, return result
            return plan

        # Parsing failed or returned empty result, return empty plan
        logger.warning(
            term_color.yellow(
                f"[Split] PLAN_UPDATE_PROMPT response cannot be parsed as valid JSON, returning empty plan",
                enabled=True,
            )
        )
        return {"new_docs": [], "updates": []}

    def _call_plan_update_prompt_with_fallback(
        self,
        content_body: str,
        existing_docs_list: List[Dict[str, str]],
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
        max_call_retries: int,
        splitting_depth: int,
        max_splitting_depth: int,
    ) -> Dict:
        """Call PLAN_UPDATE_PROMPT, with fallback splitting on failure.

        Args:
            content_body: Content to split
            existing_docs_list: List of existing documents
            cfg: Configuration object
            llm: LLMClient instance
            max_call_retries: Maximum retry count
            splitting_depth: Current splitting depth
            max_splitting_depth: Maximum splitting depth

        Returns the parsed plan dictionary (containing new_docs and updates).
        """
        logger = logging.getLogger("infini_memory_classic")

        # Try normal call first
        plan = self._call_plan_update_prompt(
            content_body, existing_docs_list, cfg, llm, max_call_retries
        )

        # If successfully returned valid result, return directly
        if plan.get("new_docs") or plan.get("updates"):
            return plan

        # Parsing failed or returned empty result, perform fallback splitting
        if splitting_depth >= max_splitting_depth:
            # Reached maximum splitting depth, raise error
            logger.error(
                term_color.red(
                    f"[Split] PLAN_UPDATE_PROMPT cannot be parsed as JSON, reached maximum splitting depth {max_splitting_depth}",
                    enabled=True,
                )
            )
            raise RuntimeError(f"[Split] PLAN_UPDATE_PROMPT cannot be parsed as JSON (reached maximum splitting depth {max_splitting_depth})")

        # Split content and process recursively (fallback splitting)
        logger.warning(
            term_color.yellow(
                f"[Split] PLAN_UPDATE_PROMPT cannot be parsed as JSON, splitting content by newlines into two parts (splitting depth: {splitting_depth + 1}/{max_splitting_depth})",
                enabled=True,
            )
        )

        # Fallback splitting needs to avoid boundary fact duplication, otherwise both segments
        # would be interpreted by the LLM as needing to "create the same document"
        part1, part2 = split_content_by_newlines_non_overlapping(content_body)

        if not part1 or not part2:
            # Content cannot be split (e.g., only one line), raise error
            logger.error(
                term_color.red(
                    f"[Split] Content cannot be split (only one line or empty)",
                    enabled=True,
                )
            )
            raise RuntimeError(f"[Split] PLAN_UPDATE_PROMPT cannot be parsed as JSON, and content cannot be split")

        logger.info(
            term_color.cyan(
                f"[Split] Content split into two parts: part1={count_tokens(part1)} tokens, part2={count_tokens(part2)} tokens",
                enabled=True,
            )
        )

        # Recursively process first part
        logger.info("[Split] Recursively processing first part...")
        plan1 = self._call_plan_update_prompt_with_fallback(
            part1, existing_docs_list, cfg, llm, max_call_retries, splitting_depth + 1, max_splitting_depth
        )

        # Recursively process second part (using existing docs list after first part processing)
        logger.info("[Split] Recursively processing second part...")
        # Update existing_docs_list, add new documents generated from first part
        updated_existing_docs = existing_docs_list.copy()
        for new_doc in plan1.get("new_docs", []):
            updated_existing_docs.append({
                "id": f"temp_{len(updated_existing_docs)}",
                "summary": new_doc.get("title", "")[:200],  # Use title as summary
            })

        plan2 = self._call_plan_update_prompt_with_fallback(
            part2, updated_existing_docs, cfg, llm, max_call_retries, splitting_depth + 1, max_splitting_depth
        )

        # Merge results from both parts (using unified merge method)
        merged_plan = self._merge_split_plans(plan1, plan2)

        logger.info(
            term_color.green(
                f"[Split] Fallback splitting merge succeeded: total {len(merged_plan['new_docs'])} new documents, {len(merged_plan['updates'])} updates",
                enabled=True,
            )
        )

        return merged_plan

    def _merge_split_plans(self, plan1: Dict, plan2: Dict) -> Dict:
        """Merge two split plan results.

        Merge rules:
        - updates: For same id, concatenate new_content together
        - new_docs: Directly merge arrays

        Args:
            plan1: First split plan
            plan2: Second split plan

        Returns the merged plan dictionary.
        """
        logger = logging.getLogger("infini_memory_classic")

        # Merge updates: For same id, concatenate new_content together
        updates1 = plan1.get("updates", [])
        updates2 = plan2.get("updates", [])

        # Use dictionary to merge updates (key is id)
        updates_dict = {}
        for update in updates1:
            doc_id = update.get("id")
            if doc_id:
                updates_dict[doc_id] = update.get("new_content", "")

        for update in updates2:
            doc_id = update.get("id")
            if doc_id:
                if doc_id in updates_dict:
                    # id already exists, concatenate new_content
                    updates_dict[doc_id] += "\n\n" + update.get("new_content", "")
                else:
                    # id does not exist, add directly
                    updates_dict[doc_id] = update.get("new_content", "")

        # Convert back to list format
        merged_updates = [{"id": doc_id, "new_content": content} for doc_id, content in updates_dict.items()]

        # Merge new_docs: directly merge arrays
        new_docs1 = plan1.get("new_docs", [])
        new_docs2 = plan2.get("new_docs", [])
        merged_new_docs = new_docs1 + new_docs2

        merged_plan = {
            "new_docs": merged_new_docs,
            "updates": merged_updates,
        }

        logger.debug(
            f"[Merge] plan1: {len(new_docs1)} new_docs, {len(updates1)} updates | "
            f"plan2: {len(new_docs2)} new_docs, {len(updates2)} updates | "
            f"merged: {len(merged_new_docs)} new_docs, {len(merged_updates)} updates"
        )

        return merged_plan

    def _merge_multiple_plans(self, plans: List[Dict]) -> Dict:
        """Merge multiple split plan results.

        Merge rules:
        - updates: For same id, concatenate new_content together
        - new_docs: Directly merge arrays

        Args:
            plans: List of split plans

        Returns the merged plan dictionary.
        """
        if not plans:
            return {"new_docs": [], "updates": []}

        if len(plans) == 1:
            return plans[0]

        # Use pairwise merge method to merge multiple plans
        merged = plans[0]
        for plan in plans[1:]:
            merged = self._merge_split_plans(merged, plan)

        return merged

    def _process_split_plan(
        self,
        plan: Dict,
        existing_gen1_docs: List[DocMeta],
        existing_docs_list: List[Dict[str, str]],
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
        mm: MemoryManager,
        current_epoch: int,
        rewrite_source: Optional[str] = None,
    ) -> List[DocMeta]:
        """Process split plan, creating new documents and updating existing documents.

        Returns a list of all processed document metadata.
        """
        logger = logging.getLogger("infini_memory_classic")

        merged_metas = []  # Final merged document metadata
        created_docs = []   # Newly created documents (before merge)

        rewrite_threads = cfg.memory.rewrite_doc_threads
        new_docs_list = plan.get("new_docs", [])
        updates_list = plan.get("updates", [])

        # Convert updates with non-existent target_id to new_docs, avoiding update failures
        # caused by LLM fabricating/truncating ids.
        allowed_ids = {str(d.get("id")) for d in existing_docs_list if d.get("id")}
        filtered_updates: List[Dict[str, Any]] = []
        for u in updates_list:
            uid = str(u.get("id") or "").strip()
            ucontent = str(u.get("new_content") or "").strip()
            if not uid or uid not in allowed_ids:
                if ucontent:
                    new_docs_list.append({"title": "", "content": ucontent})
                continue
            filtered_updates.append({"id": uid, "new_content": ucontent})
        updates_list = filtered_updates

        # Process new documents (new_docs list returned by PLAN_UPDATE_PROMPT)
        logger.info("[Split] Starting to process new documents, total %d, using %d threads", len(new_docs_list), rewrite_threads)
        if rewrite_threads > 1 and len(new_docs_list) > 1:
            # Multi-threaded processing of new documents
            with ThreadPoolExecutor(max_workers=rewrite_threads, thread_name_prefix="RewriteNew") as executor:
                futures = {}
                for idx_new, new_doc in enumerate(new_docs_list):
                    future = executor.submit(
                        _process_new_doc,
                        new_doc,
                        llm,
                        cfg,
                        mm,
                        current_epoch,
                        idx_new,
                        len(new_docs_list),
                        rewrite_source,
                    )
                    futures[future] = (idx_new, new_doc.get("title", ""))

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            merged_metas.append(result)
                            created_docs.append(result)
                    except Exception as e:
                        idx_new, title = futures[future]
                        logger.error("[Split] New document processing exception: idx=%d, title=%s, error=%s", idx_new, title, e)
        else:
            # Single-threaded processing of new documents
            for idx_new, new_doc in enumerate(new_docs_list):
                result = _process_new_doc(
                    new_doc, llm, cfg, mm, current_epoch,
                    idx_new, len(new_docs_list), rewrite_source
                )
                if result:
                    merged_metas.append(result)
                    created_docs.append(result)

        # Process update plans
        logger.info("[Split] Starting to process update plans, total %d, using %d threads", len(updates_list), rewrite_threads)
        if rewrite_threads > 1 and len(updates_list) > 1:
            # Multi-threaded processing of updates
            with ThreadPoolExecutor(max_workers=rewrite_threads, thread_name_prefix="RewriteUpdate") as executor:
                futures = {}
                for idx_update, update in enumerate(updates_list):
                    future = executor.submit(
                        _process_update,
                        update,
                        llm,
                        cfg,
                        mm,
                        existing_docs_list,
                        idx_update,
                        len(updates_list),
                    )
                    futures[future] = (idx_update, update.get("id", ""))

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            merged_metas.append(result)
                    except Exception as e:
                        idx_update, target_id = futures[future]
                        logger.error("[Split] Update processing exception: idx=%d, target_id=%s, error=%s", idx_update, target_id, e)
        else:
            # Single-threaded processing of updates
            for idx_update, update in enumerate(updates_list):
                result = _process_update(
                    update, llm, cfg, mm, existing_docs_list,
                    idx_update, len(updates_list)
                )
                if result:
                    merged_metas.append(result)

        # Calculate merge statistics
        merged_to_existing = len([m for m in merged_metas if m.id not in [d.id for d in created_docs]])
        new_doc_count = len(created_docs)

        logger.info("[Split] Processing completed statistics: new_docs=%d, updates_merged=%d, total_docs=%d",
                    new_doc_count, merged_to_existing, len(merged_metas))

        return merged_metas

    def _clear_current_doc(self, current_epoch: int, mm: MemoryManager, user_id: str) -> None:
        """Clear CURRENT document and increment current_epoch.

        Note: If CURRENT document does not exist (e.g., in SPLIT_MERGE mode), skip the clear operation.
        """
        logger = logging.getLogger("infini_memory_classic")

        # Check if CURRENT document exists
        docs = mm.list_docs()
        current_doc = next((d for d in docs if d.id == "CURRENT"), None)

        if current_doc is None:
            logger.info(
                term_color.yellow(
                    f"[Split] CURRENT document does not exist (user_id={user_id}), skipping clear operation",
                    enabled=True,
                )
            )
            return

        # Increment current_epoch before clearing CURRENT document
        next_epoch = current_epoch + 1
        # Use thread-safe method to update current_epoch
        mm.update_doc_in_index("CURRENT", lambda d: d.update({"current_epoch": next_epoch}))

        mm.clear_doc("CURRENT")
        logger.info(
            term_color.yellow(
                f"[Split] CURRENT document cleared (user_id={user_id}, current_epoch: {current_epoch} -> {next_epoch}), awaiting new content",
                enabled=True,
            )
        )

    def _merge_docs(
        self,
        mm: MemoryManager,
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
    ) -> None:
        """Select and merge documents with similar content (only merge documents with tokens below merge_max_tokens).

        Note: This method may be called concurrently in multi-threaded environments, need to ensure each thread
        only processes documents created by its split operation.
        """
        logger = logging.getLogger("infini_memory_classic")

        logger.info(
            term_color.cyan(
                f"[MERGE] Starting merge phase, checking for documents with similar content to merge",
                enabled=True,
            )
        )

        # Get all documents (excluding CURRENT), only merge documents with tokens below merge_max_tokens
        all_docs = mm.list_docs()
        MERGE_candidates = [
            d for d in all_docs
            if d.id != "CURRENT" and d.tokens < cfg.memory.merge_max_tokens
        ]

        # Log after filtering ineligible documents
        filtered_out = [
            d for d in all_docs
            if d.id != "CURRENT" and d.tokens >= cfg.memory.merge_max_tokens
        ]
        if filtered_out:
            logger.info(
                term_color.cyan(
                    f"[MERGE] Filtered out {len(filtered_out)} documents (tokens >= {cfg.memory.merge_max_tokens})",
                    enabled=True,
                )
            )

        # Get current thread info to identify if in multi-threaded environment
        import threading
        thread_name = threading.current_thread().name
        is_multi_thread = "SplitMergeWorker" in thread_name or "RewriteFileWorker" in thread_name

        # If in multi-threaded environment (SPLIT_MERGE phase with --thread parameter), use thread-safe processing
        if is_multi_thread:
            logger.info(
                term_color.cyan(
                    f"[MERGE] Multi-threaded environment detected (thread={thread_name}), will use serial processing to avoid content duplication",
                    enabled=True,
                )
            )
        else:
            logger.info(
                term_color.cyan(
                    f"[MERGE] Single-threaded environment (thread={thread_name})",
                    enabled=True,
                )
            )

        if len(MERGE_candidates) >= 2:
            # Call SELECT_MERGE_GROUPS_PROMPT to select document groups for merging
            try:
                docs_list_for_MERGE = [{
                    "id": d.id,
                    "summary": d.summary,
                    "updated_at": d.updated_at,
                } for d in MERGE_candidates]

                logger.debug("[MERGE] Calling SELECT_MERGE_GROUPS_PROMPT, document count=%d", len(docs_list_for_MERGE))

                MERGE_groups_json = llm.chat([
                    {"role": "system", "content": SELECT_MERGE_GROUPS_PROMPT},
                    {"role": "user", "content": json_dumps_safe({"docs": docs_list_for_MERGE})},
                ], model=cfg.llm.model, temperature=cfg.llm.temperature)

                logger.debug("[MERGE] SELECT_MERGE_GROUPS_PROMPT response: %s", MERGE_groups_json)

                # Parse the returned group info (using robust parsing function)
                MERGE_groups_data = parse_merge_groups(MERGE_groups_json)
                MERGE_groups = MERGE_groups_data.get("groups", [])

                # If parsing failed (returned empty groups), log warning
                if not MERGE_groups and MERGE_groups_json.strip():
                    logger.warning(
                        term_color.yellow(
                            "[MERGE] Failed to parse group info or returned empty groups, LLM may have returned invalid JSON format",
                            enabled=True,
                        )
                    )

                if MERGE_groups:
                    logger.info(
                        term_color.cyan(
                            f"[MERGE] Selected {len(MERGE_groups)} document groups for merging",
                            enabled=True,
                        )
                    )

                    # Use multi-threading for merge processing
                    MERGE_threads = cfg.memory.rewrite_doc_threads

                    # Important: If in multi-threaded environment (SPLIT_MERGE phase with --thread parameter),
                    # force single-threaded merge processing to avoid different threads reading same documents causing content duplication
                    if is_multi_thread:
                        MERGE_threads = 1
                        logger.info(
                            term_color.cyan(
                                "[MERGE] Multi-threaded environment forces single-threaded merge (to avoid content duplication)",
                                enabled=True,
                            )
                        )

                    logger.info("[MERGE] Starting merge processing, total %d groups, using %d threads", len(MERGE_groups), MERGE_threads)

                    MERGEd_doc_ids = set()  # Record merged document IDs
                    MERGEd_doc_tokens_before = 0  # Total tokens before merge
                    MERGEd_doc_tokens_after = 0   # Total tokens after merge

                    if MERGE_threads > 1 and len(MERGE_groups) > 1:
                        # Multi-threaded merge processing
                        with ThreadPoolExecutor(max_workers=MERGE_threads, thread_name_prefix="MERGEWorker") as executor:
                            futures = {}
                            for idx_group, group in enumerate(MERGE_groups):
                                future = executor.submit(
                                    _process_MERGE_group,
                                    group,
                                    MERGE_candidates,
                                    llm,
                                    cfg,
                                    mm,
                                    idx_group,
                                    len(MERGE_groups),
                                )
                                futures[future] = (idx_group, group.get("doc_ids", []))

                            # Collect results
                            for future in as_completed(futures):
                                try:
                                    result = future.result()
                                    if result:
                                        # Record merged document IDs and token statistics
                                        MERGEd_doc_ids.update(result.get("MERGEd_doc_ids", []))
                                        MERGEd_doc_tokens_before += result.get("tokens_before", 0)
                                        MERGEd_doc_tokens_after += result.get("tokens_after", 0)
                                except Exception as e:
                                    idx_group, doc_ids = futures[future]
                                    logger.error("[MERGE] Merge group processing exception: idx=%d, doc_ids=%s, error=%s", idx_group, doc_ids, e)
                    else:
                        # Single-threaded merge processing
                        for idx_group, group in enumerate(MERGE_groups):
                            result = _process_MERGE_group(
                                group, MERGE_candidates, llm, cfg, mm,
                                idx_group, len(MERGE_groups)
                            )
                            if result:
                                MERGEd_doc_ids.update(result.get("MERGEd_doc_ids", []))
                                MERGEd_doc_tokens_before += result.get("tokens_before", 0)
                                MERGEd_doc_tokens_after += result.get("tokens_after", 0)

                    # Delete old merged documents
                    if MERGEd_doc_ids:
                        logger.info(
                            term_color.cyan(
                                f"[MERGE] Deleting {len(MERGEd_doc_ids)} old merged documents",
                                enabled=True,
                            )
                        )
                        # Delete document files first (no lock needed, each file path is unique)
                        idx = mm._load_index()
                        for d in idx.get("docs", []):
                            if d.get("id") in MERGEd_doc_ids:
                                try:
                                    doc_rel_path = d.get("path", "")
                                    if mm.file_exists(doc_rel_path):
                                        mm.delete_file(doc_rel_path)
                                except Exception as e:
                                    logger.warning("[MERGE] Failed to delete document file: id=%s, error=%s", d.get("id"), e)
                        # Use thread-safe method to remove documents from index
                        mm.remove_docs_from_index(MERGEd_doc_ids)

                        # Output post-merge document statistics
                        final_docs = mm.list_docs()
                        final_docs_excluding_current = [d for d in final_docs if d.id != "CURRENT"]

                        # Calculate token change
                        tokens_saved = MERGEd_doc_tokens_before - MERGEd_doc_tokens_after
                        tokens_percent = (tokens_saved / MERGEd_doc_tokens_before * 100) if MERGEd_doc_tokens_before > 0 else 0

                        # Output merge statistics (using consistent base)
                        new_merged_count = len(MERGE_groups)
                        net_change = len(MERGEd_doc_ids) - new_merged_count
                        logger.info(
                            term_color.cyan(
                                f"[MERGE] Merge completed: deleted {len(MERGEd_doc_ids)} old documents, created {new_merged_count} merged documents (net reduction {net_change})",
                                enabled=True,
                            )
                        )
                        logger.info(
                            term_color.cyan(
                                f"[MERGE] Token statistics: before merge {MERGEd_doc_tokens_before:,} -> after merge {MERGEd_doc_tokens_after:,} (saved {tokens_saved:,} tokens, {tokens_percent:.1f}%)",
                                enabled=True,
                            )
                        )
                else:
                    logger.info("[MERGE] No document groups selected for merging")

            except Exception as e:
                logger.warning("[MERGE] Merge phase execution exception: %s, skipping merge", e)
                logger.exception("[MERGE] Merge phase exception details")
        else:
            logger.info("[MERGE] Document count less than 2, skipping merge phase")

    def _split_current(
        self,
        final_content: str,
        current_epoch: int,
        user_id: str,
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
        mm: MemoryManager,
        rewrite_source: Optional[str] = None,
    ) -> None:
        """Split CURRENT document and execute merge by frequency.

        Args:
            final_content: Content to split (already rewritten content in SPLIT_MERGE mode)
            current_epoch: Current epoch number
            user_id: User ID
            cfg: Configuration object
            llm: LLMClient instance
            mm: MemoryManager instance
            rewrite_source: Source filename from rewrite directory (e.g., REWRITE_CURRENT_1.md)
        """
        logger = logging.getLogger("infini_memory_classic")

        # Get current split count (inferred from existing document count, excluding CURRENT)
        docs = mm.list_docs()
        existing_docs = [d for d in docs if d.id != "CURRENT"]
        split_count = len(existing_docs)

        logger.info(
            term_color.yellow(
                f"[Split] Triggering split: CURRENT document needs splitting",
                enabled=True,
            )
        )

        # Extract content body (remove possible YAML Frontmatter)
        _, content_body = extract_summary_from_markdown(final_content)
        content_tokens = count_tokens(content_body)

        logger.info(
            term_color.yellow(
                f"[Split] Starting split, content tokens: {content_tokens:,}",
                enabled=True,
            )
        )

        # Use content to call PLAN_UPDATE_PROMPT to generate split plan
        # Build existing document list (excluding CURRENT, empty documents, and documents exceeding markdown_length)
        original_doc_count = len(existing_docs)
        existing_docs_list = [
            {"id": d.id, "summary": d.summary}
            for d in existing_docs
            if d.tokens <= cfg.memory.markdown_length
        ]
        filtered_count = original_doc_count - len(existing_docs_list)
        if filtered_count > 0:
            logger.info(
                term_color.yellow(
                    f"[Split] Filtered out {filtered_count} documents exceeding markdown_length ({cfg.memory.markdown_length})",
                    enabled=True,
                )
            )

        # Call PLAN_UPDATE_PROMPT with retry mechanism
        import time
        plan_start_time = time.time()
        logger.debug("[Split] Calling PLAN_UPDATE_PROMPT: current_epoch=%d, existing_docs=%d",
                    current_epoch, len(existing_docs_list))

        plan = self._get_split_plan(content_body, existing_docs_list, cfg, llm)

        plan_elapsed = time.time() - plan_start_time
        logger.info(
            term_color.yellow(
                f"[Split] PLAN_UPDATE_PROMPT completed, elapsed: {plan_elapsed:.1f} seconds",
                enabled=True,
            )
        )

        logger.info(
            term_color.yellow(
                f"[Split] CURRENT round {current_epoch + 1} split (user_id={user_id}): planning to update {len(plan['updates'])} existing documents, create {len(plan['new_docs'])} new documents",
                enabled=True,
            )
        )

        # Process split plan
        process_start_time = time.time()
        merged_metas = self._process_split_plan(
            plan, existing_docs, existing_docs_list, cfg, llm, mm, current_epoch, rewrite_source
        )
        process_elapsed = time.time() - process_start_time

        # Calculate merge statistics
        created_doc_ids = set()
        merged_to_existing_count = 0
        for m in merged_metas:
            # Check if this is a newly created document (parent_id is "CURRENT")
            if m.parent_id == "CURRENT":
                created_doc_ids.add(m.id)
            else:
                merged_to_existing_count += 1
        new_doc_count = len(created_doc_ids)

        logger.info(
            term_color.yellow(
                f"[Split] CURRENT split #{split_count + 1} completed: created {new_doc_count} new documents, "
                f"{merged_to_existing_count} merged with existing documents, "
                f"total {len(set(m.id for m in merged_metas))} documents after merge, "
                f"split plan processing elapsed: {process_elapsed:.1f} seconds",
                enabled=True,
            )
        )

        # Clear CURRENT document
        self._clear_current_doc(current_epoch, mm, user_id)

        # Increment split count
        current_split_count = mm.increment_split_count()

        # Execute merge phase by frequency, or when small document count exceeds threshold
        should_merge = False
        merge_reason = ""

        if not cfg.memory.merge_enabled:
            merge_reason = "Merge feature not enabled"
        else:
            # Check if merge frequency is reached
            if current_split_count % cfg.memory.merge_frequency == 0:
                should_merge = True
                merge_reason = f"Reached merge frequency (every {cfg.memory.merge_frequency} times)"
            else:
                # Check if small document count exceeds threshold
                all_docs = mm.list_docs()
                small_docs = [d for d in all_docs if d.id != "CURRENT" and d.tokens < cfg.memory.merge_max_tokens]
                if len(small_docs) > cfg.memory.merge_trigger_min_count:
                    should_merge = True
                    merge_reason = f"Small document count ({len(small_docs)}) exceeds threshold ({cfg.memory.merge_trigger_min_count})"

        if should_merge:
            logger.info(
                term_color.cyan(
                    f"[Split] SPLIT #{current_split_count}, {merge_reason}, executing merge",
                    enabled=True,
                )
            )
            self._merge_docs(mm, cfg, llm)
            # SPLIT_MERGE mode: skip post-merge split (oversized docs handled in eval_add.py)
            if rewrite_source is None:
                # After merge, check all documents and split those exceeding markdown_length
                self._split_oversized_docs(mm, cfg, llm, user_id)
            else:
                logger.info(
                    term_color.cyan(
                        f"[Split] Skipping post-merge split in SPLIT_MERGE mode (source: {rewrite_source}), oversized documents will be handled by eval_add.py",
                        enabled=True,
                    )
                )
        else:
            logger.info(
                term_color.cyan(
                    f"[Split] SPLIT #{current_split_count}, merge frequency not reached (every {cfg.memory.merge_frequency} times), skipping merge",
                    enabled=True,
                )
            )

        # Auto-generate skills after split
        if (cfg.memory.skills_enabled
                and cfg.memory.skills_auto_generate
                and current_split_count % cfg.memory.skills_generate_frequency == 0):
            try:
                from .skills import SkillsManager, generate_skills_from_memory
                sm = SkillsManager(
                    root=cfg.root,
                    data_root=cfg.memory.data_root,
                    store=mm.store,
                    user_id=user_id,
                    storage=cfg.get_storage_backend(),
                    skills_dir=cfg.memory.skills_dir,
                )
                generate_skills_from_memory(
                    docs=mm.list_docs(),
                    existing_skills=sm.list_skills(),
                    cfg=cfg, llm=llm, mm=mm, sm=sm,
                    trigger="after_split",
                )
            except Exception as e:
                logger.warning("[Split] Auto skill generation failed: %s", e)

    def _split_oversized_docs(
        self,
        mm: MemoryManager,
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
        user_id: str,
    ) -> None:
        """Check all documents and split those exceeding markdown_length."""
        logger = logging.getLogger("infini_memory_classic")

        all_docs = mm.list_docs()
        oversized_docs = [d for d in all_docs if d.id != "CURRENT" and d.tokens > int(cfg.memory.markdown_length)]

        if oversized_docs:
            logger.info(
                term_color.cyan(
                    f"[Split] Found {len(oversized_docs)} oversized documents that need splitting",
                    enabled=True,
                )
            )

            for doc in oversized_docs:
                logger.info(
                    term_color.cyan(
                        f"[Split] Processing oversized document: id={doc.id}, tokens={doc.tokens}",
                        enabled=True,
                    )
                )
                # Read document content
                doc_content = mm.read_file(doc.path)
                # Remove YAML Frontmatter
                _, content_body = extract_summary_from_markdown(doc_content)

                # Get existing document list (excluding CURRENT, the oversized document itself, and documents exceeding markdown_length)
                all_other_docs = [d for d in all_docs if d.id != "CURRENT" and d.id != doc.id]
                original_doc_count = len(all_other_docs)
                existing_docs = [d for d in all_other_docs if d.tokens <= cfg.memory.markdown_length]
                filtered_count = original_doc_count - len(existing_docs)
                if filtered_count > 0:
                    logger.info(
                        term_color.yellow(
                            f"[Split] Filtered out {filtered_count} documents exceeding markdown_length ({cfg.memory.markdown_length})",
                            enabled=True,
                        )
                    )
                existing_docs_list = [{"id": d.id, "summary": d.summary} for d in existing_docs]

                # Call PLAN_UPDATE_PROMPT to get split plan
                try:
                    plan = self._get_split_plan(content_body, existing_docs_list, cfg, llm)
                except Exception as e:
                    logger.error(f"[Split] Failed to process oversized document {doc.id}: {e}")
                    continue

                # Get document's current_epoch
                current_epoch = getattr(doc, "current_epoch", 0)

                # Process split plan
                merged_metas = self._process_split_plan(
                    plan, existing_docs, existing_docs_list, cfg, llm, mm, current_epoch
                )

                logger.info(
                    term_color.cyan(
                        f"[Split] Oversized document {doc.id} split completed: created {len(merged_metas)} documents",
                        enabled=True,
                    )
                )

                # Delete original oversized document
                try:
                    if mm.file_exists(doc.path):
                        mm.delete_file(doc.path)
                    # Use thread-safe method to remove document from index
                    mm.remove_docs_from_index({doc.id})
                    logger.info(f"[Split] Deleted oversized document: {doc.id}")
                except Exception as e:
                    logger.error(f"[Split] Failed to delete oversized document {doc.id}: {e}")

                # Update all_docs list
                all_docs = mm.list_docs()

    def add(
        self,
        messages: Any,
        *,
        store: str,
        user_id: str,
        cfg: Optional[InfiniMemoryConfig] = None,
        llm: Optional[LLMClient] = None,
        stage: Literal["FULL", "CURRENT", "SPLIT_MERGE"] = "FULL",
        seq: int = 1,
        doc_id: str = "CURRENT",
        seq_start: int = 1,
        source_file: Optional[str] = None,
        infer: bool = True,
    ) -> int:
        """Add content and immediately update to file system.

        New logic: Always write and update in a document with id 'CURRENT'.
        When CURRENT document length exceeds markdown_length, use PLAN_UPDATE_PROMPT to split and merge,
        then clear the CURRENT document.

        - messages: Any object; if list, process each item; otherwise convert to string for single processing.
        - user_id: User ID (required), used to isolate document storage paths for different users (data_root/<user_id>/doc).
        - cfg: Configuration object; memory must be enabled for disk write operations.
        - llm: Optional custom LLMClient.
        - stage: Execution stage, options:
            - "FULL" (default): Complete execution flow (extract -> write CURRENT -> split -> merge)
            - "CURRENT": Only extract info and write to CURRENT document, no SPLIT/MERGE, rename to CURRENT_<seq> in raw directory when exceeding length
            - "SPLIT_MERGE": Directly execute SPLIT/MERGE logic, messages should be content from raw/CURRENT_<seq> documents
        - seq: Current sequence number (managed and incremented by caller)
        - doc_id: Document ID to use (defaults to "CURRENT"), can be "CURRENT_THREAD_<n>" in multi-threaded mode
        - seq_start: Thread's seq starting number (defaults to 1), used to calculate raw file sequence range
        - source_file: Source filename (optional), for logging (e.g., "CURRENT_46.md")
        Returns the number of processed items.
        """
        logger = logging.getLogger("infini_memory_classic")
        # Colored start log
        stage_info = f" (stage={stage})" if stage != "FULL" else ""
        logger.info(term_color.green(f"--add--{stage_info}", enabled=True))

        processed = 0
        if not cfg or not getattr(cfg, "memory", None) or not cfg.memory.enabled:
            return processed

        try:
            llm = llm or LLMClient(
                api_key=cfg.llm.openai_api_key,
                base_url=cfg.llm.openai_base_url,
                retry_max_attempts=cfg.llm.retry_max_attempts,
                retry_initial_wait=cfg.llm.retry_initial_wait,
                retry_max_wait=cfg.llm.retry_max_wait,
                retry_jitter=cfg.llm.retry_jitter,
            )
            mm = MemoryManager(
                root=cfg.root,
                data_root=cfg.memory.data_root,
                doc_dir=cfg.memory.doc_dir,
                meta_dir=cfg.memory.metadata_dir,
                index_file=cfg.memory.index_file,
                store=store,
                user_id=user_id,
                storage=cfg.get_storage_backend(),
            )
        except Exception as e:
            logger.exception("add() initialization exception: %s", e)
            logger.info("Processed add items: %s", processed)
            return processed

        # ============== REWRITE_CURRENT mode ==============
        if stage == "REWRITE_CURRENT":
            # messages should be the content of raw/CURRENT_<seq> documents
            # Rewrite content by topic aggregation, write to rewrite/REWRITE_CURRENT_<seq>.md
            try:
                if not messages or not str(messages).strip():
                    logger.error("REWRITE_CURRENT mode requires non-empty messages")
                    return 0

                raw_content = str(messages)
                source_info = f" (source: {source_file})" if source_file else ""
                logger.info(
                    term_color.yellow(
                        f"[REWRITE_CURRENT] Received raw content, raw length: {len(raw_content)} chars{source_info}",
                        enabled=True,
                    )
                )

                # Extract body (remove possible YAML Frontmatter)
                _, content_body = extract_summary_from_markdown(raw_content)
                raw_tokens = count_tokens(content_body)
                logger.info(
                    term_color.yellow(
                        f"[REWRITE_CURRENT] Body extraction completed, raw tokens: {raw_tokens:,}{source_info}",
                        enabled=True,
                    )
                )

                # Call REWRITE_CURRENT_PROMPT to rewrite content by topic aggregation
                import time
                rewrite_start_time = time.time()
                logger.info(
                    term_color.yellow(
                        f"[REWRITE_CURRENT] ====== Starting document rewrite (by topic aggregation) ======{source_info}",
                        enabled=True,
                    )
                )

                # Output raw content preview with source filename
                source_preview_label = source_file if source_file else "CURRENT"
                logger.info(
                    term_color.cyan(
                        f"[REWRITE_CURRENT] Raw content preview (source: {source_preview_label}, first 500 chars):\n{content_body[:500]}",
                        enabled=True,
                    )
                )

                rewrite_current_prompt = REWRITE_CURRENT_PROMPT.format(
                    summary_length=cfg.memory.summary_length,
                    current_content=content_body,
                )

                # Calculate prompt token count
                prompt_tokens = count_tokens(rewrite_current_prompt)
                logger.info(
                    term_color.cyan(
                        f"[REWRITE_CURRENT] REWRITE_CURRENT_PROMPT length: {prompt_tokens:,} tokens",
                        enabled=True,
                    )
                )

                logger.info(
                    term_color.cyan(
                        f"[REWRITE_CURRENT] Preparing to call LLM for rewriting...",
                        enabled=True,
                    )
                )

                rewritten_content = _llm_chat_with_retry(
                    llm,
                    [{"role": "system", "content": rewrite_current_prompt}],
                    cfg.llm.model,
                    cfg,
                    context="[REWRITE_CURRENT][RewriteCurrent]",
                )

                rewrite_elapsed = time.time() - rewrite_start_time
                logger.info(
                    term_color.cyan(
                        f"[REWRITE_CURRENT] LLM returned, result length: {len(rewritten_content)} chars, elapsed: {rewrite_elapsed:.1f} seconds",
                        enabled=True,
                    )
                )

                # Extract rewritten body (remove YAML Frontmatter)
                _, rewritten_body = extract_summary_from_markdown(rewritten_content)

                # Calculate rewritten token count
                rewritten_tokens = count_tokens(rewritten_body)
                tokens_delta = rewritten_tokens - raw_tokens
                delta_percent = (tokens_delta / raw_tokens * 100) if raw_tokens > 0 else 0

                logger.info(
                    term_color.yellow(
                        f"[REWRITE_CURRENT] ====== Document rewrite completed ======{source_info}",
                        enabled=True,
                    )
                )
                logger.info(
                    term_color.yellow(
                        f"[REWRITE_CURRENT] Token change: {raw_tokens:,} -> {rewritten_tokens:,} "
                        f"({tokens_delta:+d}, {delta_percent:+.1f}%)",
                        enabled=True,
                    )
                )

                # Save rewritten content to rewrite/REWRITE_CURRENT_<seq>.md
                rewrite_dir = mm.data_dir / "rewrite"
                rewrite_dir_rel = mm._rel(rewrite_dir)
                mm.storage.mkdir(rewrite_dir_rel)

                # Extract sequence number from source filename, e.g., "CURRENT_46.md" -> "46"
                if source_file and source_file.startswith("CURRENT_"):
                    current_number = source_file.split("_")[1].split(".")[0]
                    rewrite_filename = f"REWRITE_CURRENT_{current_number}.md"
                else:
                    # If no source filename, use timestamp
                    from datetime import datetime
                    from zoneinfo import ZoneInfo
                    ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
                    rewrite_filename = f"REWRITE_CURRENT_{ts}.md"

                rewrite_path = rewrite_dir / rewrite_filename
                rewrite_rel = mm._rel(rewrite_path)
                mm.write_file(rewrite_rel, rewritten_body)
                logger.info(
                    term_color.cyan(
                        f"[REWRITE_CURRENT] Saved rewritten content to: {rewrite_rel} ({rewritten_tokens:,} tokens)",
                        enabled=True,
                    )
                )
                processed = 1
                logger.info(
                    term_color.yellow(
                        f"[REWRITE_CURRENT] ====== Processing completed ======",
                        enabled=True,
                    )
                )
            except Exception as e:
                logger.exception("REWRITE_CURRENT mode processing exception: %s", e)
                raise
            return processed

        # ============== SPLIT_MERGE mode ==============
        if stage == "SPLIT_MERGE":
            # messages should be content from rewrite/REWRITE_CURRENT_<seq>.md documents (already rewritten)
            # Directly execute SPLIT/MERGE logic
            try:
                if not messages or not str(messages).strip():
                    logger.error("SPLIT_MERGE mode requires non-empty messages")
                    return 0

                rewritten_content = str(messages)
                source_info = f" (source: {source_file})" if source_file else ""
                logger.info(
                    term_color.yellow(
                        f"[SPLIT_MERGE] Received rewritten content, length: {len(rewritten_content)} chars{source_info}",
                        enabled=True,
                    )
                )

                # Extract body (remove possible YAML Frontmatter)
                _, content_body = extract_summary_from_markdown(rewritten_content)
                rewritten_tokens = count_tokens(content_body)
                logger.info(
                    term_color.yellow(
                        f"[SPLIT_MERGE] Body extraction completed, tokens: {rewritten_tokens:,}{source_info}",
                        enabled=True,
                    )
                )

                # Directly use rewritten content for splitting
                logger.info(
                    term_color.yellow(
                        f"[SPLIT_MERGE] ====== Starting split phase ======",
                        enabled=True,
                    )
                )

                # Get current epoch
                docs = mm.list_docs()
                current_epoch = 0
                for d in docs:
                    if d.id == "CURRENT":
                        current_epoch = getattr(d, "current_epoch", 0)
                        break

                logger.info(
                    term_color.cyan(
                        f"[SPLIT_MERGE] Current epoch: {current_epoch}",
                        enabled=True,
                    )
                )

                # Call split logic (pass in rewritten content)
                logger.info(
                    term_color.cyan(
                        f"[SPLIT_MERGE] Calling _split_current method...",
                        enabled=True,
                    )
                )
                self._split_current(content_body, current_epoch, user_id, cfg, llm, mm, source_file)
                processed = 1
                logger.info(
                    term_color.yellow(
                        f"[SPLIT_MERGE] ====== Processing completed ======",
                        enabled=True,
                    )
                )
            except Exception as e:
                logger.exception("SPLIT_MERGE mode processing exception: %s", e)
                raise
            return processed

        # ============== CURRENT or FULL mode ==============
        # Process messages as text list
        texts: List[str] = []
        if isinstance(messages, (list, tuple)):
            for m in messages:
                t = self._to_text(m)
                if t:
                    texts.append(t)
        else:
            t = self._to_text(messages)
            if t:
                texts.append(t)

        for user_text in texts:
            try:
                # Check if text is empty or contains only whitespace
                if not user_text or not user_text.strip():
                    logger.debug("Skipping empty or whitespace-only text")
                    continue

                # Check LLM configuration (only needed when infer=True)
                if infer and not cfg.llm.openai_api_key:
                    logger.error("LLM API key not configured, cannot process data")
                    logger.info("Processed add items: %s", processed)
                    return processed
                if infer and not cfg.llm.model:
                    logger.error("LLM model not configured, cannot process data")
                    logger.info("Processed add items: %s", processed)
                    return processed

                # Extract Markdown (or skip if infer=False)
                if infer:
                    extract_md = self._extract_markdown(user_text, cfg, llm, mm, seq=seq)
                else:
                    extract_md = "# raw memory content\n\n" + user_text

                # Find or create CURRENT document (using doc_id parameter)
                docs = mm.list_docs()
                current_doc = next((d for d in docs if d.id == doc_id), None)

                # Get current current_epoch (from existing CURRENT document or default 0)
                current_epoch = getattr(current_doc, "current_epoch", 0) if current_doc else 0

                if current_doc is None:
                    # First run: create CURRENT document (using doc_id)
                    self._create_current_doc(extract_md, current_epoch, cfg, mm, doc_id)
                    processed += 1
                    continue

                # ============== Staleness check: CURRENT doc not updated for too long ==============
                if (
                    current_doc.tokens > 0
                    and cfg.memory.current_stale_seconds > 0
                    and stage in ("FULL", "CURRENT")
                ):
                    from datetime import datetime
                    from zoneinfo import ZoneInfo
                    try:
                        last_updated = datetime.fromisoformat(current_doc.updated_at)
                        now_dt = datetime.now(ZoneInfo("Asia/Shanghai"))
                        elapsed = (now_dt - last_updated).total_seconds()
                        if elapsed >= cfg.memory.current_stale_seconds:
                            logger.info(
                                term_color.yellow(
                                    f"[Stale] CURRENT document not updated for {elapsed:.0f}s "
                                    f"(threshold: {cfg.memory.current_stale_seconds}s), "
                                    f"archiving before new write",
                                    enabled=True,
                                )
                            )
                            stale_content = mm.read_file(current_doc.path)
                            if stage == "FULL":
                                self._split_current(stale_content, current_epoch, user_id, cfg, llm, mm)
                            else:
                                mm.save_to_raw(stale_content)
                                mm.clear_doc(doc_id)
                            # Refresh after archiving (CURRENT is now empty)
                            docs = mm.list_docs()
                            current_doc = next((d for d in docs if d.id == doc_id), None)
                            current_epoch = getattr(current_doc, "current_epoch", 0) if current_doc else 0
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[Stale] Failed to parse updated_at for staleness check: {e}")

                # Append content to CURRENT document
                meta = self._append_to_current(extract_md, current_doc, cfg, mm)

                # Read updated content for split check
                final_content = mm.read_file(current_doc.path)

                # ============== CURRENT mode: rename and move to raw directory when exceeding length ==============
                if stage == "CURRENT":
                    if meta.tokens > int(cfg.memory.max_current_length):
                        # Extract thread worker ID from doc_id (if in CURRENT_THREAD_<n> format)
                        thread_worker_id = 0
                        if doc_id.startswith("CURRENT_THREAD_"):
                            try:
                                thread_worker_id = int(doc_id.split("_")[-1])
                            except (ValueError, IndexError):
                                thread_worker_id = 0

                        logger.info(
                            term_color.yellow(
                                f"[CURRENT] Document exceeds length ({meta.tokens} > {cfg.memory.max_current_length}), "
                                f"saving to raw directory (thread {thread_worker_id}, start seq {seq_start})",
                                enabled=True,
                            )
                        )
                        # Save to raw directory (auto-extract max seq from content as filename)
                        mm.save_to_raw(final_content)
                        # Clear CURRENT document (using doc_id)
                        mm.clear_doc(doc_id)
                        logger.info(
                            term_color.yellow(
                                f"[CURRENT] Cleared {doc_id} document, awaiting next batch of content",
                                enabled=True,
                            )
                        )
                    processed += 1
                    continue

                # ============== FULL mode: normal SPLIT/MERGE execution ==============
                # If exceeding length limit, split CURRENT document and clear
                if meta.tokens > int(cfg.memory.max_current_length):
                    self._split_current(final_content, current_epoch, user_id, cfg, llm, mm)

                processed += 1
            except RuntimeError as e:
                # Specifically handle SPLIT failure cases
                error_str = str(e)
                if "[Split]" in error_str and ("PLAN_UPDATE_PROMPT" in error_str or "empty result" in error_str):
                    # SPLIT failed, output detailed debug info
                    logger.error(term_color.red("=" * 80, enabled=True))
                    logger.error(term_color.red("[Split] Split failed", enabled=True))
                    logger.error(term_color.red("=" * 80, enabled=True))
                    logger.error("")
                    logger.error(term_color.yellow("[Split] Prompt used (PLAN_UPDATE_PROMPT):", enabled=True))
                    logger.error(term_color.cyan(PLAN_UPDATE_PROMPT, enabled=True))
                    logger.error("")
                    # Try to get current content_body and existing_docs_list for debugging
                    try:
                        # Re-read CURRENT document content
                        docs = mm.list_docs()
                        current_doc = next((d for d in docs if d.id == "CURRENT"), None)
                        if current_doc:
                            old_content = mm.read_file(current_doc.path)
                            _, content_body = extract_summary_from_markdown(old_content)
                            existing_docs = [d for d in docs if d.id != "CURRENT"]
                            existing_docs_list = [{"id": d.id, "summary": d.summary} for d in existing_docs]

                            logger.error(term_color.yellow("[Split] LLM input:", enabled=True))
                            logger.error(term_color.cyan(json_dumps_safe({
                                "new_content": content_body[:1000] + "..." if len(content_body) > 1000 else content_body,
                                "docs": existing_docs_list,
                            }), enabled=True))
                            logger.error("")
                            logger.error(term_color.yellow(f"[Split] CURRENT document tokens: {count_tokens(content_body):,}", enabled=True))
                            logger.error(term_color.yellow(f"[Split] Existing document count: {len(existing_docs_list)}", enabled=True))
                    except Exception as debug_e:
                        logger.error(f"[Split] Failed to retrieve debug info: {debug_e}")
                    logger.error("")
                    logger.error(term_color.red("=" * 80, enabled=True))
                    logger.error(term_color.red(f"[Split] Error details: {error_str}", enabled=True))
                    logger.error(term_color.red("=" * 80, enabled=True))
                else:
                    # Other RuntimeError
                    logger.error(
                        "add() single item processing exception: user_id=%s, tokens=%d, text_preview=%.100s",
                        user_id,
                        count_tokens(user_text) if user_text else 0,
                        user_text[:100] if user_text else "",
                    )
                logger.exception("Full exception info: %s", e)
                continue
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "add() single item processing exception: user_id=%s, tokens=%d, text_preview=%.100s",
                    user_id,
                    count_tokens(user_text) if user_text else 0,
                    user_text[:100] if user_text else "",
                )
                logger.exception("Full exception info: %s", e)
                continue

        logger.info("Processed add items: %s", processed)
        return processed

    def _init_mm_and_docs(self, cfg: InfiniMemoryConfig, store: str, user_id: str) -> Tuple[MemoryManager, List[DocMeta]]:
        """Initialize MemoryManager and get document list.

        Returns (mm, docs) tuple. CURRENT document is included in docs
        and participates in normal search strategies (BM25, LLM, etc.).
        """
        from .manager import MemoryManager  # Local import to avoid circular dependency
        mm = MemoryManager(
            root=cfg.root,
            data_root=cfg.memory.data_root,
            doc_dir=cfg.memory.doc_dir,
            meta_dir=cfg.memory.metadata_dir,
            index_file=cfg.memory.index_file,
            store=store,
            user_id=user_id,
            storage=cfg.get_storage_backend(),
        )
        docs = mm.list_docs()
        # Exclude empty documents only; CURRENT participates in search like any other doc
        docs = [d for d in docs if d.tokens > 0]

        return mm, docs

    def _execute_llm_search(
        self,
        query: str,
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
    ) -> List[DocMeta]:
        """Execute LLM-based search.

        Returns the list of matched documents.
        """
        doc_list_json = [{"id": d.id, "summary": d.summary} for d in docs]
        search_sys = SEARCH_MEMORY_PROMPT.format(search_limit=cfg.memory.search_limit)
        selection_json = _llm_chat_with_retry(
            llm,
            [
                {"role": "system", "content": search_sys},
                {"role": "user", "content": json_dumps_safe({
                    "query": query,
                    "docs": doc_list_json,
                    "limit": cfg.memory.search_limit,
                })},
            ],
            cfg.llm.model,
            cfg,
            context="[Search]",
        )
        ids = parse_id_list(selection_json, limit=cfg.memory.search_limit)
        return [d for d in docs if d.id in ids][: cfg.memory.search_limit]

    def _execute_bm25_search(
        self,
        query: str,
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
    ) -> List[DocMeta]:
        """Execute BM25 algorithm-based search.

        Returns the list of matched documents.
        """
        return fallback_pick_by_content(query, docs, limit=cfg.memory.search_limit, storage=cfg.get_storage_backend())

    def _execute_bm25_partition_search(
        self,
        query: str,
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
    ) -> List[DocMeta]:
        """Execute BM25 search on partitions split by H1 headings.

        Returns the list of matched documents (selected from documents containing matched partitions).
        """
        partitions = _bm25_search_partitions(query, docs, cfg.memory.search_limit, storage=cfg.get_storage_backend())

        # Collect matched document IDs (deduplicated)
        doc_ids = set(p["id"] for p in partitions)

        # Return matched documents (keeping original DocMeta objects)
        return [d for d in docs if d.id in doc_ids][: cfg.memory.search_limit]

    def _execute_folder_bm25_partition_search(
        self,
        query: str,
        cfg: InfiniMemoryConfig,
        user_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Search md documents in the specified directory, split by H1 headings then use BM25 algorithm.

        Unlike BM25_partition:
        - Does not search from doc directory, but from the directory specified by search_folder
        - Returns partition info list (not DocMeta objects)

        Args:
            query: Query string
            cfg: Configuration object
            user_id: User ID

        Returns:
            (picked, partitions) tuple:
            - picked: List of matched document/fragment info (for answer generation)
            - partitions: List of partition info (for log output)
        """
        from pathlib import Path
        import math
        logger = logging.getLogger("infini_memory_classic")

        # Get search_folder configuration
        search_folder = cfg.memory.search_folder
        if not search_folder:
            logger.warning("FOLDER_BM25_partition strategy requires memory.search_folder configuration, using default 'doc'")
            search_folder = "doc"

        # Build directory path
        folder_rel = str(Path(cfg.memory.data_root) / user_id / search_folder)

        if not cfg.get_storage_backend().exists(folder_rel):
            logger.warning(f"Search directory does not exist: {folder_rel}")
            return [], []

        # Read all .md files
        md_file_rels = sorted(cfg.get_storage_backend().glob(folder_rel, "*.md"))
        if not md_file_rels:
            logger.warning(f"No md files found in search directory: {folder_rel}")
            return [], []

        logger.info(f"FOLDER_BM25_partition: Read {len(md_file_rels)} md files from {search_folder} directory")

        # Tokenize
        q_tokens = _tokenize(query)
        q_terms = set(q_tokens)

        # Collect all partitions
        all_partitions: List[Dict[str, Any]] = []

        for md_file_rel in md_file_rels:
            try:
                content = cfg.get_storage_backend().read_text(md_file_rel)
                # Remove YAML Frontmatter
                _, body = extract_summary_from_markdown(content)

                # Split by H1 headings
                partitions = _split_doc_by_h1(body)

                for partition in partitions:
                    all_partitions.append({
                        "id": Path(md_file_rel).stem,  # Use filename as id
                        "path": md_file_rel,
                        "partition_index": partition["index"],
                        "partition_title": partition["title"],
                        "content": partition["content"],
                    })
            except Exception as e:
                logger.warning(f"Failed to read file: {md_file.name}, error: {e}")

        if not all_partitions:
            logger.warning("No partitions obtained")
            return [], []

        # Build corpus (content of all partitions)
        corpus = []  # [(partition_info, tokens)]
        df = {}
        for p in all_partitions:
            tokens = _tokenize(p["content"])
            corpus.append((p, tokens))
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        N = max(1, len(corpus))
        avgdl = sum(len(toks) for _, toks in corpus) / N

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        def score_partition(tokens: List[str]) -> float:
            score = 0.0
            dl = len(tokens)
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for q in q_terms:
                if q not in df:
                    continue
                n_q = df.get(q, 0)
                idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
                f = tf.get(q, 0)
                denom = f + k1 * (1 - b + b * (dl / avgdl))
                score += idf * (f * (k1 + 1)) / denom if denom > 0 else 0.0
            return score

        # Score all partitions
        scored = []
        for p, toks in corpus:
            s = score_partition(toks)
            if s > 0:
                p_copy = p.copy()
                p_copy["score"] = s
                scored.append((s, p_copy))

        # Sort by score, return top limit partitions
        scored.sort(key=lambda x: x[0], reverse=True)
        limit = max(1, int(cfg.memory.search_limit))
        top_partitions = [p for _, p in scored[:limit]]

        # Sort by partition_index (maintain original order)
        top_partitions.sort(key=lambda x: x["partition_index"])

        # Build the returned picked list (for answer generation)
        picked = []
        for p in top_partitions:
            picked.append({
                "id": p["id"],
                "path": p["path"],
                "content": "# " + str(p['partition_title']) + "\n\n" + str(p['content']),  # Include title
                "partitions": [{
                    "id": p["id"],
                    "partition_index": p["partition_index"],
                    "partition_title": p["partition_title"],
                    "score": p["score"],
                    "content": p["content"],
                }]
            })

        return picked, top_partitions


    def _agentic_build_line_cache(
        self,
        doc_id: str,
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
        line_cache: Dict[str, List[str]],
    ) -> List[str]:
        """Lazy load: read document content (without frontmatter), cache as line list."""
        if doc_id in line_cache:
            return line_cache[doc_id]

        doc = next((d for d in docs if d.id == doc_id), None)
        if doc is None:
            line_cache[doc_id] = []
            return []

        try:
            content = cfg.get_storage_backend().read_text(doc.path)
        except Exception:
            line_cache[doc_id] = []
            return []

        _, body = extract_summary_from_markdown(content)
        lines = body.split("\n")
        line_cache[doc_id] = lines
        return lines

    def _agentic_build_partition_cache(
        self,
        doc_id: str,
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
        partition_cache: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Lazy load: read document and split by H1, cache to partition_cache."""
        if doc_id in partition_cache:
            return partition_cache[doc_id]

        doc = next((d for d in docs if d.id == doc_id), None)
        if doc is None:
            partition_cache[doc_id] = []
            return []

        try:
            content = cfg.get_storage_backend().read_text(doc.path)
        except Exception:
            partition_cache[doc_id] = []
            return []

        _, body = extract_summary_from_markdown(content)
        parts = _split_doc_by_h1(body)
        partition_cache[doc_id] = parts
        return parts

    def _agentic_execute_tool(
        self,
        tool_call: Dict[str, Any],
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
        line_cache: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Execute a single tool call for Agentic search (grep-style precise tools)."""
        import re as re_mod
        logger = logging.getLogger("infini_memory_classic")

        tool_name = tool_call.get("tool", "")
        args = tool_call.get("args", {})

        if tool_name == "grep":
            pattern_str = args.get("pattern", "")
            limit = min(int(args.get("limit", cfg.memory.agentic_grep_limit)), cfg.memory.agentic_grep_limit)
            context_lines = min(int(args.get("context_lines", cfg.memory.agentic_grep_context_lines)), 5)
            max_line_len = cfg.memory.agentic_grep_max_line_length

            try:
                compiled = re_mod.compile(pattern_str, re_mod.IGNORECASE)
            except re_mod.error as e:
                logger.warning("[Agentic-Tool] grep regex compilation failed: %s", e)
                return {"tool": tool_name, "args": args, "result": {"error": f"invalid regex: {e}"}}

            matches = []
            total_matches = 0
            for d in docs:
                lines = self._agentic_build_line_cache(d.id, docs, cfg, line_cache)
                for i, line in enumerate(lines):
                    if compiled.search(line):
                        total_matches += 1
                        if len(matches) < limit:
                            matched_line = line[:max_line_len] + ("..." if len(line) > max_line_len else "")
                            ctx_before = lines[max(0, i - context_lines):i]
                            ctx_after = lines[i + 1:min(len(lines), i + 1 + context_lines)]
                            matches.append({
                                "doc_id": d.id,
                                "line_number": i + 1,  # 1-based
                                "matched_line": matched_line,
                                "context_before": ctx_before,
                                "context_after": ctx_after,
                            })

            logger.info("[Agentic-Tool] grep(%s, limit=%d) -> %d/%d matches",
                        pattern_str[:50], limit, len(matches), total_matches)
            return {"tool": tool_name, "args": args, "result": {
                "total_matches": total_matches,
                "matches": matches,
                "truncated": total_matches > limit,
            }}

        elif tool_name == "grep_doc":
            # Search within a single document, return all matches (no limit restriction)
            target_doc_id = args.get("doc_id", "")
            pattern_str = args.get("pattern", "")
            context_lines = min(int(args.get("context_lines", cfg.memory.agentic_grep_context_lines)), 5)
            max_line_len = cfg.memory.agentic_grep_max_line_length

            try:
                compiled = re_mod.compile(pattern_str, re_mod.IGNORECASE)
            except re_mod.error as e:
                logger.warning("[Agentic-Tool] grep_doc regex compilation failed: %s", e)
                return {"tool": tool_name, "args": args, "result": {"error": f"invalid regex: {e}"}}

            lines = self._agentic_build_line_cache(target_doc_id, docs, cfg, line_cache)
            if not lines:
                return {"tool": tool_name, "args": args, "result": {"error": f"doc_id {target_doc_id} not found"}}

            matches = []
            for i, line in enumerate(lines):
                if compiled.search(line):
                    matched_line = line[:max_line_len] + ("..." if len(line) > max_line_len else "")
                    ctx_before = lines[max(0, i - context_lines):i]
                    ctx_after = lines[i + 1:min(len(lines), i + 1 + context_lines)]
                    matches.append({
                        "line_number": i + 1,  # 1-based
                        "matched_line": matched_line,
                        "context_before": ctx_before,
                        "context_after": ctx_after,
                    })

            logger.info("[Agentic-Tool] grep_doc(%s, %s) -> %d matches",
                        target_doc_id[:30], pattern_str[:30], len(matches))
            return {"tool": tool_name, "args": args, "result": {
                "doc_id": target_doc_id,
                "total_matches": len(matches),
                "matches": matches,
            }}

        elif tool_name == "search":
            query_str = args.get("query", "")
            limit = min(int(args.get("limit", cfg.memory.search_limit)), cfg.memory.search_limit)
            partitions = _bm25_search_partitions(query_str, docs, limit, storage=cfg.get_storage_backend())

            snippet_tokens = cfg.memory.agentic_search_snippet_tokens
            q_terms = set(_tokenize(query_str))
            results = []
            for p in partitions:
                content = p.get("content", "")
                content_lines = content.split("\n")

                # Find the line containing the most query terms as the snippet center
                best_line_idx = 0
                best_score = -1
                for li, line in enumerate(content_lines):
                    line_terms = set(_tokenize(line))
                    overlap = len(q_terms & line_terms)
                    if overlap > best_score:
                        best_score = overlap
                        best_line_idx = li

                # Expand outward from the best line until reaching snippet_tokens
                snippet_lines = [content_lines[best_line_idx]]
                token_count = len(_tokenize(snippet_lines[0]))
                lo, hi = best_line_idx - 1, best_line_idx + 1
                while token_count < snippet_tokens and (lo >= 0 or hi < len(content_lines)):
                    if lo >= 0:
                        line_tokens = len(_tokenize(content_lines[lo]))
                        if token_count + line_tokens > snippet_tokens * 1.5:
                            break
                        snippet_lines.insert(0, content_lines[lo])
                        token_count += line_tokens
                        lo -= 1
                    if hi < len(content_lines) and token_count < snippet_tokens:
                        line_tokens = len(_tokenize(content_lines[hi]))
                        if token_count + line_tokens > snippet_tokens * 1.5:
                            break
                        snippet_lines.append(content_lines[hi])
                        token_count += line_tokens
                        hi += 1

                results.append({
                    "doc_id": p["id"],
                    "partition_index": p["partition_index"],
                    "partition_title": p.get("partition_title", ""),
                    "score": round(p.get("score", 0.0), 4),
                    "snippet": "\n".join(snippet_lines),
                    "total_lines": len(content_lines),
                })

            logger.info("[Agentic-Tool] search(%s, limit=%d) -> %d partitions",
                        query_str[:50], limit, len(results))
            return {"tool": tool_name, "args": args, "result": results}

        elif tool_name == "list_docs":
            offset = int(args.get("offset", 0))
            limit = min(int(args.get("limit", cfg.memory.agentic_list_docs_page_size)), cfg.memory.agentic_list_docs_page_size)
            page = docs[offset:offset + limit]
            result = {
                "total_docs": len(docs),
                "docs": [{"doc_id": d.id, "summary": d.summary, "tokens": d.tokens} for d in page],
                "has_more": offset + limit < len(docs),
            }
            logger.info("[Agentic-Tool] list_docs(offset=%d, limit=%d) -> %d docs",
                        offset, limit, len(page))
            return {"tool": tool_name, "args": args, "result": result}

        elif tool_name == "read_lines":
            doc_id = args.get("doc_id", "")
            start_line = max(1, int(args.get("start_line", 1)))
            end_line = int(args.get("end_line", start_line + 49))
            max_range = cfg.memory.agentic_read_lines_max_range
            # Limit maximum lines per request
            if end_line - start_line + 1 > max_range:
                end_line = start_line + max_range - 1

            lines = self._agentic_build_line_cache(doc_id, docs, cfg, line_cache)
            if not lines:
                return {"tool": tool_name, "args": args, "result": {"error": f"doc_id {doc_id} not found"}}

            total_lines = len(lines)
            # Clip to valid range
            start_line = min(start_line, total_lines)
            end_line = min(end_line, total_lines)

            selected = lines[start_line - 1:end_line]
            numbered = "\n".join(f"{start_line + i}: {l}" for i, l in enumerate(selected))

            result = {
                "doc_id": doc_id,
                "total_lines": total_lines,
                "start_line": start_line,
                "end_line": end_line,
                "lines": numbered,
            }
            logger.info("[Agentic-Tool] read_lines(%s, %d-%d)", doc_id[:30], start_line, end_line)
            return {"tool": tool_name, "args": args, "result": result}

        else:
            logger.warning("[Agentic-Tool] Unknown tool: %s", tool_name)
            return {"tool": tool_name, "args": args, "result": {"error": f"unknown tool: {tool_name}"}}

    def _execute_agentic_search(
        self,
        query: str,
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
        llm: LLMClient,
    ) -> Tuple[List[DocMeta], List[DocMeta], List[DocMeta], Optional[List[Dict[str, Any]]]]:
        """Execute the Agentic Tool-Calling search loop (grep-style precise tools).

        LLM acts as a search agent, obtaining result snippets through precise tools
        such as grep / grep_doc / search / list_docs / read_lines,
        and returns sufficient context for answering.

        Returns (picked, llm_picked, bm25_picked, bm25_partitions),
        consistent with the variable pattern of other strategies in search().
        """
        logger = logging.getLogger("infini_memory_classic")
        max_iterations = cfg.memory.agentic_max_iterations

        doc_map: Dict[str, DocMeta] = {d.id: d for d in docs}

        # Line cache and partition cache
        line_cache: Dict[str, List[str]] = {}
        partition_cache: Dict[str, List[Dict[str, Any]]] = {}

        # Build compact document summary catalog
        catalog_summary_len = cfg.memory.agentic_catalog_summary_length
        doc_catalog = []
        for d in docs:
            summary_text = (d.summary or "")[:catalog_summary_len]
            doc_catalog.append({"id": d.id, "summary": summary_text})

        # Multi-turn conversation history -- send doc_catalog to help Agent locate documents
        sys_prompt = AGENTIC_TOOL_AGENT_PROMPT.format(
            search_limit=cfg.memory.search_limit,
            max_iterations=max_iterations,
            grep_limit=cfg.memory.agentic_grep_limit,
            grep_context_lines=cfg.memory.agentic_grep_context_lines,
            grep_max_line_length=cfg.memory.agentic_grep_max_line_length,
            search_snippet_tokens=cfg.memory.agentic_search_snippet_tokens,
            list_docs_page_size=cfg.memory.agentic_list_docs_page_size,
            read_lines_max_range=cfg.memory.agentic_read_lines_max_range,
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json_dumps_safe({
                "query": query,
                "doc_count": len(docs),
                "doc_catalog": doc_catalog,
                "current_time": json_datetime_now(),
            })},
        ]

        collected_partitions: List[Dict[str, Any]] = []
        bm25_picked_total: List[DocMeta] = []

        for iteration in range(1, max_iterations + 1):
            logger.info("[Agentic-Tool] === Iteration %d/%d ===", iteration, max_iterations)

            # Call LLM
            response = _llm_chat_with_retry(
                llm, messages, cfg.llm.model, cfg, context="[Agentic-Tool]"
            )

            # Parse JSON response
            parsed = self._parse_agentic_json(response)
            if parsed is None:
                # JSON parse failed, try sending a retry message for LLM to fix output
                logger.warning("[Agentic-Tool] JSON parse failed, requesting LLM to re-output")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": json_dumps_safe({
                    "error": "Your response was not valid JSON. Please output ONLY valid JSON with either tool_calls or done field. No other text.",
                    "iteration": iteration,
                    "remaining_iterations": max_iterations - iteration,
                })})
                # Consume one iteration for retry
                continue

            # Check if Agent is done
            if parsed.get("done"):
                logger.info("[Agentic-Tool] Agent done: %s", parsed.get("reasoning", ""))

                # Process relevant_docs -- return full document content
                for doc_id in parsed.get("relevant_docs", []):
                    if doc_id not in doc_map:
                        continue
                    lines = self._agentic_build_line_cache(doc_id, docs, cfg, line_cache)
                    if not lines:
                        continue
                    full_content = "\n".join(lines)
                    collected_partitions.append({
                        "id": doc_id,
                        "partition_index": -1,  # Special marker: full document
                        "partition_title": "(full document)",
                        "score": 1.0,
                        "content": full_content,
                        "summary": doc_map[doc_id].summary,
                    })

                # Map relevant_snippets back to partition format -- use full partition content
                for snippet in parsed.get("relevant_snippets", []):
                    doc_id = snippet.get("doc_id", "")
                    start_line = max(1, int(snippet.get("start_line", 1)))
                    end_line = int(snippet.get("end_line", start_line + 49))

                    if doc_id not in doc_map:
                        continue

                    lines = self._agentic_build_line_cache(doc_id, docs, cfg, line_cache)
                    if not lines:
                        continue

                    # Determine the H1 partition containing the snippet, return full partition content
                    parts = self._agentic_build_partition_cache(doc_id, docs, cfg, partition_cache)
                    if not parts:
                        # No partition info, return the requested line range (with context expansion)
                        expanded_start = max(0, start_line - 11)
                        expanded_end = min(len(lines), end_line + 10)
                        snippet_content = "\n".join(lines[expanded_start:expanded_end])
                        collected_partitions.append({
                            "id": doc_id,
                            "partition_index": 0,
                            "partition_title": "",
                            "score": 1.0,
                            "content": snippet_content,
                            "summary": doc_map[doc_id].summary,
                        })
                        continue

                    # Find overlapping partitions by line number, return full partition content
                    current_line = 1
                    matched = False
                    for p in parts:
                        p_lines_count = len(p["content"].split("\n"))
                        p_start = current_line
                        p_end = current_line + p_lines_count - 1

                        if p_start <= end_line and p_end >= start_line:
                            # Use full partition content instead of just snippet lines
                            collected_partitions.append({
                                "id": doc_id,
                                "partition_index": p["index"],
                                "partition_title": p.get("title", ""),
                                "score": 1.0,
                                "content": p["content"],
                                "summary": doc_map[doc_id].summary,
                            })
                            matched = True
                            # Don't break, continue checking for cross-partition cases
                        elif matched:
                            # Already found matching partition and current one no longer overlaps, can stop
                            break

                        current_line = p_end + 1

                    if not matched:
                        # No overlapping partition found, return requested line range (with context expansion)
                        expanded_start = max(0, start_line - 11)
                        expanded_end = min(len(lines), end_line + 10)
                        snippet_content = "\n".join(lines[expanded_start:expanded_end])
                        collected_partitions.append({
                            "id": doc_id,
                            "partition_index": 0,
                            "partition_title": parts[0].get("title", "") if parts else "",
                            "score": 1.0,
                            "content": snippet_content,
                            "summary": doc_map[doc_id].summary,
                        })
                break

            # Agent wants to call tools
            tool_calls = parsed.get("tool_calls", [])
            if not tool_calls:
                logger.warning("[Agentic-Tool] No tool_calls and not done, stopping iteration")
                break

            # Execute each tool call
            tool_results = []
            for tc in tool_calls:
                result = self._agentic_execute_tool(tc, docs, cfg, line_cache)
                tool_results.append(result)

                # Record BM25 hits (for log statistics)
                t_name = tc.get("tool", "")
                if t_name in ("search", "grep", "grep_doc"):
                    r_data = result.get("result", {})
                    # search returns list, grep/grep_doc returns dict with matches
                    if isinstance(r_data, list):
                        for r in r_data:
                            did = r.get("doc_id", "")
                            if did in doc_map:
                                bm25_picked_total.append(doc_map[did])
                    elif isinstance(r_data, dict):
                        for m in r_data.get("matches", []):
                            did = m.get("doc_id", tc.get("args", {}).get("doc_id", ""))
                            if did in doc_map:
                                bm25_picked_total.append(doc_map[did])

            # Append Agent response and tool results to conversation history
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": json_dumps_safe({
                "tool_results": tool_results,
                "iteration": iteration,
                "remaining_iterations": max_iterations - iteration,
            })})

            logger.info("[Agentic-Tool] Iteration %d executed %d tool calls", iteration, len(tool_results))

        # Build final results -- deduplicate snippets
        seen_keys: set = set()
        deduped_partitions: List[Dict[str, Any]] = []
        for p in collected_partitions:
            key = (p["id"], p["partition_index"])
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_partitions.append(p)
            else:
                # Same partition multiple snippets, merge content
                for dp in deduped_partitions:
                    if dp["id"] == p["id"] and dp["partition_index"] == p["partition_index"]:
                        dp["content"] = dp["content"] + "\n\n" + p["content"]
                        break
        deduped_partitions = deduped_partitions[:cfg.memory.search_limit]

        # Extract deduplicated DocMeta from partition results
        picked_doc_ids = list(dict.fromkeys(p["id"] for p in deduped_partitions))
        picked = [doc_map[did] for did in picked_doc_ids if did in doc_map]

        # Calculate total content tokens to determine if supplementation is needed
        total_content_tokens = sum(len(_tokenize(p.get("content", ""))) for p in deduped_partitions)

        # Fallback strategy: when Agent has no results or results are too few, supplement with BM25 partition search
        # Extract core question from query (strip "The history chats..." prefix) to improve BM25 hit rate
        import re as _re_mod
        _core_q_match = _re_mod.search(r"(?:Now Answer the Question:|Question:)\s*(.+?)(?:\s*Answer:|\s*$)", query, _re_mod.DOTALL)
        bm25_query = _core_q_match.group(1).strip() if _core_q_match else query

        if not picked and not deduped_partitions:
            logger.info("[Agentic-Tool] Agent has no results, using BM25 partition fallback (top 5), query=%s", bm25_query[:80])
            fallback_partitions = _bm25_search_partitions(
                bm25_query, docs, min(5, cfg.memory.search_limit), storage=cfg.get_storage_backend()
            )
            deduped_partitions = fallback_partitions
            picked_doc_ids = list(dict.fromkeys(p["id"] for p in fallback_partitions))
            picked = [doc_map[did] for did in picked_doc_ids if did in doc_map]
            bm25_picked_total.extend(picked)
        elif total_content_tokens < 500 and deduped_partitions:
            # Agent found results but content is too few, supplement with BM25
            logger.info("[Agentic-Tool] Agent result content too few (%d tokens), supplementing with BM25", total_content_tokens)
            supplement_partitions = _bm25_search_partitions(
                bm25_query, docs, min(3, cfg.memory.search_limit), storage=cfg.get_storage_backend()
            )
            existing_keys = {(p["id"], p["partition_index"]) for p in deduped_partitions}
            for sp in supplement_partitions:
                sp_key = (sp["id"], sp["partition_index"])
                if sp_key not in existing_keys:
                    deduped_partitions.append(sp)
                    existing_keys.add(sp_key)
                    if sp["id"] in doc_map and doc_map[sp["id"]] not in picked:
                        picked.append(doc_map[sp["id"]])
                        bm25_picked_total.append(doc_map[sp["id"]])
            deduped_partitions = deduped_partitions[:cfg.memory.search_limit]

        logger.info("[Agentic-Tool] Final results: %d docs, %d snippets, %d tokens",
                     len(picked), len(deduped_partitions), total_content_tokens)

        return (
            picked[:cfg.memory.search_limit],
            [],  # llm_picked -- no more semantic_search
            _deduplicate_docs(bm25_picked_total),
            deduped_partitions if deduped_partitions else None,
        )

    @staticmethod
    def _parse_agentic_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON returned by the Agentic search LLM.

        Multi-stage parsing: direct JSON -> code block extraction -> curly brace extraction.
        Returns None on failure.
        """
        import json
        import re

        if not text or not text.strip():
            return None

        text = text.strip()

        # Stage 1: Direct parsing
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Stage 2: Extract from ```json ... ``` code block
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block:
            try:
                result = json.loads(code_block.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Stage 3: Extract from first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None


    def _get_bm25_partition_info(
        self,
        query: str,
        docs: List[DocMeta],
        cfg: InfiniMemoryConfig,
    ) -> List[Dict[str, Any]]:
        """Get BM25_partition search partition info (for log output).

        Returns a list of partition info, each containing:
        - id: document ID
        - partition_index: partition index
        - partition_title: partition title
        - score: BM25 score
        """
        return _bm25_search_partitions(query, docs, cfg.memory.search_limit, storage=cfg.get_storage_backend())

    def _build_search_results(
        self,
        picked: List[DocMeta],
        cfg: InfiniMemoryConfig,
        partitions: Optional[List[Dict[str, Any]]] = None,
        use_partition_content_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build search result list containing document content and metadata.

        Returns a list of result dictionaries.
        When partitions are provided, partition info is included in results (for BM25_partition strategy).
        When use_partition_content_only=True, content only uses partition snippet content
        (for AGENTIC strategy, to avoid returning full documents).
        """
        results: List[Dict[str, Any]] = []

        # Build mapping from picked documents to partitions (if partitions exist)
        partitions_by_doc_id: Dict[str, List[Dict[str, Any]]] = {}
        if partitions:
            for p in partitions:
                doc_id = p["id"]
                if doc_id not in partitions_by_doc_id:
                    partitions_by_doc_id[doc_id] = []
                partitions_by_doc_id[doc_id].append(p)

        for d in picked:
            if use_partition_content_only and d.id in partitions_by_doc_id:
                # AGENTIC strategy: content only uses snippet content, not reading full document
                content = "\n\n".join(p.get("content", "") for p in partitions_by_doc_id[d.id])
            else:
                # BM25_partition strategy: context should use hit partition content for answer generation.
                # Full document content is preserved here (for debugging/tracing), but partitions are included in result.
                raw_content = cfg.get_storage_backend().read_text(d.path)
                _, content = extract_summary_from_markdown(raw_content)
            result = {
                "id": d.id,
                "path": d.path,
                "summary": d.summary,
                "tokens": d.tokens,
                "created_at": d.created_at,
                "updated_at": d.updated_at,
                "parent_id": d.parent_id,
                "content": content,
            }
            # If partition info exists, add to result
            if d.id in partitions_by_doc_id:
                result["partitions"] = partitions_by_doc_id[d.id]
            results.append(result)

        return results

    def _log_search_results(
        self,
        query: str,
        picked: List[DocMeta],
        llm_picked: List[DocMeta],
        bm25_picked: List[DocMeta],
        strategy: str,
        bm25_partitions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Output search result logs."""
        logger = logging.getLogger("infini_memory_classic")

        # Calculate overlap count (for logging only)
        llm_ids = set(d.id for d in llm_picked)
        bm25_ids = set(d.id for d in bm25_picked)
        overlap = len(llm_ids & bm25_ids)

        # Output different logs based on strategy
        if strategy == "LLM":
            logger.info("Search: query='%s' %s", query, yellow(f"strategy=LLM, hit docs={len(picked)} (LLM hit {len(llm_picked)})"))
        elif strategy == "BM25":
            logger.info("Search: query='%s' %s", query, yellow(f"strategy=BM25, hit docs={len(picked)} (BM25 hit {len(bm25_picked)})"))
        elif strategy == "BM25_partition":
            logger.info("Search: query='%s' %s", query, yellow(f"strategy=BM25_partition, hit docs={len(picked)} (BM25 hit {len(picked)}, {len(bm25_partitions) if bm25_partitions else 0} partitions)"))
        elif strategy == "LLM_and_BM25":
            logger.info("Search: query='%s' %s", query, yellow(f"strategy=LLM_and_BM25, hit docs={len(picked)} (LLM hit {len(llm_picked)}, BM25 hit {len(bm25_picked)}, overlap {overlap})"))
        elif strategy == "LLM_and_BM25_partition":
            logger.info(
                "Search: query='%s' %s",
                query,
                yellow(
                    f"strategy=LLM_and_BM25_partition, hit docs={len(picked)} (LLM hit {len(llm_picked)}, BM25_partition hit {len(bm25_picked)}, {len(bm25_partitions) if bm25_partitions else 0} partitions)"
                ),
            )
        elif strategy == "FOLDER_BM25_partition":
            logger.info("Search: query='%s' %s", query, yellow(f"strategy=FOLDER_BM25_partition, hit docs={len(picked)} (BM25 hit {len(bm25_picked)}, {len(bm25_partitions) if bm25_partitions else 0} partitions)"))

        elif strategy == "AGENTIC":
            snippet_count = len(bm25_partitions) if bm25_partitions else 0
            logger.info("Search: query='%s' %s", query, yellow(f"strategy=AGENTIC, hit docs={len(picked)} ({snippet_count} snippets)"))

        else:  # LLM_or_BM25
            if llm_picked:
                logger.info("Search: query='%s' %s", query, yellow(f"strategy=LLM_or_BM25, hit docs={len(picked)} (LLM hit {len(llm_picked)})"))
            else:
                logger.info("Search: query='%s' %s", query, yellow(f"strategy=LLM_or_BM25, LLM no hit, using BM25, hit docs={len(picked)} (BM25 hit {len(bm25_picked)})"))

        # Record hit documents
        if strategy == "LLM_and_BM25_partition":
            llm_ids = {d.id for d in llm_picked}
            for d in llm_picked:
                logger.info("  - id=%s summary=%s %s", d.id, d.summary, term_color.cyan("[LLM]", enabled=True))
            for d in picked:
                if d.id in llm_ids:
                    continue
        elif strategy != "BM25_partition":
            # BM25_partition strategy only prints partition info, not full document summaries
            for d in picked:
                logger.info("  - id=%s summary=%s", d.id, d.summary)

        # For BM25_partition / LLM_and_BM25_partition strategies, additionally output partition info
        if strategy in {"BM25_partition", "LLM_and_BM25_partition", "FOLDER_BM25_partition", "AGENTIC"} and bm25_partitions:
            logger.info("  [BM25_partition hit partitions]")
            total_tokens = 0
            content_to_titles: dict[str, list[str]] = {}
            for p in bm25_partitions:
                title_colored = term_color.yellow(str(p["partition_title"]), enabled=True)
                score_colored = term_color.green(f"{float(p['score']):.4f}", enabled=True)
                tokens_colored = term_color.magenta(f"{len(_tokenize(p.get('content', '')))} tokens", enabled=True)
                partition_tokens = len(_tokenize(p.get('content', '')))
                total_tokens += partition_tokens
                content = p.get('content', '')
                if content not in content_to_titles:
                    content_to_titles[content] = []
                content_to_titles[content].append(str(p["partition_title"]))
                logger.info(
                    "    - id=%s partition[%d] title=%s score=%s %s",
                    p["id"],
                    p["partition_index"],
                    title_colored,
                    score_colored,
                    tokens_colored,
                )
            avg_tokens = total_tokens / len(bm25_partitions)
            log_line = f'{len(bm25_partitions)} partitions total tokens:{total_tokens}, avg tokens per partition:{avg_tokens:.1f}'
            duplicate_info = []
            for content, titles in content_to_titles.items():
                if len(titles) > 1:
                    duplicate_info.append(f'{titles[0]}(x{len(titles)})')
            if duplicate_info:
                log_line += f" {term_color.red(f'[Duplicate content: {', '.join(duplicate_info)}]', enabled=True)}"
            logger.info(f"  {term_color.yellow(log_line, enabled=True)}")

    def search(self, query: str, *, store: str, user_id: str, cfg: Optional[InfiniMemoryConfig] = None, llm: Optional[LLMClient] = None) -> Any:
        """Search interface: call LLM to select relevant documents based on query and document summaries, and return metadata and content.

        - user_id: User ID (required), used to isolate document storage paths for different users (data_root/<user_id>/doc).
        - When memory is not enabled, falls back to simple similarity return (backward compatible).
        - When enabled: selects document ids based on SEARCH_MEMORY_PROMPT, reads content and returns dict.
        - Automatically retries up to 5 times on API call failure, returns {"api_error": True, "results": []} on retry failure.
        """
        import time

        logger = logging.getLogger("infini_memory_classic")
        # Colored start log: --search--
        logger.info(term_color.green("--search--", enabled=True))
        if not query:
            return []

        if not cfg or not getattr(cfg, "memory", None) or not cfg.memory.enabled:
            # No valid config provided, return empty results (no longer uses in-memory database)
            logger.info("Search: file system memory not enabled, returning empty results")
            return []

        max_retries = cfg.memory.retry_max_attempts
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Initialize and get documents
                _, docs = self._init_mm_and_docs(cfg, store, user_id)

                # Output current document status
                if not docs:
                    logger.info("No documents available for search")

                # Execute different search strategies based on search_strategy
                strategy = cfg.memory.search_strategy
                llm_picked: List[DocMeta] = []
                bm25_picked: List[DocMeta] = []
                bm25_partitions: Optional[List[Dict[str, Any]]] = None
                picked: List[DocMeta] = []

                llm = llm or LLMClient(
                    api_key=cfg.llm.openai_api_key,
                    base_url=cfg.llm.openai_base_url,
                    retry_max_attempts=cfg.llm.retry_max_attempts,
                    retry_initial_wait=cfg.llm.retry_initial_wait,
                    retry_max_wait=cfg.llm.retry_max_wait,
                    retry_jitter=cfg.llm.retry_jitter,
                )

                if strategy == "LLM":
                    # Only use LLM
                    llm_picked = self._execute_llm_search(query, docs, cfg, llm)
                    picked = llm_picked

                elif strategy == "BM25":
                    # Only use BM25
                    bm25_picked = self._execute_bm25_search(query, docs, cfg)
                    picked = bm25_picked

                elif strategy == "BM25_partition":
                    # Use BM25 to search partitions split by H1 headings
                    bm25_picked = self._execute_bm25_partition_search(query, docs, cfg)
                    picked = bm25_picked
                    # Get partition info for log output
                    bm25_partitions = self._get_bm25_partition_info(query, docs, cfg)

                elif strategy == "LLM_and_BM25_partition":
                    # First use LLM to select some documents by summary, then use BM25_partition on remaining documents
                    llm_picked = self._execute_llm_search(query, docs, cfg, llm)

                    llm_ids = {d.id for d in llm_picked}
                    remaining_docs = [d for d in docs if d.id not in llm_ids]

                    bm25_picked = self._execute_bm25_partition_search(query, remaining_docs, cfg)
                    bm25_partitions = self._get_bm25_partition_info(query, remaining_docs, cfg)

                    picked_ids = set()
                    picked = []
                    for d in llm_picked:
                        if d.id not in picked_ids:
                            picked_ids.add(d.id)
                            picked.append(d)
                    for d in bm25_picked:
                        if d.id not in picked_ids:
                            picked_ids.add(d.id)
                            picked.append(d)

                elif strategy == "LLM_and_BM25":
                    # Use both LLM and BM25, merge and deduplicate
                    llm_picked = self._execute_llm_search(query, docs, cfg, llm)
                    bm25_picked = self._execute_bm25_search(query, docs, cfg)

                    # Merge and deduplicate: deduplicate by id, LLM results take priority, use union (no truncation)
                    picked_ids = set()
                    picked = []
                    for d in llm_picked:
                        if d.id not in picked_ids:
                            picked_ids.add(d.id)
                            picked.append(d)
                    for d in bm25_picked:
                        if d.id not in picked_ids:
                            picked_ids.add(d.id)
                            picked.append(d)
                    # No truncation, keep complete union

                elif strategy == "FOLDER_BM25_partition":
                    # Search md documents in the specified directory by splitting with H1 headings and using BM25
                    picked_dict, bm25_partitions = self._execute_folder_bm25_partition_search(query, cfg, user_id)
                    # Convert picked_dict to DocMeta format for compatibility
                    picked = []
                    for item in picked_dict:
                        picked.append(DocMeta(
                            id=item["id"],
                            path=item["path"],
                            summary="",
                            tokens=0,
                            created_at="",
                            updated_at="",
                            parent_id="",
                            update_count=0,
                            current_epoch=0,
                        ))
                    bm25_picked = picked

                elif strategy == "AGENTIC":
                    # Agentic search: multi-iteration intelligent search
                    picked, llm_picked, bm25_picked, bm25_partitions = self._execute_agentic_search(
                        query, docs, cfg, llm
                    )

                else:  # LLM_or_BM25 (default)
                    # Use LLM first, fall back to BM25 only if LLM has no hits
                    llm_picked = self._execute_llm_search(query, docs, cfg, llm)

                    # If LLM has no hits, use BM25
                    if not llm_picked:
                        bm25_picked = self._execute_bm25_search(query, docs, cfg)
                        picked = bm25_picked
                    else:
                        picked = llm_picked

                # Build results
                # For BM25_partition / LLM_and_BM25_partition strategies, pass partitions info
                partitions_arg = bm25_partitions if strategy in {"BM25_partition", "LLM_and_BM25_partition", "FOLDER_BM25_partition", "AGENTIC"} else None
                use_partition_only = (strategy == "AGENTIC")
                results = self._build_search_results(picked, cfg, partitions_arg, use_partition_content_only=use_partition_only)

                # Output logs (pass bm25_partitions parameter)
                self._log_search_results(query, picked, llm_picked, bm25_picked, strategy, bm25_partitions)

                return {"query": query, "results": results}

            except Exception as e:
                last_error = e
                # Check if it's a retryable error (429 rate limit, 5xx server errors, etc.)
                error_str = str(e).lower()
                is_retryable = any(code in error_str for code in ["429", "500", "502", "503", "504", "rate limit", "quota", "timeout"])

                if is_retryable and attempt < max_retries:
                    wait_time = attempt * cfg.memory.retry_initial_wait
                    logger.warning(
                        "search() attempt %d/%d failed: %s, retrying after %d seconds...",
                        attempt, max_retries, e, wait_time
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.exception("search() exception: %s", e)
                    return {"query": query, "results": [], "api_error": True}

        # All retries exhausted
        logger.error("search() still failed after %d retries, returning empty results", max_retries)
        return {"query": query, "results": [], "api_error": True}


def json_dumps_safe(obj: Dict) -> str:
    import json
    import logging

    logger = logging.getLogger("infini_memory_classic")

    try:
        result = json.dumps(obj, ensure_ascii=False)
        # Log serialized JSON length for debugging
        logger.debug("json_dumps_safe: input keys=%s, output length=%d chars",
                    list(obj.keys()) if isinstance(obj, dict) else "<non-dict>", len(result))
        return result
    except Exception as e:
        logger.error("json_dumps_safe: JSON serialization failed: %s, object type: %s", e, type(obj).__name__)
        # If serialization fails, at least return a JSON containing the error info (instead of empty object)
        return json.dumps({"error": f"JSON serialization failed: {e}"}, ensure_ascii=False)


def extract_summary_from_markdown(md_content: str) -> tuple:
    """Extract the summary field from YAML Frontmatter in Markdown content.

    Returns (summary, content_without_frontmatter).
    Returns empty string if not present.
    Also removes the YAML Frontmatter, returning pure Markdown body.
    """
    import re

    # Match YAML Frontmatter: starts with ---, ends with ---
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, md_content, flags=re.DOTALL)

    if match:
        frontmatter = match.group(1)
        summary = ""

        # Extract summary
        summary_match = re.search(r'summary:\s*(.+)', frontmatter)
        if summary_match:
            summary = summary_match.group(1).strip()

        # Remove YAML Frontmatter, return pure body
        content_without_frontmatter = md_content[match.end():]
        return summary, content_without_frontmatter
    else:
        # No YAML Frontmatter, return empty value and original content
        return "", md_content


def update_yaml_frontmatter(md_content: str, meta: "DocMeta") -> str:
    """Update or add YAML Frontmatter in Markdown content with complete metadata.

    Metadata includes: summary, id, created_at, updated_at, tokens, parent_id, update_count, current_epoch, merged_from, history
    """
    # Extract existing summary (if any)
    summary, content = extract_summary_from_markdown(md_content)

    # If no summary extracted from frontmatter, use meta.summary
    if not summary and meta.summary:
        summary = meta.summary

    # Build YAML Frontmatter
    frontmatter_lines = ["---"]
    if summary:
        frontmatter_lines.append(f"summary: {summary}")
    if hasattr(meta, "id") and meta.id:
        frontmatter_lines.append(f"id: {meta.id}")
    if hasattr(meta, "created_at") and meta.created_at:
        frontmatter_lines.append(f"created_at: {meta.created_at}")
    if hasattr(meta, "updated_at") and meta.updated_at:
        frontmatter_lines.append(f"updated_at: {meta.updated_at}")
    if hasattr(meta, "tokens") and meta.tokens is not None:
        frontmatter_lines.append(f"tokens: {meta.tokens}")
    if hasattr(meta, "parent_id") and meta.parent_id:
        frontmatter_lines.append(f"parent_id: {meta.parent_id}")
    # update_count and current_epoch are always output (even when 0)
    if hasattr(meta, "update_count"):
        frontmatter_lines.append(f"update_count: {meta.update_count}")
    if hasattr(meta, "current_epoch"):
        frontmatter_lines.append(f"current_epoch: {meta.current_epoch}")
    if hasattr(meta, "merged_from") and meta.merged_from:
        frontmatter_lines.append(f"merged_from: {json_dumps_safe(meta.merged_from)}")
    if hasattr(meta, "history") and meta.history:
        # history is a list, needs indented formatting
        frontmatter_lines.append("history:")
        for item in meta.history:
            frontmatter_lines.append(f"  - {json_dumps_safe(item)}")
    frontmatter_lines.append("---")
    frontmatter_lines.append("")

    return "\n".join(frontmatter_lines) + content


def parse_plan_updates(text: str) -> Dict:
    import json
    import re
    import logging

    logger = logging.getLogger("infini_memory_classic")

    # Method 1: Try standard JSON parsing
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "updates" in data and "new_docs" in data:
            return _normalize_plan_data(data)
    except Exception:
        pass

    # Method 2: Try fixing common JSON format issues (trailing commas, comments, etc.)
    try:
        # Remove possible JavaScript comments
        cleaned = re.sub(r'//.*?\n', '\n', text)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "updates" in data and "new_docs" in data:
            return _normalize_plan_data(data)
    except Exception:
        pass

    # Method 3: Use regex to extract JSON block (handle markdown code block wrapping)
    try:
        # Extract ```json ... ``` code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict) and "updates" in data and "new_docs" in data:
                return _normalize_plan_data(data)
    except Exception:
        pass

    # Method 4: Try extracting JSON object directly from text
    try:
        # Find content between first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            json_str = text[first_brace:last_brace + 1]
            data = json.loads(json_str)
            if isinstance(data, dict) and "updates" in data and "new_docs" in data:
                return _normalize_plan_data(data)
    except Exception:
        pass

    # Method 5: Try fixing truncated JSON (complete unclosed structures)
    try:
        # Find first { and all content after it
        first_brace = text.find('{')
        if first_brace >= 0:
            # Extract all content starting from the first {
            json_str = text[first_brace:]

            # Check if truncated: use stack to check bracket balance
            stack = []
            for i, char in enumerate(json_str):
                if char in '{[':
                    stack.append((char, i))
                elif char == '}':
                    if stack and stack[-1][0] == '{':
                        stack.pop()
                    elif not stack:
                        # Extra closing bracket, ignore
                        pass
                elif char == ']':
                    if stack and stack[-1][0] == '[':
                        stack.pop()
                    elif not stack:
                        # Extra closing bracket, ignore
                        pass

            # Remaining brackets in stack are unclosed
            if stack:
                # Count brackets to complete (in closing order)
                to_close = []
                for char, _ in reversed(stack):
                    if char == '{':
                        to_close.append('}')
                    elif char == '[':
                        to_close.append(']')

                if to_close:
                    missing_str = ''.join(to_close)
                    logger.info(f"[parse_plan_updates] Detected truncated JSON, attempting to complete: {missing_str}")

                    # Complete missing closing symbols
                    fixed_str = json_str + missing_str
                    data = json.loads(fixed_str)
                    if isinstance(data, dict) and "updates" in data and "new_docs" in data:
                        logger.info(f"[parse_plan_updates] Successfully fixed and parsed truncated JSON")
                        return _normalize_plan_data(data)
    except Exception as e:
        logger.debug(f"[parse_plan_updates] Fix truncated JSON failed: {e}")

    # All methods failed, return empty result
    return {"updates": [], "new_docs": []}


def _normalize_plan_data(data: Dict) -> Dict:
    """Normalize plan data, ensuring correct types."""
    # Basic validation
    updates = data.get("updates")
    if not isinstance(updates, list):
        data["updates"] = []
    # Normalize updates
    norm_updates: List[Dict] = []
    for u in data["updates"]:
        if not isinstance(u, dict):
            continue
        uid = u.get("id")
        new_content = u.get("new_content")
        if not isinstance(uid, (str, int)):
            continue
        if not isinstance(new_content, str):
            new_content = ""
        norm_updates.append({"id": str(uid), "new_content": new_content})
    data["updates"] = norm_updates

    # Parse new documents list
    new_docs = data.get("new_docs")
    norm_new_docs: List[Dict[str, str]] = []
    if isinstance(new_docs, list):
        for doc in new_docs:
            if not isinstance(doc, dict):
                continue
            title = str(doc.get("title", "") or "").strip()
            content = str(doc.get("content", "") or "").strip()
            if content:
                norm_new_docs.append({"title": title, "content": content})
    data["new_docs"] = norm_new_docs

    return {"updates": data["updates"], "new_docs": data["new_docs"]}


def parse_merge_groups(text: str) -> Dict:
    """Robustly parse the return result of SELECT_MERGE_GROUPS_PROMPT.

    Args:
        text: JSON string returned by LLM

    Returns:
        Parsed dictionary, format is {"groups": [...]}, returns {"groups": []} on parse failure
    """
    import json
    import re
    import logging

    logger = logging.getLogger("infini_memory_classic")

    # Method 1: Try standard JSON parsing
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "groups" in data:
            return _normalize_merge_groups_data(data)
    except Exception:
        pass

    # Method 2: Try fixing common JSON format issues (trailing commas, comments, etc.)
    try:
        # Remove possible JavaScript comments
        cleaned = re.sub(r'//.*?\n', '\n', text)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "groups" in data:
            return _normalize_merge_groups_data(data)
    except Exception:
        pass

    # Method 3: Use regex to extract JSON block (handle markdown code block wrapping)
    try:
        # Extract ```json ... ``` code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict) and "groups" in data:
                return _normalize_merge_groups_data(data)
    except Exception:
        pass

    # Method 4: Try extracting the first complete JSON object from text
    try:
        # Find first { and matching } using stack
        first_brace = text.find('{')
        if first_brace >= 0:
            # Use stack to find matching closing bracket
            stack = []
            json_str = ""
            for i, char in enumerate(text[first_brace:], start=first_brace):
                json_str += char
                if char in '{[':
                    stack.append(char)
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        if not stack:
                            # Found complete JSON object
                            break
                elif char == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()

            if json_str:
                data = json.loads(json_str)
                if isinstance(data, dict) and "groups" in data:
                    return _normalize_merge_groups_data(data)
    except Exception:
        pass

    # Method 5: Try fixing truncated JSON (complete unclosed structures)
    try:
        first_brace = text.find('{')
        if first_brace >= 0:
            json_str = text[first_brace:]

            # Check if truncated: use stack to check bracket balance
            stack = []
            for i, char in enumerate(json_str):
                if char in '{[':
                    stack.append((char, i))
                elif char == '}':
                    if stack and stack[-1][0] == '{':
                        stack.pop()
                elif char == ']':
                    if stack and stack[-1][0] == '[':
                        stack.pop()

            if stack:
                # Count brackets to complete
                to_close = []
                for char, _ in reversed(stack):
                    if char == '{':
                        to_close.append('}')
                    elif char == '[':
                        to_close.append(']')

                if to_close:
                    missing_str = ''.join(to_close)
                    logger.info(f"[parse_merge_groups] Detected truncated JSON, attempting to complete: {missing_str}")
                    fixed_str = json_str + missing_str
                    data = json.loads(fixed_str)
                    if isinstance(data, dict) and "groups" in data:
                        logger.info(f"[parse_merge_groups] Successfully fixed and parsed truncated JSON")
                        return _normalize_merge_groups_data(data)
    except Exception as e:
        logger.debug(f"[parse_merge_groups] Fix truncated JSON failed: {e}")

    # All methods failed, return empty result
    logger.warning("[parse_merge_groups] All parsing methods failed, returning empty groups")
    return {"groups": []}


def _normalize_merge_groups_data(data: Dict) -> Dict:
    """Normalize merge groups data, ensuring correct types."""
    groups = data.get("groups")
    if not isinstance(groups, list):
        return {"groups": []}

    norm_groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        doc_ids = group.get("doc_ids")
        reason = group.get("reason", "")

        if not isinstance(doc_ids, list):
            continue

        # Normalize doc_ids to string list
        norm_doc_ids = []
        for doc_id in doc_ids:
            if isinstance(doc_id, (str, int)):
                norm_doc_ids.append(str(doc_id))

        if norm_doc_ids:
            norm_groups.append({
                "doc_ids": norm_doc_ids,
                "reason": str(reason) if reason else ""
            })

    return {"groups": norm_groups}


def split_content_by_newlines(content: str) -> tuple[str, str]:
    """Split content evenly into two parts by newlines, without splitting within a line.

    Args:
        content: Content to split

    Returns:
        (part1, part2): The two split parts

    Strategy:
    1. Split content into line list by newlines
    2. Calculate midpoint (half of total lines)
    3. Search for the best split point near the midpoint (prefer splitting at blank lines)
    4. Return two parts
    """
    import logging
    logger = logging.getLogger("infini_memory_classic")

    # Count original content
    original_tokens = count_tokens(content)
    lines = content.split('\n')
    total_lines = len(lines)

    logger.info(
        term_color.cyan(
            f"[Split] Starting content split: original {total_lines} lines, {original_tokens:,} tokens",
            enabled=True,
        )
    )

    if total_lines <= 1:
        # Only one line or empty content, cannot split
        logger.warning(
            term_color.yellow(
                f"[Split] Content has only {total_lines} line(s), cannot split",
                enabled=True,
            )
        )
        return content, ""

    # Calculate ideal split position (midpoint)
    mid = total_lines // 2
    logger.debug(f"[Split] Ideal split position: line {mid} (total {total_lines} lines)")

    # Prefer splitting at blank lines (search for blank lines near mid)
    best_split = mid
    search_radius = min(50, total_lines // 4)  # Search range

    # Search backward for blank lines
    for i in range(mid, max(0, mid - search_radius), -1):
        if i < total_lines and not lines[i].strip():
            best_split = i
            logger.debug(f"[Split] Found blank line at line {i} (backward search), splitting here")
            break

    # If not found, search forward for blank lines
    if best_split == mid:
        for i in range(mid, min(total_lines, mid + search_radius)):
            if not lines[i].strip():
                best_split = i
                logger.debug(f"[Split] Found blank line at line {i} (forward search), splitting here")
                break

    part1_lines = lines[:best_split]
    part2_lines = lines[best_split:]

    # Remove trailing blank lines
    removed_trailing = 0
    while part1_lines and not part1_lines[-1].strip():
        part1_lines.pop()
        removed_trailing += 1
    # Remove leading blank lines
    removed_leading = 0
    while part2_lines and not part2_lines[0].strip():
        part2_lines.pop(0)
        removed_leading += 1

    part1 = '\n'.join(part1_lines)
    part2 = '\n'.join(part2_lines)

    # Count post-split token numbers
    part1_tokens = count_tokens(part1)
    part2_tokens = count_tokens(part2)

    logger.info(
        term_color.cyan(
            f"[Split] Split complete: part1={len(part1_lines)} lines ({part1_tokens:,} tokens), "
            f"part2={len(part2_lines)} lines ({part2_tokens:,} tokens) | "
            f"original={original_tokens:,} tokens -> post-split={part1_tokens + part2_tokens:,} tokens "
            f"(change: {part1_tokens + part2_tokens - original_tokens:+d} tokens)",
            enabled=True,
        )
    )

    if removed_trailing > 0:
        logger.debug(f"[Split] Removed {removed_trailing} trailing blank lines from part1")
    if removed_leading > 0:
        logger.debug(f"[Split] Removed {removed_leading} leading blank lines from part2")

    return part1, part2


def split_content_by_newlines_non_overlapping(content: str) -> tuple[str, str]:
    """Split content into two parts by newlines, minimizing semantic "overlap".

    This function is used for fallback splitting (when PLAN_UPDATE_PROMPT output cannot be parsed as JSON).
    In Markdown documents, the same fact may be described repeatedly at the boundary of adjacent paragraphs.
    If simply split at blank lines, the LLM may "see" the same fact in both parts and generate documents for each,
    causing subsequent doc content overlap.

    Strategy:
    - Use existing blank-line-preferred split logic to get initial (part1, part2)
    - Take a small segment from the tail of part1 and head of part2 as "boundary window"
    - If part2's beginning contains the same text prefix as part1's ending, trim that duplicate prefix from part2

    Note: This is heuristic deduplication, only handles very common boundary repetition cases.
    """

    part1, part2 = split_content_by_newlines(content)
    if not part1 or not part2:
        return part1, part2

    # Take last N chars from part1 and first N*2 chars from part2 for overlap detection
    window_chars = 800
    tail = part1[-window_chars:]
    head = part2[: window_chars * 2]

    # Only search for "tail text" occurrence in head, to avoid false deletions
    # Use progressively shorter suffix matching: find longest common string and trim from part2
    max_overlap = 0
    tail_len = len(tail)
    # Use 50-char step, find maximum overlap
    step = 50
    for n in range(tail_len, step - 1, -step):
        suffix = tail[-n:]
        if suffix.strip() and head.startswith(suffix):
            max_overlap = n
            break

    if max_overlap > 0:
        trimmed = part2[max_overlap:]
        # Clean up leading blank lines caused by trimming
        trimmed_lines = trimmed.split("\n")
        while trimmed_lines and not trimmed_lines[0].strip():
            trimmed_lines.pop(0)
        part2 = "\n".join(trimmed_lines)

    return part1, part2


def parse_id_list(text: str, *, limit: int) -> List[str]:
    import json
    import re

    # 1) Try strict JSON parsing
    try:
        data = json.loads(text)
        ids = data.get("ids") if isinstance(data, dict) else None
        if isinstance(ids, list):
            out: List[str] = []
            for x in ids:
                s = str(x).strip()
                if s:
                    out.append(s)
                if len(out) >= max(1, int(limit)):
                    break
            return out
    except Exception:
        ...

    # 2) Relaxed: extract `"ids": [ ... ]` fragment from text
    try:
        m = re.search(r"\"ids\"\s*:\s*\[(.*?)\]", text, flags=re.S)
        if m:
            inside = m.group(1)
            # Allow quoted/unquoted UUID or any id characters
            cand = re.findall(r"[\w\-]+", inside)
            out2: List[str] = []
            for s in cand:
                s = s.strip().strip('"\'')
                if s and s not in out2:
                    out2.append(s)
                if len(out2) >= max(1, int(limit)):
                    break
            if out2:
                return out2
    except Exception:
        ...

    # 3) Fallback: extract possible UUIDs from text as id candidates
    try:
        uuids = re.findall(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", text)
        out3 = []
        for s in uuids:
            if s not in out3:
                out3.append(s)
            if len(out3) >= max(1, int(limit)):
                break
        return out3
    except Exception:
        return []


def _deduplicate_docs(docs: List[DocMeta]) -> List[DocMeta]:
    """Deduplicate DocMeta list by id, preserving first-occurrence order."""
    seen: set = set()
    result: List[DocMeta] = []
    for d in docs:
        if d.id not in seen:
            seen.add(d.id)
            result.append(d)
    return result


def _tokenize(text: str) -> List[str]:
    import re
    # Supports Chinese and English with numbers, splits by alphanumeric sequences and Chinese characters
    tokens = []
    for m in re.finditer(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower()):
        tokens.append(m.group(0))
    return tokens


def _split_doc_by_h1(content: str) -> List[Dict[str, str]]:
    """Split document content into multiple partitions by H1 headings (# heading).

    Args:
        content: Document content (YAML Frontmatter already removed)

    Returns:
        List of partitions, each partition is a dict containing:
        - title: partition title (extracted from H1 heading, or "Untitled" if none)
        - content: partition content
        - index: partition index (starting from 0)
    """
    import re
    import logging

    partitions = []
    lines = content.split('\n')

    current_title = "Untitled"
    current_content_lines = []

    for line in lines:
        # Detect H1 headings (# heading)
        h1_match = re.match(r'^#\s+(.+)$', line)
        if h1_match:
            # If there's existing content, save the previous partition first
            if current_content_lines:
                partitions.append({
                    "title": current_title,
                    "content": '\n'.join(current_content_lines),
                    "index": len(partitions),
                })
            # Start new partition
            current_title = h1_match.group(1).strip()
            current_content_lines = [line]
        else:
            current_content_lines.append(line)

    # Save the last partition
    if current_content_lines:
        partitions.append({
            "title": current_title,
            "content": '\n'.join(current_content_lines),
            "index": len(partitions),
        })

    # If no partitions were split out, return entire content as one partition
    if not partitions and content.strip():
        return [{
            "title": "Untitled",
            "content": content,
            "index": 0,
        }]

    return partitions


def _bm25_search_partitions(
    query: str,
    docs: List[Any],
    limit: int,
    root=None,
    storage=None,
) -> List[Dict[str, Any]]:
    """Use BM25 algorithm to search partitions split by H1 headings.

    Args:
        query: Query string
        docs: Document list (DocMeta objects)
        limit: Maximum number of results to return
        root: Document root directory path (deprecated, use storage)
        storage: StorageBackend instance

    Returns:
        List of partition results, each containing:
        - id: document ID
        - partition_index: partition index
        - partition_title: partition title
        - score: BM25 score
        - content: partition content
        - summary: document summary
    """
    import math
    import logging

    logger = logging.getLogger("infini_memory_classic")

    if (not root and not storage) or not docs:
        return []

    # Tokenize
    q_tokens = _tokenize(query)
    q_terms = set(q_tokens)

    # Collect all partitions
    all_partitions: List[Dict[str, Any]] = []

    for d in docs:
        try:
            doc_path = getattr(d, "path", "")
            if storage is not None:
                doc_content = storage.read_text(doc_path)
            else:
                doc_content = (root / doc_path).read_text(encoding="utf-8")
        except Exception:
            continue

        # Remove YAML Frontmatter
        _, body = extract_summary_from_markdown(doc_content)

        # Split by H1 headings
        partitions = _split_doc_by_h1(body)

        for partition in partitions:
            all_partitions.append({
                "id": d.id,
                "partition_index": partition["index"],
                "partition_title": partition["title"],
                "content": partition["content"],
                "summary": getattr(d, "summary", ""),
            })

    # Build corpus (content of all partitions)
    corpus = []  # [(partition_info, tokens)]
    df = {}
    for p in all_partitions:
        tokens = _tokenize(p["content"])
        corpus.append((p, tokens))
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    N = max(1, len(corpus))
    avgdl = sum(len(toks) for _, toks in corpus) / N

    # BM25 parameters
    k1 = 1.5
    b = 0.75

    def score_partition(tokens: List[str]) -> float:
        score = 0.0
        dl = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for q in q_terms:
            if q not in df:
                continue
            n_q = df.get(q, 0)
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
            f = tf.get(q, 0)
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score += idf * (f * (k1 + 1)) / denom if denom > 0 else 0.0
        return score

    # Score all partitions
    scored = []
    for p, toks in corpus:
        s = score_partition(toks)
        if s > 0:
            p_copy = p.copy()
            p_copy["score"] = s
            scored.append((s, p_copy))

    # Sort by score, return top limit
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[: max(1, int(limit))]]


def fallback_pick_by_summary(query: str, docs: List[Any], *, limit: int) -> List[Any]:
    # Calculate word overlap ratio between query and summary
    q_tokens = set(_tokenize(query))
    scored = []
    for d in docs:
        s_tokens = set(_tokenize(getattr(d, "summary", "")))
        inter = q_tokens & s_tokens
        score = len(inter) / max(1, len(q_tokens))
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored][: max(1, int(limit))]


def fallback_pick_by_content(query: str, docs: List[Any], *, limit: int, root=None, storage=None) -> List[Any]:
    # Use simplified BM25 on document body for fallback search, suitable for cross-language or summary-irrelevant scenarios
    if not root and not storage:
        return []

    import math

    # Tokenize
    q_tokens = _tokenize(query)
    q_terms = set(q_tokens)

    # Build corpus term frequencies and document lengths
    corpus = []  # [(doc, tokens)]
    df = {}
    for d in docs:
        try:
            doc_path = getattr(d, "path", "")
            if storage is not None:
                content = storage.read_text(doc_path)
            else:
                content = (root / doc_path).read_text(encoding="utf-8")
        except Exception:
            content = ""
        tokens = _tokenize(content)
        corpus.append((d, tokens))
        seen = set()
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    N = max(1, len(corpus))
    avgdl = sum(len(toks) for _, toks in corpus) / N

    # BM25 parameters
    k1 = 1.5
    b = 0.75

    def score_doc(tokens: List[str]) -> float:
        score = 0.0
        dl = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for q in q_terms:
            if q not in df:
                continue
            n_q = df.get(q, 0)
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
            f = tf.get(q, 0)
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score += idf * (f * (k1 + 1)) / denom if denom > 0 else 0.0
        return score

    scored = []
    for d, toks in corpus:
        s = score_doc(toks)
        if s > 0:
            scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored][: max(1, int(limit))]


def _process_new_doc(
    new_doc: Dict[str, str],
    llm: LLMClient,
    cfg: InfiniMemoryConfig,
    mm: MemoryManager,
    current_epoch: int,
    idx_new: int,
    total_new: int,
    rewrite_source: Optional[str] = None,
) -> Optional[DocMeta]:
    """Process a single new document's REWRITE_DOC_PROMPT call and creation.

    Returns DocMeta or None (on failure).
    """
    import threading
    logger = logging.getLogger("infini_memory_classic")
    # Get thread ID from current thread name (e.g., "RewriteNew_0" -> 0)
    thread_name = threading.current_thread().name
    thread_id = 0
    if "_" in thread_name:
        try:
            thread_id = int(thread_name.split("_")[-1])
        except (ValueError, IndexError):
            thread_id = 0

    p_title = new_doc.get("title", "")
    p_content = new_doc.get("content", "")

    # Calculate input content token count
    p_tokens_input = count_tokens(p_content)

    logger.info(
        term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
        "[Split] Processing new document %d/%d: title=%s, tokens=%d",
        idx_new + 1, total_new, p_title, p_tokens_input
    )

    if not p_content:
        logger.warning(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] Skipping empty content new document: title=%s", p_title
        )
        return None

    try:
        # Call REWRITE_DOC_PROMPT to optimize document content then create new document (with retry)
        logger.info(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] Calling REWRITE_DOC_PROMPT (new document): title=%s", p_title
        )
        rewrite_prompt = REWRITE_DOC_PROMPT.format(
            summary_length=cfg.memory.summary_length,
            old_content="",
            new_content=f"# {p_title}\n\n{p_content}" if p_title else p_content,
        )
        optimized_content = _llm_chat_with_retry(
            llm,
            [{"role": "system", "content": rewrite_prompt}],
            cfg.llm.model,
            cfg,
            context="[Split][NewDoc]",
        )

        # Calculate REWRITE returned content token count
        p_tokens_output = count_tokens(optimized_content)

        logger.info(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] REWRITE_DOC_PROMPT returned (new document): title=%s, tokens=%d",
            p_title, p_tokens_output
        )

        # Extract summary and body
        new_summary, new_body = extract_summary_from_markdown(optimized_content)
        if not new_summary:
            new_summary = p_title

        p_tokens = count_tokens(new_body)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import uuid
        ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        doc_id = f"{ts}_{uuid.uuid4()}"
        filename = f"{doc_id}.md"
        path = mm.doc_dir / filename
        now = json_datetime_now()

        # Build history record
        history = []
        if rewrite_source:
            # Created from rewrite document split in SPLIT_MERGE mode
            history.append({
                "event": "created_from_rewrite",
                "source": rewrite_source,
                "timestamp": now,
            })
        else:
            # Created from SPLIT stage in FULL mode
            history.append({
                "event": "created_from_split",
                "timestamp": now,
            })

        p_meta = DocMeta(
            id=doc_id,
            path=mm._rel(path),
            created_at=now,
            updated_at=now,
            tokens=p_tokens,
            summary=new_summary,
            parent_id="CURRENT",
            current_epoch=current_epoch,
            history=history,
        )
        final_p_content = update_yaml_frontmatter(optimized_content, p_meta)
        # Ensure directory exists
        mm._ensure_dirs()
        mm.write_file(mm._rel(path), final_p_content)
        # Use thread-safe method to add to index
        mm.add_doc_to_index(p_meta)
        # Record operation history
        mm.log_event({
            "event": "created_doc",
            "doc_id": doc_id,
            "timestamp": now,
        })
        logger.info(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] New document created successfully: id=%s, title=%s", doc_id, p_title
        )
        return p_meta
    except Exception as e:
        logger.error(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] Processing new document failed: title=%s, error=%s", p_title, e
        )
        logger.exception("[Split] New document processing exception details")
        return None


def _process_update(
    update: Dict[str, Any],
    llm: LLMClient,
    cfg: InfiniMemoryConfig,
    mm: MemoryManager,
    existing_docs_list: List[Dict[str, str]],
    idx_update: int,
    total_updates: int,
) -> Optional[DocMeta]:
    """Process a single update's REWRITE_DOC_PROMPT call and document update.

    Returns DocMeta or None (on failure).
    """
    import threading
    import logging
    logger = logging.getLogger("infini_memory_classic")
    # Get thread ID from current thread name (e.g., "RewriteUpdate_0" -> 0)
    thread_name = threading.current_thread().name
    thread_id = 0
    if "_" in thread_name:
        try:
            thread_id = int(thread_name.split("_")[-1])
        except (ValueError, IndexError):
            thread_id = 0

    target_id = str(update.get("id") or "").strip()
    new_content = update.get("new_content", "")

    # If LLM output a non-existent target_id, treat it as "new document" creation (same as new_docs handling),
    # to ensure update doesn't fail due to wrong id, while avoiding duplicate creation from retries.
    allowed_ids = {str(d.get("id")) for d in existing_docs_list if d.get("id")}
    if not target_id or target_id not in allowed_ids:
        logger.warning(
            term_color.yellow(
                "[Split] Update target id not in existing document list, converting to new document: target_id=%s",
                enabled=True,
            ),
            target_id,
        )
        # Cannot reliably pass current_epoch/rewrite_source here (determined by upper-level SPLIT flow),
        # so return None and let upper level convert this update to new_docs for unified creation.
        return None

    logger.info(
        term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
        "[Split] Processing update %d/%d: target_id=%s, new_content_len=%d",
        idx_update + 1, total_updates, target_id, len(new_content)
    )

    if not target_id or not new_content:
        logger.warning(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] Skipping invalid update: target_id=%s, new_content_len=%d", target_id, len(new_content)
        )
        return None

    try:
        # Read target document
        target_doc = mm.get_doc_by_id(target_id)
        if not target_doc:
            # Try fuzzy matching via summary in existing_docs_list
            # This handles cases where LLM returns incomplete IDs
            logger.warning(
                term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
                "[Split] Target document does not exist: %s, trying fuzzy match via summary", target_id
            )

            # Extract UUID part of target_id (for logging)
            uuid_part = target_id.split("_")[-1] if "_" in target_id else target_id

            # Search for documents with similar summaries in existing_docs_list
            best_match_doc = None
            best_match_summary = None
            for doc_info in existing_docs_list:
                doc_id = doc_info.get("id", "")
                doc_summary = doc_info.get("summary", "")

                # Check if UUID matches
                if uuid_part in doc_id:
                    candidate_doc = mm.get_doc_by_id(doc_id)
                    if candidate_doc:
                        logger.info(
                            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
                            "[Split] Found document via UUID match: input=%s -> matched=%s", target_id, doc_id
                        )
                        target_doc = candidate_doc
                        break

                # If UUID doesn't match, record summary for debugging
                if best_match_summary is None or (len(doc_summary) < len(best_match_summary) if best_match_summary else True):
                    best_match_doc = doc_id
                    best_match_summary = doc_summary

            if not target_doc:
                logger.error(
                    term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
                    "[Split] Failed to find document via UUID match: target_id=%s (uuid_part=%s), best_candidate=%s", target_id, uuid_part, best_match_doc
                )
                return None

        old_doc_content = mm.read_file(target_doc.path)
        _, old_body = extract_summary_from_markdown(old_doc_content)

        logger.info(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] new_content length: %d", len(new_content)
        )

        # Use REWRITE_DOC_PROMPT to rewrite (with retry)
        logger.info(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] Calling REWRITE_DOC_PROMPT (update document): target_id=%s", target_id
        )
        rewrite_prompt = REWRITE_DOC_PROMPT.format(
            summary_length=cfg.memory.summary_length,
            old_content=old_body,
            new_content=new_content,
        )
        merged_content = _llm_chat_with_retry(
            llm,
            [{"role": "system", "content": rewrite_prompt}],
            cfg.llm.model,
            cfg,
            context="[Split][Update]",
        )

        # Calculate REWRITE returned content token count
        merged_tokens_output = count_tokens(merged_content)

        logger.info(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] REWRITE_DOC_PROMPT returned (update document): target_id=%s, tokens=%d",
            target_id, merged_tokens_output
        )

        # Extract summary and body
        new_summary, new_body = extract_summary_from_markdown(merged_content)
        if not new_summary:
            new_summary = target_doc.summary

        new_tokens = count_tokens(new_body)

        # Update document (update_doc already thread-safely updates the index internally)
        now = json_datetime_now()
        try:
            updated_meta = mm.update_doc(target_id, merged_content, summary=new_summary, tokens=new_tokens)
            updated_meta.updated_at = now

            # Update history: add update record
            if updated_meta.history is None:
                updated_meta.history = []
            updated_meta.history.append({
                "event": "updated",
                "source_doc_id": target_id,  # Record the source document ID of the update
                "update_count": updated_meta.update_count,
                "timestamp": now,
            })

            # Update YAML Frontmatter (file content)
            final_merged_content = update_yaml_frontmatter(merged_content, updated_meta)
            mm.write_file(target_doc.path, final_merged_content)

            # Sync update history in index
            mm.update_doc_in_index(target_id, lambda d: d.update({"history": updated_meta.history}))

            logger.info(
                term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
                "[Split] Document update successful: id=%s", target_id
            )

            # Update summary in existing documents list
            for doc_info in existing_docs_list:
                if doc_info["id"] == target_id:
                    doc_info["summary"] = new_summary
                    break

            return updated_meta
        except ValueError as e:
            # Document does not exist: in SPLIT_MERGE stage, creating new document from update is not allowed,
            # otherwise it easily produces overlapping documents with duplicate content.
            if "Document does not exist" in str(e):
                logger.warning(
                    term_color.yellow(
                        "[Split] Target document does not exist, skipping this update (creating new document from update is forbidden): target_id=%s",
                        enabled=True,
                    ),
                    target_id,
                )
                return None
            raise
    except Exception as e:
        logger.error(
            term_color.blue(f"[rewrite thread_id:{thread_id}] ", enabled=True) +
            "[Split] Processing update failed: target_id=%s, error=%s", target_id, e
        )
        logger.exception("[Split] Update processing exception details")
        return None


def _process_MERGE_group(
    group: Dict[str, Any],
    all_docs: List[DocMeta],
    llm: LLMClient,
    cfg: InfiniMemoryConfig,
    mm: MemoryManager,
    idx_group: int,
    total_groups: int,
) -> Optional[Dict[str, Any]]:
    """Process document merge for a single merge group.

    Returns Dict containing MERGEd_doc_ids (list of merged document IDs), tokens_before (total tokens before merge), tokens_after (total tokens after merge) or None (on failure).
    """
    import threading
    logger = logging.getLogger("infini_memory_classic")
    # Get thread ID from current thread name (e.g., "MERGEWorker_0" -> 0)
    thread_name = threading.current_thread().name
    thread_id = 0
    if "_" in thread_name:
        try:
            thread_id = int(thread_name.split("_")[-1])
        except (ValueError, IndexError):
            thread_id = 0

    doc_ids = group.get("doc_ids", [])
    reason = group.get("reason", "")

    logger.info(
        term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
        "[MERGE] Processing merge group %d/%d: doc_ids=%s, reason=%s",
        idx_group + 1, total_groups, doc_ids, reason
    )

    if not doc_ids or len(doc_ids) < 2:
        logger.warning(
            term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
            "[MERGE] Skipping invalid group: doc_ids count < 2"
        )
        return None

    try:
        # Find document objects
        docs_to_MERGE = []
        for doc_id in doc_ids:
            doc = next((d for d in all_docs if d.id == doc_id), None)
            if doc:
                docs_to_MERGE.append(doc)
            else:
                logger.warning(
                    term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
                    "[MERGE] Document does not exist: id=%s", doc_id
                )

        if len(docs_to_MERGE) < 2:
            logger.warning(
                term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
                "[MERGE] Valid document count < 2, skipping merge"
            )
            return None

        # Calculate total tokens before merge
        tokens_before = sum(d.tokens for d in docs_to_MERGE)

        # Sort by updated_at (from oldest to newest)
        docs_to_MERGE.sort(key=lambda d: d.updated_at or "")

        # Read all document content (remove YAML Frontmatter, keep body only)
        docs_content = []
        for doc in docs_to_MERGE:
            try:
                doc_text = mm.read_file(doc.path)
                _, body = extract_summary_from_markdown(doc_text)
                docs_content.append(body)
            except Exception as e:
                logger.warning(
                    term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
                    "[MERGE] Failed to read document: id=%s, error=%s", doc.id, e
                )
                return None

        # Concatenate all document content, separated by delimiters
        MERGEd_input = "\n\n---\n\n".join(docs_content)

        # Calculate input content token count
        input_tokens = sum(d.tokens for d in docs_to_MERGE)

        logger.info(
            term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
            "[MERGE] Calling MERGE_DOCS_PROMPT: doc_count=%d, total_tokens=%d",
            len(docs_to_MERGE), input_tokens
        )

        # Call MERGE_DOCS_PROMPT to merge documents (with retry)
        MERGE_prompt = MERGE_DOCS_PROMPT.format(
            summary_length=cfg.memory.summary_length,
            docs_content=MERGEd_input,
        )
        MERGEd_content = _llm_chat_with_retry(
            llm,
            [{"role": "system", "content": MERGE_prompt}],
            cfg.llm.model,
            cfg,
            context="[MERGE]",
        )

        # Calculate MERGE returned content token count
        output_tokens = count_tokens(MERGEd_content)

        logger.info(
            term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
            "[MERGE] MERGE_DOCS_PROMPT returned: tokens=%d",
            output_tokens
        )

        # Extract summary and body
        new_summary, new_body = extract_summary_from_markdown(MERGEd_content)
        if not new_summary:
            # Use first document's summary as default
            new_summary = docs_to_MERGE[0].summary

        new_tokens = count_tokens(new_body)

        # Create new merged document
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import uuid
        ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        MERGEd_doc_id = f"{ts}_{uuid.uuid4()}"
        filename = f"{MERGEd_doc_id}.md"
        path = mm.doc_dir / filename
        now = json_datetime_now()

        # Get earliest created_at and latest updated_at
        earliest_created = min((d.created_at or "" for d in docs_to_MERGE), default="")
        latest_updated = max((d.updated_at or "" for d in docs_to_MERGE), default="")

        # Calculate update_count (take max of all documents + 1)
        max_update_count = max((getattr(d, 'update_count', 0) for d in docs_to_MERGE), default=0)

        MERGEd_meta = DocMeta(
            id=MERGEd_doc_id,
            path=mm._rel(path),
            created_at=earliest_created,
            updated_at=latest_updated,
            tokens=new_tokens,
            summary=new_summary,
            parent_id="CURRENT",
            update_count=max_update_count + 1,
            current_epoch=docs_to_MERGE[0].current_epoch if docs_to_MERGE else 0,
            merged_from=doc_ids,  # Record the source document ID list of the merge
            history=[{
                "event": "merged",
                "source_doc_ids": doc_ids,  # Record the merged source document ID list
                "source_count": len(doc_ids),
                "timestamp": now,
            }],
        )

        final_MERGEd_content = update_yaml_frontmatter(MERGEd_content, MERGEd_meta)
        mm._ensure_dirs()
        mm.write_file(mm._rel(path), final_MERGEd_content)

        # Use thread-safe method to add to index
        mm.add_doc_to_index(MERGEd_meta)
        # Record operation history
        mm.log_event({
            "event": "merged_docs",
            "doc_id": MERGEd_doc_id,
            "source_doc_ids": doc_ids,
            "timestamp": now,
        })

        logger.info(
            term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
            "[MERGE] Merged document created successfully: id=%s, merged %d documents",
            MERGEd_doc_id, len(doc_ids)
        )

        return {
            "MERGEd_doc_ids": doc_ids,
            "tokens_before": tokens_before,
            "tokens_after": new_tokens,
        }

    except Exception as e:
        logger.error(
            term_color.cyan(f"[MERGE thread_id:{thread_id}] ", enabled=True) +
            "[MERGE] Processing merge group failed: doc_ids=%s, error=%s", doc_ids, e
        )
        logger.exception("[MERGE] Merge group processing exception details")
        return None


__all__ = ["InfiniMemory", "MemoryItem", "SearchResult"]
