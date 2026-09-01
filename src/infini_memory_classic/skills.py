"""Skills management and generation for Infini Memory.

Provides SkillsManager for CRUD operations on per-user skills, and
generate_skills_from_memory() for LLM-driven skill generation from
accumulated memory documents.

Skills are stored as standard DeepAgents SKILL.md files at
``data/<user_id>/skills/<skill-name>/SKILL.md``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

if TYPE_CHECKING:
    from .config import InfiniMemoryConfig
    from .llm import LLMClient
    from .manager import DocMeta, MemoryManager
    from .storage import StorageBackend

logger = logging.getLogger("infini_memory_classic")


@dataclass
class SkillMeta:
    name: str
    description: str
    path: str
    extraction_hints: List[str] = field(default_factory=list)
    generated_at: str = ""
    source_doc_count: int = 0
    version: int = 1


def parse_skill_md(text: str) -> tuple[dict, str]:
    """Parse a SKILL.md file into (frontmatter_dict, body_str)."""
    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = parts[2].strip()
            return frontmatter, body
    return {}, text


def build_skill_md(frontmatter: dict, body: str) -> str:
    """Build a SKILL.md file from frontmatter dict and body string."""
    fm_str = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False).strip()
    return f"---\n{fm_str}\n---\n\n{body}\n"


class SkillsManager:
    """Manage per-user skills at ``data/<store>/<user_id>/skills/``."""

    def __init__(
        self,
        root: Path,
        data_root: str,
        store: str,
        user_id: str,
        storage: "StorageBackend",
        skills_dir: str = "skills",
    ):
        self.root = Path(root)
        self.store = store
        self.user_id = user_id
        self.data_dir = self.root / data_root / f"STORE_{store}" / f"USER_{user_id}"
        self.skills_dir_path = self.data_dir / skills_dir
        self.storage = storage

    def _rel(self, abs_path: Path) -> str:
        return str(abs_path.relative_to(self.root))

    def _skill_md_path(self, name: str) -> Path:
        return self.skills_dir_path / name / "SKILL.md"

    def get_skills_dir_abs(self) -> Path:
        return self.skills_dir_path

    def list_skills(self) -> List[SkillMeta]:
        rel = self._rel(self.skills_dir_path)
        if not self.storage.exists(rel):
            return []

        skills: List[SkillMeta] = []
        for entry_name in self.storage.listdir(rel):
            entry_rel = self._rel(self.skills_dir_path / entry_name)
            if not self.storage.is_dir(entry_rel):
                continue
            skill_md_rel = self._rel(self._skill_md_path(entry_name))
            if not self.storage.exists(skill_md_rel):
                continue
            try:
                content = self.storage.read_text(skill_md_rel)
                fm, _ = parse_skill_md(content)
                meta = fm.get("metadata", {}) or {}
                skills.append(SkillMeta(
                    name=fm.get("name", entry_name),
                    description=fm.get("description", ""),
                    path=skill_md_rel,
                    extraction_hints=fm.get("extraction_hints", []) or [],
                    generated_at=str(meta.get("generated_at", "")),
                    source_doc_count=int(meta.get("source_doc_count", 0)),
                    version=int(meta.get("version", 1)),
                ))
            except Exception as e:
                logger.warning("Failed to read skill %s: %s", entry_name, e)
        return skills

    def get_skill(self, name: str) -> Optional[SkillMeta]:
        skill_md_rel = self._rel(self._skill_md_path(name))
        if not self.storage.exists(skill_md_rel):
            return None
        try:
            content = self.storage.read_text(skill_md_rel)
            fm, _ = parse_skill_md(content)
            meta = fm.get("metadata", {}) or {}
            return SkillMeta(
                name=fm.get("name", name),
                description=fm.get("description", ""),
                path=skill_md_rel,
                extraction_hints=fm.get("extraction_hints", []) or [],
                generated_at=str(meta.get("generated_at", "")),
                source_doc_count=int(meta.get("source_doc_count", 0)),
                version=int(meta.get("version", 1)),
            )
        except Exception as e:
            logger.warning("Failed to read skill %s: %s", name, e)
            return None

    def get_skill_content(self, name: str) -> Optional[str]:
        skill_md_rel = self._rel(self._skill_md_path(name))
        if not self.storage.exists(skill_md_rel):
            return None
        return self.storage.read_text(skill_md_rel)

    def save_skill(self, name: str, content: str) -> SkillMeta:
        skill_md_rel = self._rel(self._skill_md_path(name))
        self.storage.write_text(skill_md_rel, content)
        fm, _ = parse_skill_md(content)
        meta = fm.get("metadata", {}) or {}
        return SkillMeta(
            name=fm.get("name", name),
            description=fm.get("description", ""),
            path=skill_md_rel,
            extraction_hints=fm.get("extraction_hints", []) or [],
            generated_at=str(meta.get("generated_at", "")),
            source_doc_count=int(meta.get("source_doc_count", 0)),
            version=int(meta.get("version", 1)),
        )

    def delete_skill(self, name: str) -> None:
        skill_dir_rel = self._rel(self.skills_dir_path / name)
        if self.storage.exists(skill_dir_rel):
            self.storage.rmtree(skill_dir_rel)

    def get_extraction_hints(self) -> List[str]:
        """Collect extraction_hints from all skills for this user."""
        hints: List[str] = []
        for skill in self.list_skills():
            hints.extend(skill.extraction_hints)
        return hints


def _skill_meta_to_dict(meta: SkillMeta) -> dict:
    return {
        "name": meta.name,
        "description": meta.description,
        "path": meta.path,
        "extraction_hints": meta.extraction_hints,
        "generated_at": meta.generated_at,
        "source_doc_count": meta.source_doc_count,
        "version": meta.version,
    }


def generate_skills_from_memory(
    docs: List["DocMeta"],
    existing_skills: List[SkillMeta],
    cfg: "InfiniMemoryConfig",
    llm: "LLMClient",
    mm: "MemoryManager",
    sm: SkillsManager,
    *,
    trigger: str = "on_demand",
    max_skills: int = 5,
) -> List[SkillMeta]:
    """Analyze memory documents and generate/update skills.

    Two-phase LLM process:
    1. Planning: send doc summaries + existing skills to GENERATE_SKILLS_PROMPT
    2. Writing: for each planned skill, read source docs and call WRITE_SKILL_PROMPT

    Returns list of newly created/updated SkillMeta.
    """
    from .prompts import GENERATE_SKILLS_PROMPT, WRITE_SKILL_PROMPT

    if not docs:
        logger.info("[Skills] No documents available for skill generation")
        return []

    non_current_docs = [d for d in docs if d.id != "CURRENT"]
    if len(non_current_docs) < 3:
        logger.info("[Skills] Too few documents (%d) for skill generation, need at least 3", len(non_current_docs))
        return []

    doc_summaries = [
        {"id": d.id, "summary": d.summary, "tokens": d.tokens}
        for d in non_current_docs
        if d.summary
    ]

    existing_skill_list = [
        {"name": s.name, "description": s.description}
        for s in existing_skills
    ]

    max_allowed = cfg.memory.skills_max_per_user if hasattr(cfg.memory, "skills_max_per_user") else 20
    remaining_slots = max(0, max_allowed - len(existing_skills))
    if remaining_slots == 0:
        logger.info("[Skills] Max skills per user (%d) reached, skipping generation", max_allowed)
        return []

    effective_max = min(max_skills, remaining_slots)

    plan_prompt = GENERATE_SKILLS_PROMPT.format(
        doc_summaries=json.dumps(doc_summaries, ensure_ascii=False),
        existing_skills=json.dumps(existing_skill_list, ensure_ascii=False),
        max_skills=effective_max,
    )

    logger.info("[Skills] Generating skill plan from %d documents", len(doc_summaries))

    plan_response = llm.chat(
        [
            {"role": "system", "content": plan_prompt},
            {"role": "user", "content": "Analyze the documents and generate a skill plan."},
        ],
        model=cfg.llm.model,
    )

    try:
        plan_text = plan_response.strip()
        if plan_text.startswith("```"):
            lines = plan_text.split("\n")
            plan_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        plan = json.loads(plan_text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("[Skills] Failed to parse skill plan: %s", e)
        return []

    skill_plans = plan.get("skills", [])
    if not skill_plans:
        logger.info("[Skills] No skills proposed by LLM")
        return []

    import datetime
    from zoneinfo import ZoneInfo
    now = datetime.datetime.now(datetime.UTC).astimezone(
        ZoneInfo(cfg.common.timezone)
    ).isoformat()

    created_skills: List[SkillMeta] = []

    for skill_plan in skill_plans[:effective_max]:
        name = skill_plan.get("name", "").strip()
        if not name:
            continue

        description = skill_plan.get("description", "")
        extraction_hints = skill_plan.get("extraction_hints", [])
        source_doc_ids = skill_plan.get("source_doc_ids", [])

        source_contents: List[str] = []
        for doc_id in source_doc_ids:
            try:
                content = mm.get_doc_content(doc_id)
                if content:
                    source_contents.append(f"## Document: {doc_id}\n\n{content}")
            except Exception:
                pass

        if not source_contents:
            logger.warning("[Skills] No source content found for skill '%s', skipping", name)
            continue

        write_prompt = WRITE_SKILL_PROMPT.format(
            skill_name=name,
            skill_description=description,
            extraction_hints=json.dumps(extraction_hints, ensure_ascii=False),
            source_content="\n\n---\n\n".join(source_contents),
        )

        try:
            skill_md = llm.chat(
                [
                    {"role": "system", "content": write_prompt},
                    {"role": "user", "content": "Generate the complete SKILL.md content."},
                ],
                model=cfg.llm.model,
            )
        except Exception as e:
            logger.warning("[Skills] Failed to generate skill '%s': %s", name, e)
            continue

        skill_md = skill_md.strip()
        if skill_md.startswith("```"):
            lines = skill_md.split("\n")
            skill_md = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        fm, body = parse_skill_md(skill_md)
        if not fm.get("name"):
            fm["name"] = name
        if not fm.get("description"):
            fm["description"] = description
        if "extraction_hints" not in fm:
            fm["extraction_hints"] = extraction_hints
        if "metadata" not in fm:
            fm["metadata"] = {}
        fm["metadata"]["generated_at"] = now
        fm["metadata"]["source_doc_count"] = len(source_doc_ids)
        fm["metadata"]["trigger"] = trigger
        fm["metadata"]["version"] = 1

        existing = sm.get_skill(name)
        if existing:
            fm["metadata"]["version"] = existing.version + 1

        final_content = build_skill_md(fm, body)

        try:
            meta = sm.save_skill(name, final_content)
            created_skills.append(meta)
            logger.info("[Skills] %s skill '%s' (sources: %d docs)",
                        "Updated" if existing else "Created", name, len(source_doc_ids))
        except Exception as e:
            logger.warning("[Skills] Failed to save skill '%s': %s", name, e)

    logger.info("[Skills] Generation complete: %d skills created/updated", len(created_skills))
    return created_skills


__all__ = [
    "SkillMeta",
    "SkillsManager",
    "parse_skill_md",
    "build_skill_md",
    "generate_skills_from_memory",
]
