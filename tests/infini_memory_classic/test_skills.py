"""Tests for infini_memory_classic.skills module."""
import sys
import shutil
from pathlib import Path

import pytest


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _project_root()
sys.path.insert(0, str(ROOT / "src"))

from infini_memory_classic.skills import (
    SkillMeta,
    SkillsManager,
    parse_skill_md,
    build_skill_md,
    generate_skills_from_memory,
    _skill_meta_to_dict,
)
from infini_memory_classic.storage import LocalStorage
from infini_memory_classic.manager import DocMeta


SAMPLE_SKILL_MD = """\
---
name: travel-preferences
description: Track user travel preferences and habits
extraction_hints:
  - Pay attention to airline and hotel preferences
  - Note travel frequency and preferred destinations
metadata:
  generated_at: "2026-06-17T10:00:00+08:00"
  source_doc_count: 5
  trigger: on_demand
  version: 1
---

# Travel Preferences

## When to Use
- User discusses travel plans or preferences

## Key Patterns
- Prefers window seats on flights
- Usually flies JetBlue or Delta
"""


@pytest.fixture
def tmp_skills_dir(tmp_path):
    """Create a temporary skills directory structure."""
    root = tmp_path / "project"
    root.mkdir()
    data_dir = root / "data" / "STORE_test_store" / "USER_alice" / "skills"
    data_dir.mkdir(parents=True)
    return root


@pytest.fixture
def storage(tmp_skills_dir):
    return LocalStorage(tmp_skills_dir)


@pytest.fixture
def skills_manager(tmp_skills_dir, storage):
    return SkillsManager(
        root=tmp_skills_dir,
        data_root="data",
        store="test_store",
        user_id="alice",
        storage=storage,
    )


class TestParseSkillMd:
    def test_parse_with_frontmatter(self):
        fm, body = parse_skill_md(SAMPLE_SKILL_MD)
        assert fm["name"] == "travel-preferences"
        assert fm["description"] == "Track user travel preferences and habits"
        assert len(fm["extraction_hints"]) == 2
        assert "Travel Preferences" in body

    def test_parse_without_frontmatter(self):
        fm, body = parse_skill_md("# Just a body\n\nSome content")
        assert fm == {}
        assert "Just a body" in body

    def test_parse_empty(self):
        fm, body = parse_skill_md("")
        assert fm == {}
        assert body == ""


class TestBuildSkillMd:
    def test_roundtrip(self):
        fm = {"name": "test-skill", "description": "A test skill"}
        body = "# Test\n\nContent here"
        result = build_skill_md(fm, body)
        assert result.startswith("---\n")
        parsed_fm, parsed_body = parse_skill_md(result)
        assert parsed_fm["name"] == "test-skill"
        assert "Content here" in parsed_body


class TestSkillsManagerCRUD:
    def test_list_empty(self, skills_manager):
        assert skills_manager.list_skills() == []

    def test_save_and_get(self, skills_manager):
        meta = skills_manager.save_skill("travel-preferences", SAMPLE_SKILL_MD)
        assert meta.name == "travel-preferences"
        assert meta.description == "Track user travel preferences and habits"
        assert len(meta.extraction_hints) == 2

        retrieved = skills_manager.get_skill("travel-preferences")
        assert retrieved is not None
        assert retrieved.name == "travel-preferences"

    def test_get_nonexistent(self, skills_manager):
        assert skills_manager.get_skill("nonexistent") is None

    def test_list_after_save(self, skills_manager):
        skills_manager.save_skill("travel-preferences", SAMPLE_SKILL_MD)
        skills = skills_manager.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "travel-preferences"

    def test_get_skill_content(self, skills_manager):
        skills_manager.save_skill("travel-preferences", SAMPLE_SKILL_MD)
        content = skills_manager.get_skill_content("travel-preferences")
        assert content is not None
        assert "Travel Preferences" in content

    def test_get_skill_content_nonexistent(self, skills_manager):
        assert skills_manager.get_skill_content("nonexistent") is None

    def test_delete_skill(self, skills_manager):
        skills_manager.save_skill("travel-preferences", SAMPLE_SKILL_MD)
        assert skills_manager.get_skill("travel-preferences") is not None

        skills_manager.delete_skill("travel-preferences")
        assert skills_manager.get_skill("travel-preferences") is None
        assert skills_manager.list_skills() == []

    def test_delete_nonexistent(self, skills_manager):
        skills_manager.delete_skill("nonexistent")

    def test_multiple_skills(self, skills_manager):
        skills_manager.save_skill("travel-preferences", SAMPLE_SKILL_MD)
        other_skill = build_skill_md(
            {"name": "food-preferences", "description": "Track food prefs",
             "extraction_hints": ["dietary restrictions"]},
            "# Food\n\nContent",
        )
        skills_manager.save_skill("food-preferences", other_skill)

        skills = skills_manager.list_skills()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"travel-preferences", "food-preferences"}


class TestExtractionHints:
    def test_empty_when_no_skills(self, skills_manager):
        assert skills_manager.get_extraction_hints() == []

    def test_collects_hints_from_all_skills(self, skills_manager):
        skills_manager.save_skill("travel-preferences", SAMPLE_SKILL_MD)
        other_skill = build_skill_md(
            {"name": "food-prefs", "description": "Food",
             "extraction_hints": ["dietary restrictions", "favorite cuisines"]},
            "# Food",
        )
        skills_manager.save_skill("food-prefs", other_skill)

        hints = skills_manager.get_extraction_hints()
        assert len(hints) == 4
        assert "Pay attention to airline and hotel preferences" in hints
        assert "dietary restrictions" in hints


class TestSkillMetaToDict:
    def test_conversion(self):
        meta = SkillMeta(
            name="test", description="desc", path="some/path",
            extraction_hints=["h1"], generated_at="2026-01-01",
            source_doc_count=3, version=1,
        )
        d = _skill_meta_to_dict(meta)
        assert d["name"] == "test"
        assert d["extraction_hints"] == ["h1"]


class TestGetSkillsDirAbs:
    def test_returns_correct_path(self, skills_manager, tmp_skills_dir):
        expected = tmp_skills_dir / "data" / "STORE_test_store" / "USER_alice" / "skills"
        assert skills_manager.get_skills_dir_abs() == expected
