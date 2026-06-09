import sys
import logging
import json
import shutil
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _project_root()
sys.path.insert(0, str(ROOT / "src"))

from infini_memory import InfiniMemory, InfiniMemoryConfig  # noqa: E402
from infini_memory.llm import LLMClient  # noqa: E402


test_chunk1 = "User: I flew JetBlue from San Francisco to Boston on a red-eye and slept the whole way."
test_chunk2 = "User: I had an issue with American Airlines IFE on Feb 10 from New York to Los Angeles."


def _make_mock_caller():
    """Return a callable used as LLMClient(caller=...). Routes by prompt content."""

    def caller(*, messages, model, temperature):
        sys_msg = ""
        user_msg = ""
        for m in messages:
            if m.get("role") == "system":
                sys_msg = m.get("content", "")
            elif m.get("role") == "user":
                user_msg = m.get("content", "")

        # EXTRACT_MEMORY_PROMPT -> return Markdown with YAML frontmatter
        if "personal information organizer" in sys_msg or "Extract relevant information" in sys_msg:
            return (
                "---\n"
                "summary: travel; user travel facts\n"
                "---\n"
                "# Travel\n\n"
                "- <seq=@@SEQ@@,time=empty> User mentioned a flight\n"
            )

        # REWRITE_CURRENT_PROMPT
        if "CURRENT document" in sys_msg and "aggregate" in sys_msg.lower():
            return (
                "# Travel\n\n- <seq=1,time=empty> User mentioned a flight\n"
            )

        # PLAN_UPDATE_PROMPT -> JSON
        if "updates" in sys_msg and "new_docs" in sys_msg:
            return json.dumps({"updates": [], "new_docs": []})

        # SEARCH_MEMORY_PROMPT -> ids JSON
        if "Select documents relevant" in sys_msg:
            return json.dumps({"ids": []})

        # SELECT_MERGE_GROUPS_PROMPT
        if "groups" in sys_msg and "doc_ids" in sys_msg:
            return json.dumps({"groups": []})

        # MERGE_DOCS_PROMPT
        if "Merge all documents" in sys_msg:
            return "---\nsummary: merged\n---\n# Merged\n"

        # ANSWER_WITH_CONTEXT_PROMPT or generic
        return "ok"

    return caller


def test_add():
    InfiniMemoryConfig(root=ROOT)
    logger = logging.getLogger("infini_memory.tests")
    logger.info("Starting InfiniMemory add test (mocked LLM)")

    cfg = InfiniMemoryConfig(root=ROOT)
    cfg.memory.enabled = True
    cfg.memory.data_root = "test_data"
    cfg.llm.openai_api_key = "mock-key"
    cfg.llm.model = "mock-model"
    cfg.llm.retry_max_attempts = 1
    cfg.llm.retry_initial_wait = 0.0
    cfg.llm.retry_max_wait = 0.0
    cfg.llm.retry_jitter = 0.0
    cfg.log.level = "INFO"
    cfg._setup_logging()

    td = ROOT / cfg.memory.data_root
    if td.exists():
        shutil.rmtree(td)

    mem = InfiniMemory()
    llm = LLMClient(caller=_make_mock_caller())

    mem.add(test_chunk1, user_id="test_user", cfg=cfg, llm=llm)
    mem.add(test_chunk2, user_id="test_user", cfg=cfg, llm=llm)

    logger.info("Add completed (mocked)")
