import sys
import logging
from pathlib import Path
import json


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


def _make_mock_caller():
    def caller(*, messages, model, temperature):
        sys_msg = ""
        for m in messages:
            if m.get("role") == "system":
                sys_msg = m.get("content", "")
                break

        # SEARCH_MEMORY_PROMPT
        if "Select documents relevant" in sys_msg:
            return json.dumps({"ids": []})

        # ANSWER_WITH_CONTEXT_PROMPT
        if "expert at answering questions" in sys_msg:
            return "Mock answer"

        return "ok"

    return caller


def test_search():
    InfiniMemoryConfig(root=ROOT)
    logger = logging.getLogger("infini_memory.tests")
    logger.info("Starting InfiniMemory search test (mocked LLM)")

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

    mem = InfiniMemory()
    llm = LLMClient(caller=_make_mock_caller())

    query = "User flew from San Francisco to where?"
    out = mem.search(query, user_id="test_user", cfg=cfg, llm=llm)
    logger.info(f"query={query} out={out}")

    # Test answer generation with context
    answer = llm.chat(
        [
            {"role": "system", "content": "You are an expert at answering questions based on provided content."},
            {"role": "user", "content": json.dumps({"query": query, "context": ""})},
        ],
        model=cfg.llm.model,
    )
    assert answer == "Mock answer"
