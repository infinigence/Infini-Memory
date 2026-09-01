import logging
import sys
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _project_root()
sys.path.insert(0, str(ROOT / "src"))

from infini_memory_classic.llm import LLMClient  # noqa: E402


def test_openai_simple_chat():
    """Test the LLM call path using a custom caller (no real API calls)."""
    logger = logging.getLogger("infini_memory_classic.tests")
    logger.info("Starting LLM chat test (mocked)")

    def fake_caller(*, messages, model, temperature):
        return "pong"

    llm = LLMClient(caller=fake_caller)
    content = llm.chat(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'pong' only."},
        ],
        model="mock-model",
    ).strip().lower()

    logger.info("LLM response: %s", content)
    assert "pong" in content
