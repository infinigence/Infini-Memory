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


def test_llm_client_retries_on_any_error(monkeypatch):
    calls = {"n": 0}

    def flaky_caller(*, messages, model, temperature):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    llm = LLMClient(
        caller=flaky_caller,
        retry_max_attempts=5,
        retry_initial_wait=0.0,
        retry_max_wait=0.0,
        retry_jitter=0.0,
    )

    # Avoid actual sleep
    monkeypatch.setattr("infini_memory_classic.llm.time.sleep", lambda _: None)

    out = llm.chat([{"role": "user", "content": "hi"}], model="x", temperature=0.1)
    assert out == "ok"
    assert calls["n"] == 3


def test_llm_client_raises_after_max_attempts(monkeypatch):
    calls = {"n": 0}

    def always_fail(*, messages, model, temperature):
        calls["n"] += 1
        raise RuntimeError("nope")

    llm = LLMClient(
        caller=always_fail,
        retry_max_attempts=3,
        retry_initial_wait=0.0,
        retry_max_wait=0.0,
        retry_jitter=0.0,
    )
    monkeypatch.setattr("infini_memory_classic.llm.time.sleep", lambda _: None)

    try:
        llm.chat([{"role": "user", "content": "hi"}], model="x", temperature=0.1)
        assert False, "expected exception"
    except RuntimeError as e:
        assert "LLM call failed" in str(e)
    assert calls["n"] == 3

