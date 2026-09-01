from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mem_flow import (
    ChatMessage,
    ExtractionRequest,
    FlowMetrics,
    MemFlow,
    MemFlowConfig,
    StorageConfig,
)
from mem_flow.storage import LocalObjectStore


class UnexpectedLLM:
    def complete(self, _request):
        raise AssertionError("infer=False must not call the LLM")


def test_mem_flow_writes_to_configured_local_directory(tmp_path: Path) -> None:
    flow = MemFlow.create(
        MemFlowConfig(
            storage=StorageConfig(type="local", path=tmp_path),
        ),
        store_id="notes",
        user_id="alice",
        instance_id="worker-1",
        llm=UnexpectedLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )

    result = flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="Alice likes tea.")],
            infer=False,
        )
    )

    expected = tmp_path / "STORE_notes/USER_alice/current/CURRENT_worker-1.md"
    assert result.current_key == "current/CURRENT_worker-1.md"
    assert expected.is_file()
    assert "Alice likes tea." in expected.read_text(encoding="utf-8")
    assert flow.delete_all() == 2
    assert not expected.exists()


def test_local_storage_defaults_to_project_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    flow = MemFlow.create(
        MemFlowConfig(storage=StorageConfig(type="local")),
        store_id="default",
        user_id="user",
        instance_id="worker",
        llm=UnexpectedLLM(),
        metrics=FlowMetrics.create(enabled=False),
    )
    flow.extract(
        ExtractionRequest(
            messages=[ChatMessage(role="user", content="Local memory")],
            infer=False,
        )
    )

    assert (
        tmp_path / "data/mem_flow/STORE_default/USER_user/current/CURRENT_worker.md"
    ).is_file()


def test_local_object_store_rejects_paths_outside_root(tmp_path: Path) -> None:
    store = LocalObjectStore.create(tmp_path)

    with pytest.raises(ValueError, match="unsafe local object key"):
        store.put_text("../outside.md", "secret")
    with pytest.raises(ValueError, match="unsafe local object key"):
        store.get_text("/absolute.md")


def test_s3_remains_the_default_storage_type() -> None:
    with pytest.raises(ValidationError, match="s3 configuration is required"):
        MemFlowConfig()
