"""Tests for S3Storage backend against a local SeaweedFS endpoint.

These tests require:
  - boto3 installed: pip install boto3
  - A running S3-compatible endpoint (e.g., SeaweedFS)

Tests are skipped if boto3 is not installed or the endpoint is unreachable.
"""

import sys
import time
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

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

S3_ENDPOINT = "http://seaweedfs.local-db-tools.svc.cluster.local:8333"
TEST_BUCKET = f"test-infini-memory-{int(time.time())}"


def _s3_reachable() -> bool:
    if not HAS_BOTO3:
        return False
    try:
        import urllib.request
        urllib.request.urlopen(S3_ENDPOINT + "/", timeout=3)
        return True
    except Exception:
        return False


skip_no_s3 = pytest.mark.skipif(
    not _s3_reachable(),
    reason="S3 endpoint not reachable or boto3 not installed",
)


@pytest.fixture
def s3_storage():
    from infini_memory_classic.s3_storage import S3Storage
    storage = S3Storage(
        endpoint=S3_ENDPOINT,
        bucket=TEST_BUCKET,
        access_key="",
        secret_key="",
    )
    yield storage
    # Cleanup: remove all objects and delete bucket
    try:
        paginator = storage.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=TEST_BUCKET):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                storage.client.delete_objects(
                    Bucket=TEST_BUCKET,
                    Delete={"Objects": objects},
                )
        storage.client.delete_bucket(Bucket=TEST_BUCKET)
    except Exception:
        pass


@skip_no_s3
def test_s3_write_read_roundtrip(s3_storage) -> None:
    s3_storage.write_text("test/hello.txt", "hello world")
    assert s3_storage.read_text("test/hello.txt") == "hello world"


@skip_no_s3
def test_s3_write_overwrite(s3_storage) -> None:
    s3_storage.write_text("test/overwrite.txt", "version1")
    s3_storage.write_text("test/overwrite.txt", "version2")
    assert s3_storage.read_text("test/overwrite.txt") == "version2"


@skip_no_s3
def test_s3_delete(s3_storage) -> None:
    s3_storage.write_text("test/to_delete.txt", "bye")
    assert s3_storage.exists("test/to_delete.txt")
    s3_storage.delete("test/to_delete.txt")
    assert not s3_storage.exists("test/to_delete.txt")


@skip_no_s3
def test_s3_delete_missing_ok(s3_storage) -> None:
    s3_storage.delete("test/nonexistent.txt", missing_ok=True)


@skip_no_s3
def test_s3_exists(s3_storage) -> None:
    assert not s3_storage.exists("test/no_such_file.txt")
    s3_storage.write_text("test/exists.txt", "content")
    assert s3_storage.exists("test/exists.txt")


@skip_no_s3
def test_s3_exists_directory(s3_storage) -> None:
    s3_storage.write_text("test/dir/file.txt", "content")
    assert s3_storage.exists("test/dir")


@skip_no_s3
def test_s3_glob(s3_storage) -> None:
    s3_storage.write_text("docs/file1.md", "one")
    s3_storage.write_text("docs/file2.md", "two")
    s3_storage.write_text("docs/file3.txt", "three")

    md_files = sorted(s3_storage.glob("docs", "*.md"))
    assert len(md_files) == 2
    assert all(f.endswith(".md") for f in md_files)


@skip_no_s3
def test_s3_glob_empty(s3_storage) -> None:
    assert s3_storage.glob("empty_dir", "*.md") == []


@skip_no_s3
def test_s3_rmtree(s3_storage) -> None:
    s3_storage.write_text("tree/a.txt", "a")
    s3_storage.write_text("tree/sub/b.txt", "b")
    assert s3_storage.exists("tree/a.txt")

    s3_storage.rmtree("tree")
    assert not s3_storage.exists("tree/a.txt")
    assert not s3_storage.exists("tree/sub/b.txt")


@skip_no_s3
def test_s3_listdir(s3_storage) -> None:
    s3_storage.write_text("parent/file.txt", "content")
    s3_storage.write_text("parent/subdir/child.txt", "child")

    entries = s3_storage.listdir("parent")
    assert "file.txt" in entries
    assert "subdir" in entries


@skip_no_s3
def test_s3_mkdir_noop(s3_storage) -> None:
    s3_storage.mkdir("test/dir")


@skip_no_s3
def test_s3_is_dir(s3_storage) -> None:
    s3_storage.write_text("test/file.txt", "content")
    s3_storage.write_text("test/dir/child.txt", "child")

    assert not s3_storage.is_dir("test/file.txt")
    assert s3_storage.is_dir("test/dir")


@skip_no_s3
def test_s3_bucket_auto_create() -> None:
    from infini_memory_classic.s3_storage import S3Storage
    unique_bucket = f"test-auto-create-{int(time.time())}"
    storage = S3Storage(
        endpoint=S3_ENDPOINT,
        bucket=unique_bucket,
        access_key="",
        secret_key="",
    )
    storage.write_text("test.txt", "works")
    assert storage.read_text("test.txt") == "works"
    # Cleanup
    try:
        storage.client.delete_object(Bucket=unique_bucket, Key="test.txt")
        storage.client.delete_bucket(Bucket=unique_bucket)
    except Exception:
        pass


@skip_no_s3
def test_s3_chinese_content(s3_storage) -> None:
    content = "# 用户偏好\n\n- 喜欢蓝莓和草莓\n- 名字是 Jay"
    s3_storage.write_text("test/chinese.md", content)
    assert s3_storage.read_text("test/chinese.md") == content


@skip_no_s3
def test_s3_large_content(s3_storage) -> None:
    content = "x" * 100_000
    s3_storage.write_text("test/large.txt", content)
    assert s3_storage.read_text("test/large.txt") == content
