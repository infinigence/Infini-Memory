"""Object storage adapters used by mem_flow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .config import MemFlowConfig, S3Config
from .models import MemoryScope
from .observability import FlowMetrics


@runtime_checkable
class ObjectStore(Protocol):
    """Small storage seam shared by the S3 and local backends."""

    def get_text(self, key: str) -> str: ...
    def put_text(self, key: str, content: str) -> None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str) -> list[str]: ...


class ScopedObjectStore(BaseModel):
    """Expose one user's object namespace through relative keys only.

    Workflow components never receive the unscoped physical key space.  The adapter
    validates every relative key before adding the physical user-root prefix,
    and strips that prefix from list results before returning them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: ObjectStore
    root_prefix: str

    @classmethod
    def create(
        cls,
        backend: ObjectStore,
        *,
        fixed_prefix: str,
        scope: MemoryScope,
    ) -> "ScopedObjectStore":
        fixed = fixed_prefix.strip("/")
        scope_root = f"STORE_{scope.store_id}/USER_{scope.user_id}"
        root = f"{fixed}/{scope_root}" if fixed else scope_root
        return cls(backend=backend, root_prefix=root)

    @staticmethod
    def _validate_relative_key(key: str, *, allow_empty: bool = False) -> str:
        value = key.strip("/")
        if not value and allow_empty:
            return ""
        if (
            not value
            or key.startswith("/")
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError(f"unsafe scoped object key: {key!r}")
        return value

    def _physical_key(self, key: str) -> str:
        relative = self._validate_relative_key(key)
        return f"{self.root_prefix}/{relative}"

    def get_text(self, key: str) -> str:
        return self.backend.get_text(self._physical_key(key))

    def put_text(self, key: str, content: str) -> None:
        self.backend.put_text(self._physical_key(key), content)

    def delete(self, key: str) -> None:
        self.backend.delete(self._physical_key(key))

    def exists(self, key: str) -> bool:
        return self.backend.exists(self._physical_key(key))

    def list_keys(self, prefix: str) -> list[str]:
        relative_prefix = self._validate_relative_key(prefix, allow_empty=True)
        physical_prefix = (
            f"{self.root_prefix}/{relative_prefix}"
            if relative_prefix
            else f"{self.root_prefix}/"
        )
        root = self.root_prefix + "/"
        relative_keys: list[str] = []
        for key in self.backend.list_keys(physical_prefix):
            if not key.startswith(root):
                raise ValueError(f"object store returned key outside scope: {key!r}")
            relative_keys.append(key[len(root) :])
        return sorted(relative_keys)


class LocalObjectStore(BaseModel):
    """Store objects as UTF-8 files below one local root directory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path

    @classmethod
    def create(cls, root: str | Path = Path("data/mem_flow")) -> "LocalObjectStore":
        resolved = Path(root).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return cls(root=resolved)

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.storage")

    @staticmethod
    def _validate_key(key: str, *, allow_empty: bool = False) -> str:
        value = key.strip("/")
        if not value and allow_empty:
            return ""
        if (
            not value
            or key.startswith("/")
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError(f"unsafe local object key: {key!r}")
        return value

    def _path(self, key: str, *, allow_empty: bool = False) -> Path:
        relative = self._validate_key(key, allow_empty=allow_empty)
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"local object key escapes storage root: {key!r}")
        return candidate

    def get_text(self, key: str) -> str:
        path = self._path(key)
        self.logger.debug("local_get path=%s", path)
        return path.read_text(encoding="utf-8")

    def put_text(self, key: str, content: str) -> None:
        path = self._path(key)
        self.logger.debug(
            "local_put path=%s bytes=%d", path, len(content.encode("utf-8"))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def delete(self, key: str) -> None:
        path = self._path(key)
        self.logger.debug("local_delete path=%s", path)
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str) -> list[str]:
        directory = self._path(prefix, allow_empty=True)
        if not directory.exists():
            return []
        candidates = [directory] if directory.is_file() else directory.rglob("*")
        keys: list[str] = []
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                raise ValueError(f"local storage entry escapes root: {path}")
            keys.append(path.relative_to(self.root).as_posix())
        return sorted(keys)


class S3ObjectStore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: S3Config
    metrics: FlowMetrics
    client: object

    @classmethod
    def create(
        cls,
        config: S3Config,
        metrics: FlowMetrics,
        *,
        client: object | None = None,
    ) -> "S3ObjectStore":
        if client is None:
            try:
                import boto3
                from botocore import UNSIGNED
                from botocore.config import Config as BotoConfig
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required by mem_flow's S3 backend"
                ) from exc
            access_key = config.access_key.get_secret_value()
            secret_key = config.secret_key.get_secret_value()
            signature_version: str = "s3v4" if access_key and secret_key else UNSIGNED
            kwargs: dict[str, object] = {
                "endpoint_url": config.endpoint_url,
                "region_name": config.region,
                "config": BotoConfig(
                    signature_version=signature_version,
                    s3={"addressing_style": "path"},
                ),
            }
            if access_key and secret_key:
                kwargs.update(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
            client = boto3.client("s3", **kwargs)
        store = cls(config=config, metrics=metrics, client=client)
        if config.ensure_bucket:
            store._ensure_bucket()
        return store

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.storage")

    def _record(self, operation: str, status: str) -> None:
        self.metrics.s3_total.labels(operation=operation, status=status).inc()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.config.bucket)
            self._record("head_bucket", "success")
        except Exception:
            self.logger.info("s3_bucket_create bucket=%s", self.config.bucket)
            self.client.create_bucket(Bucket=self.config.bucket)
            self._record("create_bucket", "success")

    def get_text(self, key: str) -> str:
        self.logger.debug("s3_get bucket=%s key=%s", self.config.bucket, key)
        try:
            response = self.client.get_object(Bucket=self.config.bucket, Key=key)
            value = response["Body"].read().decode("utf-8")
        except Exception:
            self._record("get", "error")
            self.logger.exception(
                "s3_get_failed bucket=%s key=%s", self.config.bucket, key
            )
            raise
        self._record("get", "success")
        return value

    def put_text(self, key: str, content: str) -> None:
        encoded = content.encode("utf-8")
        self.logger.debug(
            "s3_put bucket=%s key=%s bytes=%d", self.config.bucket, key, len(encoded)
        )
        try:
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=encoded,
                ContentType="text/markdown; charset=utf-8",
            )
        except Exception:
            self._record("put", "error")
            self.logger.exception(
                "s3_put_failed bucket=%s key=%s", self.config.bucket, key
            )
            raise
        self._record("put", "success")

    def delete(self, key: str) -> None:
        self.logger.debug("s3_delete bucket=%s key=%s", self.config.bucket, key)
        try:
            self.client.delete_object(Bucket=self.config.bucket, Key=key)
        except Exception:
            self._record("delete", "error")
            self.logger.exception(
                "s3_delete_failed bucket=%s key=%s", self.config.bucket, key
            )
            raise
        self._record("delete", "success")

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.config.bucket, Key=key)
        except Exception:
            self._record("head", "miss")
            return False
        self._record("head", "success")
        return True

    def list_keys(self, prefix: str) -> list[str]:
        normalized = prefix.rstrip("/") + "/"
        self.logger.debug("s3_list bucket=%s prefix=%s", self.config.bucket, normalized)
        keys: list[str] = []
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=self.config.bucket, Prefix=normalized
            ):
                keys.extend(str(item["Key"]) for item in page.get("Contents", []))
        except Exception:
            self._record("list", "error")
            self.logger.exception(
                "s3_list_failed bucket=%s prefix=%s", self.config.bucket, normalized
            )
            raise
        self._record("list", "success")
        return sorted(keys)


def create_object_store(config: MemFlowConfig, metrics: FlowMetrics) -> ObjectStore:
    """Create the backend selected by ``MemFlowConfig.storage``."""
    if config.storage.type == "local":
        return LocalObjectStore.create(config.storage.path)
    assert config.s3 is not None
    return S3ObjectStore.create(config.s3, metrics)


__all__ = [
    "LocalObjectStore",
    "ObjectStore",
    "S3ObjectStore",
    "ScopedObjectStore",
    "create_object_store",
]
