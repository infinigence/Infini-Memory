from __future__ import annotations

import fnmatch
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class StorageBackend(ABC):
    """Abstract interface for file storage operations.

    All paths are relative strings (e.g., "data/alice/doc/xxx.md").
    The backend resolves them to actual storage locations.
    """

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read text content from a file."""
        ...

    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        """Write text content to a file. Creates parent directories as needed."""
        ...

    @abstractmethod
    def delete(self, path: str, missing_ok: bool = True) -> None:
        """Delete a file."""
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        ...

    @abstractmethod
    def mkdir(self, path: str) -> None:
        """Ensure a directory exists (like mkdir -p)."""
        ...

    @abstractmethod
    def glob(self, dir_path: str, pattern: str) -> List[str]:
        """List files in a directory matching a glob pattern. Returns relative paths."""
        ...

    @abstractmethod
    def rmtree(self, path: str) -> None:
        """Remove an entire directory tree."""
        ...

    @abstractmethod
    def listdir(self, path: str) -> List[str]:
        """List entry names (not full paths) in a directory."""
        ...

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        """Check if a path is a directory."""
        ...


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.logger = logging.getLogger("infini_memory_classic")

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.root / path

    def read_text(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def delete(self, path: str, missing_ok: bool = True) -> None:
        self._resolve(path).unlink(missing_ok=missing_ok)

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def mkdir(self, path: str) -> None:
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    def glob(self, dir_path: str, pattern: str) -> List[str]:
        resolved = self._resolve(dir_path)
        if not resolved.exists():
            return []
        results = []
        for p in resolved.glob(pattern):
            try:
                results.append(str(p.relative_to(self.root)))
            except ValueError:
                results.append(str(p))
        return results

    def rmtree(self, path: str) -> None:
        resolved = self._resolve(path)
        if resolved.exists():
            shutil.rmtree(resolved)

    def listdir(self, path: str) -> List[str]:
        resolved = self._resolve(path)
        if not resolved.exists():
            return []
        return sorted(d.name for d in resolved.iterdir())

    def is_dir(self, path: str) -> bool:
        return self._resolve(path).is_dir()


def create_storage_backend(
    storage_type: str = "local",
    root: Path | None = None,
    s3_endpoint: str = "",
    s3_bucket: str = "infini-memory",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    s3_prefix: str = "",
) -> StorageBackend:
    """Factory function to create the appropriate storage backend."""
    if storage_type == "local":
        return LocalStorage(root or Path.cwd())
    elif storage_type == "s3":
        from .s3_storage import S3Storage
        return S3Storage(
            endpoint=s3_endpoint,
            bucket=s3_bucket,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
            prefix=s3_prefix,
        )
    else:
        raise ValueError(f"Unknown storage_type: {storage_type!r}. Supported: 'local', 's3'")


__all__ = ["StorageBackend", "LocalStorage", "create_storage_backend"]
