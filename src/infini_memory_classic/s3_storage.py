from __future__ import annotations

import fnmatch
import logging
import posixpath
from typing import List

from .storage import StorageBackend


class S3Storage(StorageBackend):
    """S3-compatible object storage backend.

    Works with AWS S3, SeaweedFS, MinIO, and other S3-compatible services.
    Requires boto3: pip install infini-memory[s3]
    """

    def __init__(
        self,
        endpoint: str,
        bucket: str = "infini-memory",
        access_key: str = "",
        secret_key: str = "",
        prefix: str = "",
    ):
        try:
            import boto3
            from botocore.config import Config as BotoConfig
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install it with: pip install infini-memory[s3]"
            )

        self.logger = logging.getLogger("infini_memory_classic")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._ClientError = ClientError

        if access_key and secret_key:
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
            )
        else:
            from botocore import UNSIGNED as _UNSIGNED
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                config=BotoConfig(
                    signature_version=_UNSIGNED,
                    s3={"addressing_style": "path"},
                ),
            )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            self.logger.debug("[S3] Bucket exists: %s", self.bucket)
        except self._ClientError:
            self.logger.info("[S3] Creating bucket: %s", self.bucket)
            self.client.create_bucket(Bucket=self.bucket)
            self.logger.debug("[S3] Bucket created: %s", self.bucket)

    def _key(self, path: str) -> str:
        path = path.lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def read_text(self, path: str) -> str:
        key = self._key(path)
        self.logger.debug("[S3] GET %s/%s", self.bucket, key)
        resp = self.client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read().decode("utf-8")

    def write_text(self, path: str, content: str) -> None:
        key = self._key(path)
        self.logger.debug("[S3] PUT %s/%s (%d bytes)", self.bucket, key, len(content.encode("utf-8")))
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

    def delete(self, path: str, missing_ok: bool = True) -> None:
        key = self._key(path)
        self.logger.debug("[S3] DELETE %s/%s", self.bucket, key)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except self._ClientError:
            if not missing_ok:
                raise

    def exists(self, path: str) -> bool:
        key = self._key(path)
        self.logger.debug("[S3] HEAD %s/%s", self.bucket, key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._ClientError:
            prefix = key.rstrip("/") + "/"
            resp = self.client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=1
            )
            return resp.get("KeyCount", 0) > 0

    def mkdir(self, path: str) -> None:
        self.logger.debug("[S3] MKDIR (no-op) %s", path)

    def glob(self, dir_path: str, pattern: str) -> List[str]:
        prefix = self._key(dir_path).rstrip("/") + "/"
        self.logger.debug("[S3] LIST %s/%s (pattern=%s)", self.bucket, prefix, pattern)
        results: List[str] = []

        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_to_dir = key[len(prefix):]
                if "/" in rel_to_dir:
                    continue
                if fnmatch.fnmatch(rel_to_dir, pattern):
                    if self.prefix:
                        results.append(key[len(self.prefix) + 1:])
                    else:
                        results.append(key)

        return results

    def rmtree(self, path: str) -> None:
        prefix = self._key(path).rstrip("/") + "/"
        self.logger.debug("[S3] RMTREE %s/%s", self.bucket, prefix)

        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects},
                )

    def listdir(self, path: str) -> List[str]:
        raw_prefix = self._key(path)
        prefix = raw_prefix.rstrip("/") + "/" if raw_prefix else ""
        self.logger.debug("[S3] LISTDIR %s/%s", self.bucket, prefix)
        names: set[str] = set()

        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"][len(prefix):].rstrip("/")
                if name:
                    names.add(name)
            for obj in page.get("Contents", []):
                name = obj["Key"][len(prefix):]
                if name and "/" not in name:
                    names.add(name)

        return sorted(names)

    def is_dir(self, path: str) -> bool:
        prefix = self._key(path).rstrip("/") + "/"
        resp = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, MaxKeys=1
        )
        return resp.get("KeyCount", 0) > 0


__all__ = ["S3Storage"]
