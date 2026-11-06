"""Local filesystem and Google Cloud Storage object adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel


class ObjectRef(BaseModel):
    logical_id: str
    sha256: str
    version: str
    media_type: str
    size: int


class StorageError(RuntimeError):
    """Raised when an object cannot be read or written."""


class ObjectStore(Protocol):
    def put(self, logical_id: str, data: bytes, *, media_type: str) -> ObjectRef: ...

    def get(self, logical_id: str, *, version: str | None = None) -> bytes: ...

    def stat(self, logical_id: str) -> ObjectRef: ...

    def list_refs(self) -> list[ObjectRef]: ...

    def probe(self) -> None: ...


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_objects(source: ObjectStore, dest: ObjectStore) -> int:
    """Copy every generation and verify content hashes. Claim IRIs stay unchanged."""
    copied = 0
    for ref in source.list_refs():
        data = source.get(ref.logical_id, version=ref.version)
        written = dest.put(ref.logical_id, data, media_type=ref.media_type)
        if written.sha256 != ref.sha256 or sha256_bytes(data) != ref.sha256:
            raise StorageError(f"Copy verification failed for {ref.logical_id}")
        copied += 1
    return copied


class LocalObjectStore:
    """Filesystem store. Versions are generation tokens; bytes are kept by hash."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._blobs = root / "blobs"
        self._pointers = root / "pointers"
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._pointers.mkdir(parents=True, exist_ok=True)

    def put(self, logical_id: str, data: bytes, *, media_type: str) -> ObjectRef:
        digest = sha256_bytes(data)
        (self._blobs / digest).write_bytes(data)
        pointer = self._read_pointer(logical_id)
        generations = dict(cast(dict[str, str], pointer.get("generations", {})))
        version = str(int(str(pointer.get("version", "0"))) + 1) if pointer else "1"
        generations[version] = digest
        payload = {
            "logical_id": logical_id,
            "sha256": digest,
            "version": version,
            "media_type": media_type,
            "size": len(data),
            "generations": generations,
        }
        self._pointer_path(logical_id).write_text(json.dumps(payload), encoding="utf-8")
        return ObjectRef(
            logical_id=logical_id,
            sha256=digest,
            version=version,
            media_type=media_type,
            size=len(data),
        )

    def get(self, logical_id: str, *, version: str | None = None) -> bytes:
        pointer = self._require(logical_id)
        chosen = version or str(pointer["version"])
        generations = cast(dict[str, str], pointer["generations"])
        digest = generations.get(chosen)
        if digest is None:
            raise StorageError(f"Unknown version {chosen} for {logical_id}")
        return (self._blobs / digest).read_bytes()

    def stat(self, logical_id: str) -> ObjectRef:
        pointer = self._require(logical_id)
        return ObjectRef(
            logical_id=logical_id,
            sha256=str(pointer["sha256"]),
            version=str(pointer["version"]),
            media_type=str(pointer["media_type"]),
            size=int(str(pointer["size"])),
        )

    def list_refs(self) -> list[ObjectRef]:
        refs: list[ObjectRef] = []
        for path in sorted(self._pointers.glob("*.json")):
            pointer = cast(
                dict[str, object], json.loads(path.read_text(encoding="utf-8"))
            )
            media_type = str(pointer["media_type"])
            logical_id = str(pointer["logical_id"])
            generations = cast(dict[str, str], pointer["generations"])
            for version, digest in generations.items():
                size = (self._blobs / digest).stat().st_size
                refs.append(
                    ObjectRef(
                        logical_id=logical_id,
                        sha256=str(digest),
                        version=str(version),
                        media_type=media_type,
                        size=size,
                    )
                )
        return refs

    def probe(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def _require(self, logical_id: str) -> dict[str, object]:
        pointer = self._read_pointer(logical_id)
        if not pointer:
            raise StorageError(f"Object not found: {logical_id}")
        return pointer

    def _read_pointer(self, logical_id: str) -> dict[str, object]:
        path = self._pointer_path(logical_id)
        if not path.exists():
            return {}
        loaded = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        return loaded

    def _pointer_path(self, logical_id: str) -> Path:
        safe = logical_id.replace("/", "__")
        return self._pointers / f"{safe}.json"


class GcsObjectStore:
    """GCS adapter. Citations store the object generation, not a machine path."""

    def __init__(self, bucket: str) -> None:
        self._bucket_name = bucket
        self._client: Any | None = None

    def put(self, logical_id: str, data: bytes, *, media_type: str) -> ObjectRef:
        blob = self._bucket().blob(logical_id)
        blob.upload_from_string(data, content_type=media_type)
        blob.reload()
        return ObjectRef(
            logical_id=logical_id,
            sha256=sha256_bytes(data),
            version=str(blob.generation),
            media_type=media_type,
            size=len(data),
        )

    def get(self, logical_id: str, *, version: str | None = None) -> bytes:
        generation = int(version) if version else None
        blob = self._bucket().blob(logical_id, generation=generation)
        data: bytes = blob.download_as_bytes()
        return data

    def stat(self, logical_id: str) -> ObjectRef:
        blob = self._bucket().blob(logical_id)
        blob.reload()
        return ObjectRef(
            logical_id=logical_id,
            sha256=str(blob.md5_hash or ""),
            version=str(blob.generation),
            media_type=str(blob.content_type or "application/octet-stream"),
            size=int(blob.size or 0),
        )

    def list_refs(self) -> list[ObjectRef]:
        refs: list[ObjectRef] = []
        blobs = self._client_or_create().list_blobs(self._bucket_name, versions=True)
        for blob in blobs:
            refs.append(
                ObjectRef(
                    logical_id=str(blob.name),
                    sha256=str(blob.md5_hash or ""),
                    version=str(blob.generation),
                    media_type=str(blob.content_type or "application/octet-stream"),
                    size=int(blob.size or 0),
                )
            )
        return refs

    def probe(self) -> None:
        if not self._bucket().exists():
            raise StorageError(f"GCS bucket not found: {self._bucket_name}")

    def _bucket(self) -> Any:
        bucket: Any = self._client_or_create().bucket(self._bucket_name)
        return bucket

    def _client_or_create(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client
