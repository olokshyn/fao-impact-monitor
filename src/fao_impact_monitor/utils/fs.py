"""fsspec helpers for reading/writing blob paths."""

from __future__ import annotations

from typing import Literal, overload

from fsspec.core import url_to_fs


@overload
def read_file(file_path: str, mode: Literal["r"]) -> str: ...


@overload
def read_file(file_path: str, mode: Literal["rb"]) -> bytes: ...


def read_file(file_path: str, mode: Literal["r", "rb"]) -> str | bytes:
    """Read an fsspec path (local or remote URL).

    ``mode="rb"`` returns raw bytes. ``mode="r"`` decodes as UTF-8
    (errors replaced).
    """
    fs, path = url_to_fs(file_path)
    with fs.open(path, "rb") as handle:
        raw = handle.read()
    data = raw if isinstance(raw, bytes) else bytes(raw)
    if mode == "rb":
        return data
    return data.decode("utf-8", errors="replace")
