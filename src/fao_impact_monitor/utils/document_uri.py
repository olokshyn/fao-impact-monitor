"""Local document URI helpers for storage and Markdown output."""

from pathlib import Path
from urllib.parse import quote, unquote


def file_document_uri(path: Path) -> str:
    """Return a file URI while preserving project-relative document paths."""
    if path.is_absolute():
        return path.as_uri()
    return f"file://{quote(path.as_posix(), safe='/')}"


def markdown_document_target(uri: str, *, base_dir: Path | None = None) -> str:
    """Convert a local file URI into a Markdown-safe relative link when possible."""
    if not uri.startswith("file://"):
        return uri

    raw_path = unquote(uri.removeprefix("file://"))
    path = Path(raw_path)
    if path.is_absolute():
        base = (base_dir or Path.cwd()).resolve()
        try:
            path = path.resolve().relative_to(base)
        except ValueError:
            return path.as_uri()
    return quote(path.as_posix(), safe="/%")
