"""Local document URI helpers for storage and Markdown output."""

import unicodedata
from pathlib import Path
from urllib.parse import quote, unquote

_ASCII_PUNCTUATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def ascii_relative_path(path: str) -> str:
    """Fold a relative path to ASCII (``ñ``→``n``, unicode dashes→``-``).

    Generated report names and Preview Launch ``/F`` filespecs must be
    MacRoman-encodable; ASCII is a safe subset.
    """
    text = unicodedata.normalize("NFKD", path).translate(_ASCII_PUNCTUATION)
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return text.encode("ascii", "ignore").decode("ascii")


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
