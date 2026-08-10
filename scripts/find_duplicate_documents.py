"""Find documents with different names but identical contents."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "fao_data"


def sha256sum(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as document:
        while chunk := document.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_documents(folder: Path) -> list[tuple[str, list[Path]]]:
    """Group files with identical contents and at least two different names."""
    files_by_hash: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(
        candidate for candidate in folder.rglob("*") if candidate.is_file()
    ):
        files_by_hash[sha256sum(path)].append(path)

    duplicates = [
        (checksum, paths)
        for checksum, paths in files_by_hash.items()
        if len({path.name for path in paths}) > 1
    ]
    return sorted(duplicates, key=lambda group: str(group[1][0]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find differently named files with identical contents."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"folder to scan recursively (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    folder: Path = args.folder.resolve()

    if not folder.is_dir():
        parser.error(f"folder does not exist or is not a directory: {folder}")

    duplicates = find_duplicate_documents(folder)
    if not duplicates:
        print(f"No differently named duplicate documents found in {folder}")
        return

    print(f"Found {len(duplicates)} duplicate group(s) in {folder}:\n")
    for checksum, paths in duplicates:
        print(f"SHA-256: {checksum}")
        for path in paths:
            print(f"  {path.relative_to(folder)}")
        print()


if __name__ == "__main__":
    main()
