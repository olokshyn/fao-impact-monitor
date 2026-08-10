from pathlib import Path

from scripts.find_duplicate_documents import find_duplicate_documents


def test_finds_same_content_under_different_names(tmp_path: Path) -> None:
    (tmp_path / "first.pdf").write_bytes(b"same document")
    (tmp_path / "renamed.pdf").write_bytes(b"same document")
    (tmp_path / "different.pdf").write_bytes(b"different document")

    duplicates = find_duplicate_documents(tmp_path)

    assert len(duplicates) == 1
    assert {path.name for path in duplicates[0][1]} == {"first.pdf", "renamed.pdf"}


def test_ignores_same_name_in_different_directories(tmp_path: Path) -> None:
    first_folder = tmp_path / "first"
    second_folder = tmp_path / "second"
    first_folder.mkdir()
    second_folder.mkdir()
    (first_folder / "document.pdf").write_bytes(b"same document")
    (second_folder / "document.pdf").write_bytes(b"same document")

    assert find_duplicate_documents(tmp_path) == []
