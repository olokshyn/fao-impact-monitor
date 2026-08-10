from pathlib import Path

from fao_impact_monitor.utils.document_uri import (
    file_document_uri,
    markdown_document_target,
)


def test_relative_pdf_path_has_file_scheme_and_is_url_encoded() -> None:
    uri = file_document_uri(Path("fao_data/El Niño Plan.pdf"))

    assert uri == "file://fao_data/El%20Ni%C3%B1o%20Plan.pdf"


def test_local_file_uri_becomes_relative_markdown_target(tmp_path: Path) -> None:
    pdf = tmp_path / "fao_data" / "El Niño Plan.pdf"

    target = markdown_document_target(pdf.as_uri(), base_dir=tmp_path)

    assert target == "fao_data/El%20Ni%C3%B1o%20Plan.pdf"
