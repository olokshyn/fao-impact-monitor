"""Coverage for multi-column wrapping and table extraction in GIEWS PDFs."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from fao_impact_monitor.config import get_config
from fao_impact_monitor.pdf_pipeline.artifacts import PdfArtifacts, sha256_file
from fao_impact_monitor.pdf_pipeline.gemini import GeminiPdfClient
from fao_impact_monitor.pdf_pipeline.ingest import PdfEvidenceIngestor

_TEST_PDF = (
    Path(__file__).resolve().parents[1] / "data" / "giews_crop_prospects_july_2026.pdf"
)
# Printed report pages 3-4; physical PDF pages for the Mozambique column wrap.
_MOZ_START_PAGE = 6
_MOZ_END_PAGE = 7
_COUNTRY_PAGE = 8
_TABLE_PAGE = 10

_LATIN_AMERICA_SECTION = "LATIN AMERICA AND THE CARIBBEAN"
_NORTH_AMERICA_EUROPE_OCEANIA_SECTION = "NORTH AMERICA, EUROPE AND OCEANIA"


def _require_gemini() -> None:
    if not get_config().gemini.api_key.get_secret_value():
        pytest.skip("GEMINI_API_KEY not configured")


def _units(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    scope = analysis.get("scope_units")
    evidence = analysis.get("evidence_units")
    items = [
        *(scope if isinstance(scope, list) else []),
        *(evidence if isinstance(evidence, list) else []),
    ]
    return [item for item in items if isinstance(item, dict)]


def _unit_blob(unit: dict[str, Any]) -> str:
    parts = [
        unit.get("source_text"),
        unit.get("unit_description"),
        unit.get("retrieval_text"),
    ]
    facts = unit.get("visual_facts")
    if isinstance(facts, list):
        parts.extend(
            fact.get("text")
            for fact in facts
            if isinstance(fact, dict) and isinstance(fact.get("text"), str)
        )
    return "\n".join(part for part in parts if isinstance(part, str))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()


def _mentions_country(unit: dict[str, Any], *, iso3: str, name: str) -> bool:
    countries = unit.get("countries")
    if isinstance(countries, list) and any(
        country == iso3 for country in countries if isinstance(country, str)
    ):
        return True
    return name.lower() in _normalize(_unit_blob(unit))


def _mentions_text(unit: dict[str, Any], *needles: str) -> bool:
    blob = _normalize(_unit_blob(unit))
    return all(_normalize(needle) in blob for needle in needles)


def _prepare_excerpt(
    tmp_path: Path, pages: list[int], *, adjacent: bool = True
) -> tuple[Path, list[int]]:
    artifacts = PdfArtifacts(tmp_path / "artifacts", sha256_file(_TEST_PDF), 72)
    artifacts.prepare(_TEST_PDF)
    if adjacent:
        included = PdfEvidenceIngestor._with_adjacent_pages(pages, 47)
    else:
        included = pages
    return artifacts.section_pdf(included), included


def test_giews_mozambique_source_text_stitches_across_column_wrap(
    tmp_path: Path,
) -> None:
    """Local exact-text matching keeps the wrapped Mozambique passage together."""
    assert _TEST_PDF.is_file(), f"missing test PDF: {_TEST_PDF}"
    artifacts = PdfArtifacts(tmp_path / "artifacts", sha256_file(_TEST_PDF), 72)
    _, pages = artifacts.prepare(_TEST_PDF)
    source_pages = [pages[_MOZ_START_PAGE - 1], pages[_MOZ_END_PAGE - 1]]
    assert "Mozambique" in source_pages[0].extracted_text
    assert "April and September 2026" in source_pages[1].extracted_text

    page_one = (
        source_pages[0]
        .extracted_text[source_pages[0].extracted_text.index("Mozambique") :]
        .strip()
    )
    page_two = source_pages[1].extracted_text.split("Namibia")[0].strip()
    proposed = f"{page_one} {page_two}"

    page_texts = PdfEvidenceIngestor()._exact_source_texts_by_page(
        proposed, source_pages
    )
    assert _MOZ_START_PAGE in page_texts
    assert _MOZ_END_PAGE in page_texts
    assert "Mozambique" in page_texts[_MOZ_START_PAGE]
    assert "529 000" in page_texts[_MOZ_END_PAGE].replace("\xa0", " ")
    assert "Namibia" not in page_texts[_MOZ_END_PAGE]


@pytest.mark.integration
def test_giews_mozambique_column_wrap_extracted_as_single_evidence(
    tmp_path: Path,
) -> None:
    """Gemini must merge the page-break wrap into one Mozambique evidence unit."""
    _require_gemini()
    assert _TEST_PDF.is_file(), f"missing test PDF: {_TEST_PDF}"

    excerpt, included_pages = _prepare_excerpt(
        tmp_path, [_MOZ_START_PAGE, _MOZ_END_PAGE]
    )
    analysis = asyncio.run(
        GeminiPdfClient().section_evidence(
            excerpt,
            section_title="Mozambique",
            pages=[_MOZ_START_PAGE, _MOZ_END_PAGE],
            included_pages=included_pages,
        )
    )

    units = _units(analysis)
    moz_units = [
        unit for unit in units if _mentions_country(unit, iso3="MOZ", name="Mozambique")
    ]
    assert moz_units, f"expected Mozambique evidence, got {units!r}"

    wrapped = [
        unit
        for unit in moz_units
        if _mentions_text(unit, "April and September 2026")
        or _mentions_text(unit, "529 000")
    ]
    assert wrapped, (
        "Mozambique unit(s) missing the page-7 continuation "
        f"(529 000 / April and September): {moz_units!r}"
    )
    for unit in wrapped:
        pages = {page for page in unit.get("pages", []) if isinstance(page, int)}
        assert {_MOZ_START_PAGE, _MOZ_END_PAGE} <= pages, unit
        countries = unit.get("countries")
        assert isinstance(countries, list) and "MOZ" in countries, unit
        continuation = unit.get("continuation_pages")
        assert isinstance(continuation, list) and _MOZ_END_PAGE in continuation, unit

    orphans = [
        item
        for item in units
        if (
            _mentions_text(item, "April and September 2026")
            or _mentions_text(item, "529 000")
        )
        and not _mentions_country(item, iso3="MOZ", name="Mozambique")
    ]
    assert not orphans, f"orphan continuation without Mozambique: {orphans!r}"


@pytest.mark.integration
def test_giews_page8_myanmar_haiti_ukraine_column_wraps_and_parent_sections(
    tmp_path: Path,
) -> None:
    """Same-page column wraps stay whole; Haiti/Ukraine keep regional section titles."""
    _require_gemini()
    assert _TEST_PDF.is_file(), f"missing test PDF: {_TEST_PDF}"

    excerpt, included_pages = _prepare_excerpt(tmp_path, [_COUNTRY_PAGE])
    analysis = asyncio.run(
        GeminiPdfClient().section_evidence(
            excerpt,
            section_title=(
                "Countries/territories requiring external assistance for food"
            ),
            pages=[_COUNTRY_PAGE],
            included_pages=included_pages,
        )
    )
    units = _units(analysis)

    myanmar = [
        unit
        for unit in units
        if _mentions_country(unit, iso3="MMR", name="Myanmar")
        and _mentions_text(unit, "June to August 2026")
        and (
            _mentions_text(unit, "1 million")
            or _mentions_text(unit, "almost 1 million")
        )
    ]
    assert myanmar, f"expected merged Myanmar column-wrap evidence, got {units!r}"
    for unit in myanmar:
        countries = unit.get("countries")
        assert isinstance(countries, list) and "MMR" in countries, unit
    myanmar_orphans = [
        unit
        for unit in units
        if _mentions_text(unit, "June to August 2026", "1 million")
        and not _mentions_country(unit, iso3="MMR", name="Myanmar")
    ]
    assert not myanmar_orphans, f"orphan Myanmar continuation: {myanmar_orphans!r}"

    haiti = [
        unit
        for unit in units
        if _mentions_country(unit, iso3="HTI", name="Haiti")
        and _mentions_text(unit, "agricultural input prices")
        and _mentions_text(unit, _LATIN_AMERICA_SECTION)
    ]
    assert haiti, (
        "expected Haiti evidence that keeps LATIN AMERICA AND THE CARIBBEAN "
        f"and the column-2?3 wrap, got {units!r}"
    )
    for unit in haiti:
        countries = unit.get("countries")
        assert isinstance(countries, list) and "HTI" in countries, unit
        blob = _normalize(_unit_blob(unit))
        assert _normalize(_LATIN_AMERICA_SECTION) in blob, unit
    haiti_orphans = [
        unit
        for unit in units
        if _mentions_text(unit, "agricultural input prices")
        and not _mentions_country(unit, iso3="HTI", name="Haiti")
    ]
    assert not haiti_orphans, f"orphan Haiti continuation: {haiti_orphans!r}"

    ukraine = [
        unit
        for unit in units
        if _mentions_country(unit, iso3="UKR", name="Ukraine")
        and _mentions_text(unit, _NORTH_AMERICA_EUROPE_OCEANIA_SECTION)
    ]
    assert ukraine, (
        "expected Ukraine evidence referencing "
        f"{_NORTH_AMERICA_EUROPE_OCEANIA_SECTION}, got {units!r}"
    )
    for unit in ukraine:
        countries = unit.get("countries")
        assert isinstance(countries, list) and "UKR" in countries, unit
        blob = _normalize(_unit_blob(unit))
        assert _normalize(_NORTH_AMERICA_EUROPE_OCEANIA_SECTION) in blob, unit


@pytest.mark.integration
def test_giews_table1_world_cereal_production_rows(tmp_path: Path) -> None:
    """Table 1 must keep every column for each region/commodity row."""
    _require_gemini()
    assert _TEST_PDF.is_file(), f"missing test PDF: {_TEST_PDF}"

    excerpt, included_pages = _prepare_excerpt(tmp_path, [_TABLE_PAGE])
    analysis = asyncio.run(
        GeminiPdfClient().section_evidence(
            excerpt,
            section_title="Table 1. World cereal production",
            pages=[_TABLE_PAGE],
            included_pages=included_pages,
        )
    )
    units = _units(analysis)
    table_units = [
        unit
        for unit in units
        if unit.get("modality") in {"table", "mixed"}
        or _mentions_text(unit, "Table 1", "World cereal production")
        or _mentions_text(unit, "1 328.9")
        or _mentions_text(unit, "2 983.2")
    ]
    assert table_units, f"expected Table 1 evidence, got {units!r}"

    blob = _normalize("\n".join(_unit_blob(unit) for unit in table_units))
    headers = ("2024", "2025 est.", "2026 f'cast", "Change: 2026 over 2025")
    for header in headers:
        assert _normalize(header) in blob, (
            f"missing column header {header!r} in {blob!r}"
        )

    # Full row vectors from Table 1: label -> (2024, 2025 est., 2026 f'cast, change%)
    expected_rows = (
        ("asia", ("1 328.9", "1 331.0", "1 340.6", "+0.7")),
        ("far east", ("1 212.6", "1 230.4", "1 232.0", "+0.1")),
        ("near east", ("75.1", "60.4", "72.3", "+19.8")),
        (
            "south caucasus and central asia",
            ("41.2", "40.3", "36.2", "-10.1"),
        ),
        ("africa", ("202.3", "212.5", "216.0", "+1.6")),
        ("north africa", ("30.5", "32.3", "35.8", "+10.8")),
        ("west africa", ("70.1", "72.1", "69.2", "-4.0")),
        ("central africa", ("7.5", "7.3", "7.4", "+1.5")),
        ("east africa", ("63.0", "60.7", "61.7", "+1.7")),
        ("southern africa", ("31.3", "40.0", "41.7", "+4.3")),
        ("central america and the caribbean", ("38.8", "36.5", "38.0", "+4.1")),
        ("south america", ("251.1", "290.1", "297.6", "+2.6")),
        ("north america", ("516.3", "578.3", "530.6", "-8.2")),
        ("europe", ("476.7", "532.7", "513.5", "-3.6")),
        ("european union", ("257.8", "294.2", "279.3", "-5.0")),
        ("cis in europe", ("131.3", "145.5", "136.6", "-6.1")),
        ("oceania", ("53.1", "59.0", "46.9", "-20.5")),
        ("world", ("2 867.3", "3 040.1", "2 983.2", "-1.9")),
        ("wheat", ("798.5", "842.4", "806.5", "-4.3")),
        ("coarse grains", ("1 516.9", "1 635.4", "1 624.2", "-0.7")),
        ("rice (milled)", ("551.9", "562.3", "552.5", "-1.8")),
    )
    missing_rows: list[str] = []
    incomplete_rows: list[tuple[str, list[str]]] = []
    for label, values in expected_rows:
        if label not in blob:
            missing_rows.append(label)
            continue
        absent = [value for value in values if _normalize(value) not in blob]
        if absent:
            incomplete_rows.append((label, absent))
    assert not missing_rows, f"Table 1 missing row labels {missing_rows} in {blob!r}"
    assert not incomplete_rows, (
        f"Table 1 missing column values for rows {incomplete_rows} in {blob!r}"
    )

    # Prefer row-local pairing: a fact/line about Asia should carry more than 2024 alone.
    asia_lines = [
        text
        for unit in table_units
        for text in _unit_blob(unit).splitlines()
        if "asia" in _normalize(text) and "far east" not in _normalize(text)
    ]
    asia_joined = _normalize("\n".join(asia_lines) if asia_lines else blob)
    for value in ("1 328.9", "1 331.0", "1 340.6", "+0.7"):
        assert _normalize(value) in asia_joined, (
            f"Asia row missing column value {value!r} near Asia text: {asia_lines!r}"
        )
