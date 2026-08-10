"""Unit tests for research CLI routing and markdown helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fao_impact_monitor.agent.researcher_agent import (
    AnswerStatement,
    EvidenceClaim,
    EvidenceGap,
    ResearcherOutput,
    SourceReference,
    StatementCitation,
)
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.data_source.faostat import FAOSTATDataResult
from fao_impact_monitor.data_source.world_bank import WorldBankDataResult
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.research_report import (
    build_report,
    build_research_pdf,
    combine_metric_reports,
    default_research_dir,
    default_research_pdf_path,
    ensure_research_output_dir,
    format_metric_section,
    format_researcher_result,
    format_structured_result,
    format_worldbank_result,
    is_worldbank_only,
    list_metric_report_files,
    metric_path,
    metric_report_path,
    report_pdf_filename,
    select_metrics,
    write_metric_report,
)


def _metric(
    *,
    name: str = "Cropland",
    sources: list[DataSourceConfig] | None = None,
) -> Metric:
    return Metric(
        name=name,
        description="Desc",
        example="Example text",
        unit="%",
        data_sources=sources or [],
    )


def test_is_worldbank_only() -> None:
    wb = _metric(
        sources=[
            DataSourceConfig.model_validate(
                {
                    "source": "WorldBank",
                    "indicator": "NV.AGR.TOTL.ZS",
                    "exclusive": True,
                }
            )
        ]
    )
    fao = _metric(
        sources=[
            DataSourceConfig.model_validate(
                {"source": "FAORepository", "root_url": "https://x"}
            )
        ]
    )
    empty = _metric(sources=[])
    assert is_worldbank_only(wb) is True
    assert metric_path(wb) == "worldbank"
    assert is_worldbank_only(fao) is False
    assert metric_path(fao) == "researcher"
    assert is_worldbank_only(empty) is False


def test_select_metrics_all_and_subset() -> None:
    metrics = [_metric(name="A"), _metric(name="B"), _metric(name="C")]
    assert [i for i, _ in select_metrics(metrics, None)] == [1, 2, 3]
    selected = select_metrics(metrics, [3, 1])
    assert [(i, m.name) for i, m in selected] == [(3, "C"), (1, "A")]


def test_select_metrics_rejects_out_of_range() -> None:
    metrics = [_metric(name="A")]
    with pytest.raises(ValueError, match="out of range"):
        select_metrics(metrics, [0])
    with pytest.raises(ValueError, match="out of range"):
        select_metrics(metrics, [2])


def test_format_worldbank_result_plot_and_indicator_ref(tmp_path: Path) -> None:
    result = WorldBankDataResult(
        source="WorldBank",
        title="Agriculture, forestry, and fishing, value added (% of GDP)",
        url="https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=KE",
        citation="cite",
        metadata={
            "indicator": "NV.AGR.TOTL.ZS",
            "country_iso3": "KEN",
            "unit": "%",
        },
        data=pd.DataFrame({"year": [2022, 2023], "value": [21.1, 20.5]}),
    )
    body, refs = format_worldbank_result(
        [result],
        plot_dir=tmp_path / "plots",
        plot_stem="gdp",
    )
    assert "![Agriculture, forestry" in body
    assert "| Year | Value |" not in body
    assert (tmp_path / "plots" / "gdp-worldbank-1.png").is_file()
    assert "NV.AGR.TOTL.ZS" in refs[0]
    assert "https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=KE" in refs[0]


def test_format_faostat_result_plot_and_reference(tmp_path: Path) -> None:
    result = FAOSTATDataResult(
        source="FAOSTAT",
        title="Crop and livestock products",
        url="https://www.fao.org/faostat/en/#data/QCL",
        citation="cite",
        metadata={"indicator": "Crop and livestock products", "country_iso3": "KEN"},
        data=pd.DataFrame(
            {
                "year": [2021, 2022, 2023],
                "value": [3_800_000.0, 4_200_000.0, 4_050_000.0],
                "item": ["Maize (corn)"] * 3,
                "element": ["Production"] * 3,
                "unit": ["t"] * 3,
                "qualifier": ["National"] * 3,
            }
        ),
    )

    body, refs = format_structured_result(
        [result],
        plot_dir=tmp_path / "plots",
        plot_stem="production",
    )

    assert "![Crop and livestock products — National]" in body
    assert "| Year | Value |" not in body
    assert (tmp_path / "plots" / "production-faostat-1.png").is_file()
    assert "FAOSTAT" in refs[0]


def test_metric_report_path_and_defaults() -> None:
    assert metric_report_path(Path("reports/el-nino/KEN"), 2) == Path(
        "reports/el-nino/KEN/0002.md"
    )
    assert default_research_dir("ken") == Path("reports/el-nino/KEN")
    assert default_research_dir("ken", use_case=Path("use-cases/custom.json")) == Path(
        "reports/custom/KEN"
    )
    assert default_research_pdf_path("ken") == Path(
        "reports/el-nino/KEN/El Niño - KEN.pdf"
    )


def test_report_pdf_filename_uses_use_case_template(tmp_path: Path) -> None:
    use_case = tmp_path / "drought.json"
    use_case.write_text(
        (
            '{"name": "El Niño", '
            '"report_pdf_template": "{name} - {country}.pdf", '
            '"metrics": []}'
        ),
        encoding="utf-8",
    )
    assert report_pdf_filename("eth", use_case=use_case) == "El Niño - ETH.pdf"
    assert default_research_pdf_path("eth", use_case=use_case) == (
        Path("reports/drought/ETH/El Niño - ETH.pdf")
    )


def test_write_metric_report(tmp_path: Path) -> None:
    output_dir = ensure_research_output_dir(tmp_path / "el-nino-KEN")
    report_path = output_dir / "0002.md"
    written = write_metric_report(report_path, "# metric 2\n")
    assert written == report_path
    assert report_path.read_text(encoding="utf-8") == "# metric 2\n"


def test_combine_metric_reports_keeps_first_title_and_per_metric_references(
    tmp_path: Path,
) -> None:
    (tmp_path / "0001.md").write_text(
        "# El Nino research - KEN\n\n"
        "## 1. First\n\nBody A\n\n"
        "### References\n\n- [A](https://a.example)\n",
        encoding="utf-8",
    )
    (tmp_path / "0003.md").write_text(
        "# El Nino research - KEN\n\n"
        "## 3. Third\n\nBody B\n\n"
        "### References\n\n- [B](https://b.example)\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    files = list_metric_report_files(tmp_path)
    assert [p.name for p in files] == ["0001.md", "0003.md"]
    combined = combine_metric_reports(files)
    assert combined.startswith("# El Nino research - KEN\n")
    assert combined.count("# El Nino research - KEN") == 1
    assert sum(1 for line in combined.splitlines() if line.startswith("# ")) == 1
    assert "## 1. First" in combined
    assert "## 3. Third" in combined
    assert "Body A" in combined
    assert "Body B" in combined
    # Separate References sections - not merged into one list.
    assert combined.count("### References") == 2
    assert "- [A](https://a.example)" in combined
    assert "- [B](https://b.example)" in combined
    first_refs = combined.index("### References")
    second_refs = combined.index("### References", first_refs + 1)
    assert first_refs < combined.index("## 3. Third") < second_refs


def test_combine_metric_reports_orders_by_section_number(
    tmp_path: Path,
) -> None:
    # Write higher number first; combine must still emit ## 1 before ## 2.
    (tmp_path / "0002.md").write_text(
        "# Title\n\n## 2. Second\n\nBody 2\n",
        encoding="utf-8",
    )
    (tmp_path / "0001.md").write_text(
        "# Title\n\n## 1. First\n\nBody 1\n",
        encoding="utf-8",
    )
    # Pass files in reverse numeric order on purpose.
    combined = combine_metric_reports([tmp_path / "0002.md", tmp_path / "0001.md"])
    assert combined.index("## 1. First") < combined.index("## 2. Second")


def test_combine_metric_reports_recovers_missing_section_heading(
    tmp_path: Path,
) -> None:
    (tmp_path / "0004.md").write_text(
        "# Title\n\nBody without a heading\n",
        encoding="utf-8",
    )
    combined = combine_metric_reports([tmp_path / "0004.md"])
    assert "## 4. Metric 4" in combined
    assert "Body without a heading" in combined


def test_build_research_pdf_writes_pdf(tmp_path: Path) -> None:
    reports = tmp_path / "el-nino-KEN"
    reports.mkdir()
    plots = reports / "plots"
    plots.mkdir()
    formatted, _ = format_worldbank_result(
        [
            WorldBankDataResult(
                source="WorldBank",
                title="Agriculture share of GDP",
                url="https://data.worldbank.org/indicator/X",
                citation="cite",
                metadata={"indicator": "X", "country_iso3": "KEN", "unit": "%"},
                data=pd.DataFrame({"year": [2022, 2023], "value": [21.1, 20.5]}),
            )
        ],
        plot_dir=plots,
        plot_stem="gdp",
    )
    (reports / "0002.md").write_text(
        "# El Nino research - KEN\n\n## 2. Pastureland\n\nSecond body.\n",
        encoding="utf-8",
    )
    (reports / "0001.md").write_text(
        "# El Nino research - KEN\n\n## 1. Cropland\n\n"
        f"**Status:** answered\n\n{formatted}\n",
        encoding="utf-8",
    )
    output = tmp_path / "el-nino-KEN.pdf"
    written = build_research_pdf(input_dir=reports, output_path=output)
    assert written == output
    assert output.is_file()
    assert output.read_bytes()[:4] == b"%PDF"
    pdf_bytes = output.read_bytes()
    # Section titles must survive PDF encoding (xhtml2pdf stores as literal text).
    assert b"Cropland" in pdf_bytes
    assert b"Pastureland" in pdf_bytes
    assert b"/Subtype /Image" in pdf_bytes


def test_build_research_pdf_makes_local_links_clickable(tmp_path: Path) -> None:
    import pymupdf

    reports = tmp_path / "reports"
    fao_data = reports / "fao_data"
    fao_data.mkdir(parents=True)
    local_pdf = fao_data / "My Doc.pdf"
    local_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    (reports / "0001.md").write_text(
        "# Title\n\n## 1. Metric\n\n"
        "See [local](fao_data/My%20Doc.pdf) and [web](https://example.com/a).\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.pdf"
    build_research_pdf(input_dir=reports, output_path=output)

    from urllib.parse import unquote

    links = [link for page in pymupdf.open(output) for link in page.get_links()]
    uris = {link.get("uri") for link in links if link.get("uri")}
    files = {unquote(str(link.get("file"))) for link in links if link.get("file")}
    assert "https://example.com/a" in uris
    # Portable relative target (fao_data/ beside report.pdf), not absolute file://.
    assert "fao_data/My Doc.pdf" in files
    assert not any(path.startswith("/") for path in files)
    assert not any(str(uri).startswith("file:") for uri in uris)
    assert b"/GoToR" not in output.read_bytes()
    assert str(local_pdf.resolve()).encode() not in output.read_bytes()


def test_format_researcher_result_citations() -> None:
    output = ResearcherOutput(
        status="answered",
        country="Kenya",
        metric_name="Cropland",
        final_summary=(
            "Cropland was affected. ([Kenya Report, p. 3](https://fao.org/doc.pdf))"
        ),
        statements=[
            AnswerStatement(
                statement_id="stmt_001",
                text="Cropland was affected.",
                supporting_claim_ids=["claim_001"],
                citations=[
                    StatementCitation(
                        document_name="Kenya Report",
                        document_uri="https://fao.org/doc.pdf",
                        page_number=3,
                        origin="FAORepository",
                    )
                ],
            )
        ],
        claims=[
            EvidenceClaim(
                claim_id="claim_001",
                source_type="vectorstore",
                source_id="vs:1:2",
                quoted_text="Cropland was affected.",
                country="Kenya",
                relevance="direct",
                url="https://fao.org/doc.pdf",
                page_number=3,
            )
        ],
        sources=[
            SourceReference(
                source_id="vs:1:2",
                source_type="vectorstore",
                document_uri="https://fao.org/doc.pdf",
                document_name="Kenya Report",
                page_number=3,
                document_source="FaoRepository",
            )
        ],
        open_gaps=[],
        research_iterations=1,
    )
    body, refs = format_researcher_result(output)
    assert "answered" in body
    assert "Cropland was affected" in body
    assert refs == ["- [Kenya Report, p. 3](https://fao.org/doc.pdf) (FAORepository)"]


def test_format_researcher_result_uses_relative_local_pdf_link() -> None:
    output = ResearcherOutput(
        status="answered",
        country="Somalia",
        metric_name="Flood impacts",
        final_summary="Flooding affected farms.",
        statements=[],
        claims=[],
        sources=[
            SourceReference(
                source_id="pdf:1",
                source_type="vectorstore",
                document_uri="file://fao_data/El%20Ni%C3%B1o%20Plan.pdf",
                document_name="El Niño Plan",
                page_number=2,
                document_source="PdfEvidencePipeline",
            )
        ],
        open_gaps=[],
        research_iterations=1,
    )

    _, refs = format_researcher_result(output)

    assert refs == [
        (
            "- [El Niño Plan, p. 2](fao_data/El%20Ni%C3%B1o%20Plan.pdf) "
            "(PdfEvidencePipeline)"
        )
    ]


def test_format_researcher_result_high_level_gaps_then_findings() -> None:
    output = ResearcherOutput(
        status="high_level_answer",
        country="Kenya",
        metric_name="Cropland",
        final_summary="unused",
        statements=[
            AnswerStatement(
                statement_id="stmt_001",
                text="Some cropland was flooded.",
                supporting_claim_ids=["claim_001"],
                citations=[
                    StatementCitation(
                        document_name="Flood Note",
                        document_uri="https://fao.org/flood.pdf",
                        page_number=1,
                        origin="tellus",
                    )
                ],
            )
        ],
        claims=[],
        sources=[],
        open_gaps=[
            EvidenceGap(
                gap_id="gap_001",
                description="No national aggregate",
                why_required="Need country-level value",
                preferred_source_type="vectorstore",
                status="open",
            )
        ],
        research_iterations=2,
    )
    body, refs = format_researcher_result(output)
    assert "high level answer, lacking detailed evidence" in body
    assert body.index("Some cropland was flooded") < body.index(
        "### Remaining evidence gaps"
    )
    assert "gap_001" in body
    assert refs == ["- [Flood Note, p. 1](https://fao.org/flood.pdf) (tellus)"]


def test_format_researcher_result_cannot_answer() -> None:
    output = ResearcherOutput(
        status="cannot_answer",
        country="Kenya",
        metric_name="Cropland",
        final_summary="unused",
        statements=[],
        claims=[],
        sources=[],
        open_gaps=[
            EvidenceGap(
                gap_id="gap_001",
                description="No usable evidence",
                why_required="Need country-level value",
                status="open",
            )
        ],
        research_iterations=1,
    )
    body, _refs = format_researcher_result(output)
    assert "cannot answer with available evidence" in body
    assert "gap_001" in body


def test_format_researcher_result_separates_answer_and_context() -> None:
    output = ResearcherOutput(
        status="answered",
        country="Kenya",
        metric_name="Production change",
        final_summary="unused",
        statements=[
            AnswerStatement(
                statement_id="stmt_001",
                text="Production declined by 18 percent.",
                statement_type="answer",
            ),
            AnswerStatement(
                statement_id="stmt_002",
                text="Average drought duration increased by 3 months.",
                statement_type="context",
            ),
        ],
        claims=[],
        sources=[],
        open_gaps=[],
        research_iterations=1,
    )

    body, _refs = format_researcher_result(output)

    assert "### Answer" in body
    assert "### Context" in body
    assert body.index("Production declined") < body.index("### Context")
    assert body.index("### Context") < body.index("drought duration")


def test_format_metric_section_and_report() -> None:
    metric = _metric(name="Pastureland")
    section = format_metric_section(
        section_number=1,
        metric=metric,
        result_markdown="Some answer.",
        reference_lines=["- [Doc, p. 1](https://example.org/a.pdf)"],
    )
    assert "## 1. Pastureland" in section
    assert "**Description:** Desc" in section
    assert "**Example:** Example text" in section
    assert "**Unit:** %" in section
    assert "Some answer." in section
    assert "[Doc, p. 1](https://example.org/a.pdf)" in section

    report = build_report(
        title="El Nino research",
        country_iso3="ken",
        sections=[section],
    )
    assert report.startswith("# El Nino research - KEN\n")
    assert "## 1. Pastureland" in report


def test_el_nino_routing_matches_plan() -> None:
    metrics = Metric.from_use_case(Path("use-cases/el-nino.json"))
    irrigated_index = next(
        index
        for index, metric in enumerate(metrics)
        if metric.name == "Irrigated cropland"
    )

    assert metric_path(metrics[0]) == "worldbank"
    assert metric_path(metrics[1]) == "worldbank"
    assert all(
        metric_path(metric) == "researcher" for metric in metrics[2:irrigated_index]
    )
    assert all(metric_path(metric) == "faostat" for metric in metrics[irrigated_index:])
