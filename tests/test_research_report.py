"""Unit tests for research CLI routing and markdown helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from fao_impact_monitor.agent.researcher_agent import (
    AnswerStatement,
    EvidenceClaim,
    PdfEvidenceEvent,
    PdfVerifiedVisualFact,
    QueryRunStat,
    ResearcherOutput,
    SourceReference,
)
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.data_source.emdat import EmDatDataResult
from fao_impact_monitor.data_source.faostat import FAOSTATDataResult
from fao_impact_monitor.data_source.world_bank import WorldBankDataResult
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.research_report import (
    build_metric_process_jobs,
    build_report,
    build_research_pdf,
    combine_metric_reports,
    default_research_dir,
    default_research_pdf_path,
    ensure_research_output_dir,
    format_metric_section,
    format_queries_section,
    format_researcher_result,
    format_structured_result,
    format_worldbank_result,
    is_worldbank_only,
    list_metric_report_files,
    markdown_to_pdf,
    metric_human_report_path,
    metric_path,
    metric_report_path,
    missing_researcher_process_jobs,
    parse_countries_iso3,
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


def test_parse_countries_iso3() -> None:
    assert parse_countries_iso3("eth, ken,MWI") == ["ETH", "KEN", "MWI"]
    assert parse_countries_iso3("ETH ETH,KEN") == ["ETH", "KEN"]
    with pytest.raises(ValueError, match="At least one"):
        parse_countries_iso3(" ,  ")


def test_build_metric_process_jobs_mixed() -> None:
    metrics = [
        _metric(
            name="WB",
            sources=[
                DataSourceConfig.model_validate(
                    {
                        "source": "WorldBank",
                        "indicator": "NV.AGR.TOTL.ZS",
                        "exclusive": True,
                    }
                )
            ],
        ),
        _metric(
            name="Text A",
            sources=[
                DataSourceConfig.model_validate(
                    {"source": "FAORepository", "root_url": "https://x"}
                )
            ],
        ),
        _metric(
            name="Text B",
            sources=[
                DataSourceConfig.model_validate(
                    {"source": "FAORepository", "root_url": "https://y"}
                )
            ],
        ),
        _metric(
            name="FAO",
            sources=[DataSourceConfig(source="FAOSTAT", exclusive=True)],
        ),
        _metric(
            name="EM",
            sources=[
                DataSourceConfig.model_validate(
                    {"source": "EMDAT", "indicator": "Total Deaths", "exclusive": True}
                )
            ],
        ),
    ]
    jobs = build_metric_process_jobs(metrics)
    assert [(job.kind, job.metric_indices, job.data_source) for job in jobs] == [
        ("worldbank", [1], "WorldBank"),
        ("faostat", [4], "FAOSTAT"),
        ("emdat", [5], "EMDAT"),
        ("researcher", [2], None),
        ("researcher", [3], None),
    ]


def test_build_metric_process_jobs_el_nino() -> None:
    metrics = Metric.from_use_case(Path("use-cases/el-nino.json"))
    jobs = build_metric_process_jobs(metrics)
    assert len(jobs) == 15
    assert jobs[0].kind == "worldbank"
    assert jobs[0].metric_indices == [1, 2]
    assert jobs[1].kind == "faostat"
    assert jobs[1].metric_indices == list(range(3, 14))
    assert jobs[1].data_source == "FAOSTAT"
    assert jobs[2].kind == "emdat"
    assert jobs[2].metric_indices == [26, 27, 28, 29]
    assert jobs[2].data_source == "EMDAT"
    researcher_jobs = jobs[3:]
    assert len(researcher_jobs) == 12
    assert [job.metric_indices[0] for job in researcher_jobs] == list(range(14, 26))
    assert all(job.kind == "researcher" for job in researcher_jobs)
    assert all(job.data_source is None for job in researcher_jobs)


def test_missing_researcher_process_jobs_skips_structured_and_written(
    tmp_path: Path,
) -> None:
    metrics = [
        _metric(
            name="WB",
            sources=[
                DataSourceConfig.model_validate(
                    {
                        "source": "WorldBank",
                        "indicator": "NV.AGR.TOTL.ZS",
                        "exclusive": True,
                    }
                )
            ],
        ),
        _metric(
            name="Text A",
            sources=[
                DataSourceConfig.model_validate(
                    {"source": "FAORepository", "root_url": "https://x"}
                )
            ],
        ),
        _metric(
            name="Text B",
            sources=[
                DataSourceConfig.model_validate(
                    {"source": "FAORepository", "root_url": "https://y"}
                )
            ],
        ),
        _metric(
            name="FAO",
            sources=[DataSourceConfig(source="FAOSTAT", exclusive=True)],
        ),
        _metric(
            name="EM",
            sources=[
                DataSourceConfig.model_validate(
                    {"source": "EMDAT", "indicator": "Total Deaths", "exclusive": True}
                )
            ],
        ),
    ]
    output_dir = tmp_path / "ETH"
    output_dir.mkdir()
    (output_dir / "0001.md").write_text("worldbank\n", encoding="utf-8")
    (output_dir / "0002.md").write_text("text A\n", encoding="utf-8")
    (output_dir / "0003.md").write_text("", encoding="utf-8")

    jobs = missing_researcher_process_jobs(metrics, output_dir)
    assert [(job.kind, job.metric_indices) for job in jobs] == [("researcher", [3])]

    missing_dir = tmp_path / "KEN"
    assert [
        job.metric_indices[0]
        for job in missing_researcher_process_jobs(metrics, missing_dir)
    ] == [2, 3]


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
    assert "## Direct evidence" in body
    assert "Source data:" not in body
    assert "| Year | Value | Unit |" not in body
    assert "## Indirect evidence\n\nNone." in body
    assert "Latest value: 20.5 % (2023)" in body
    assert (
        "Source url: https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=KE"
        in body
    )
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
                "value_raw": ["3800000", "<0.1", "4050000"],
            }
        ),
    )

    body, refs = format_structured_result(
        [result],
        plot_dir=tmp_path / "plots",
        plot_stem="production",
    )

    assert "![Crop and livestock products — National]" in body
    assert "Source data:" not in body
    assert (
        "| Year | Item | Element | Qualifier | Unit | Value raw | Value |" not in body
    )
    assert (tmp_path / "plots" / "production-faostat-1.png").is_file()
    assert "FAOSTAT" in refs[0]


def test_format_emdat_result_table_and_reference(tmp_path: Path) -> None:
    result = EmDatDataResult(
        source="EMDAT",
        title="Total Deaths",
        url="https://www.emdat.be",
        citation="cite",
        metadata={
            "indicator": "Total Deaths",
            "country_iso3": "KEN",
            "unit": "persons",
        },
        data=pd.DataFrame(
            {
                "start_year": [2023, 2023, 2016],
                "disaster_type": ["Flood", "Flood", "Flood"],
                "location": ["Nairobi", "Central", "Coast"],
                "value": [178.0, 22.0, 3.0],
                "regions": [["Nairobi", "Kiambu"], ["Nairobi"], ["Coast"]],
            }
        ),
    )

    body, refs = format_structured_result(
        [result],
        plot_dir=tmp_path / "plots",
        plot_stem="deaths",
    )

    assert "Source: Total Deaths" in body
    assert "| Start Year | Disaster Type | Location | Regions | Value | Unit |" in body
    assert "| 2023 | Flood | Nairobi | Nairobi; Kiambu | 178 | persons |" in body
    assert "| 2016 | Flood | Coast | Coast | 3 | persons |" in body
    assert "![Total Deaths](plots/deaths-emdat-1.png)" in body
    assert (tmp_path / "plots" / "deaths-emdat-1.png").is_file()
    assert "EM-DAT" in refs[0]
    assert "https://www.emdat.be" in refs[0]


def test_metric_report_path_and_defaults() -> None:
    assert metric_report_path(Path("reports/el-nino/KEN"), 2) == Path(
        "reports/el-nino/KEN/0002.md"
    )
    assert default_research_dir("ken") == Path("reports/el-nino/KEN")
    assert default_research_dir("ken", use_case=Path("use-cases/custom.json")) == Path(
        "reports/custom/KEN"
    )
    assert default_research_pdf_path("ken") == Path(
        "reports/el-nino/KEN/KEN metrics El Nino.pdf"
    )


def test_report_pdf_filename_uses_use_case_template(tmp_path: Path) -> None:
    use_case = tmp_path / "drought.json"
    use_case.write_text(
        (
            '{"name": "El Niño", '
            '"report_pdf_template": "{country} metrics {name}.pdf", '
            '"metrics": []}'
        ),
        encoding="utf-8",
    )
    assert report_pdf_filename("eth", use_case=use_case) == ("ETH metrics El Nino.pdf")
    assert report_pdf_filename("fji", use_case=use_case, human=True) == (
        "FJI metrics El Nino - human.pdf"
    )
    assert default_research_pdf_path("eth", use_case=use_case) == (
        Path("reports/drought/ETH/ETH metrics El Nino.pdf")
    )
    assert default_research_pdf_path("fji", use_case=use_case, human=True) == (
        Path("reports/drought/FJI/FJI metrics El Nino - human.pdf")
    )


def test_write_metric_report(tmp_path: Path) -> None:
    output_dir = ensure_research_output_dir(tmp_path / "el-nino-KEN")
    report_path = output_dir / "0002.md"
    written = write_metric_report(report_path, "# metric 2\n")
    assert written == report_path
    assert report_path.read_text(encoding="utf-8") == "# metric 2\n"


def test_combine_metric_reports_adds_title_and_keeps_metric_files_intact(
    tmp_path: Path,
) -> None:
    (tmp_path / "0001.md").write_text(
        "## Metric info\nSeq Number: 1\nName: First\n\n"
        "## Direct evidence\n\nNone.\n\n"
        "## References\n\n1. [A](https://a.example)\n",
        encoding="utf-8",
    )
    (tmp_path / "0003.md").write_text(
        "## Metric info\nSeq Number: 3\nName: Third\n\n"
        "## Direct evidence\n\nNone.\n\n"
        "## References\n\n1. [B](https://b.example)\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    files = list_metric_report_files(tmp_path)
    assert [p.name for p in files] == ["0001.md", "0003.md"]
    combined = combine_metric_reports(files, title="El Nino research - KEN")
    assert combined.startswith("# El Nino research - KEN\n")
    assert combined.count("## Metric info") == 2
    assert combined.index("Seq Number: 1") < combined.index("Seq Number: 3")
    assert combined.count("## References") == 2
    assert "1. [A](https://a.example)" in combined
    assert "1. [B](https://b.example)" in combined


def test_combine_metric_reports_orders_by_section_number(
    tmp_path: Path,
) -> None:
    # Write higher number first; combine must still emit sequence 1 before 2.
    (tmp_path / "0002.md").write_text(
        "## Metric info\nSeq Number: 2\nName: Second\n",
        encoding="utf-8",
    )
    (tmp_path / "0001.md").write_text(
        "## Metric info\nSeq Number: 1\nName: First\n",
        encoding="utf-8",
    )
    # Pass files in reverse numeric order on purpose.
    combined = combine_metric_reports([tmp_path / "0002.md", tmp_path / "0001.md"])
    assert combined.index("Seq Number: 1") < combined.index("Seq Number: 2")


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


def test_metric_human_report_path_uses_h_suffix(tmp_path: Path) -> None:
    assert metric_human_report_path(tmp_path, 15) == tmp_path / "0015-H.md"


def test_list_metric_report_files_human_prefers_h_twin(tmp_path: Path) -> None:
    (tmp_path / "0001.md").write_text("# 1. Text metric\n\nmachine\n", encoding="utf-8")
    (tmp_path / "0001-H.md").write_text("# 1. Text metric\n\nhuman\n", encoding="utf-8")
    (tmp_path / "0002.md").write_text("# World Bank\n\nplot\n", encoding="utf-8")
    listed = list_metric_report_files(tmp_path, human=True)
    assert listed == [tmp_path / "0001-H.md", tmp_path / "0002.md"]


def test_combine_metric_reports_keeps_numbered_h1(tmp_path: Path) -> None:
    first = tmp_path / "0001-H.md"
    first.write_text("# 1. Crop yield loss\n\n**Description:** x\n", encoding="utf-8")
    combined = combine_metric_reports([first])
    assert "# 1. Crop yield loss" in combined
    assert combined.index("# 1. Crop yield loss") > combined.index("# Research report")


def test_combine_metric_reports_human_rewrites_worldbank_layout(
    tmp_path: Path,
) -> None:
    (tmp_path / "0001.md").write_text(
        "## Metric info\n\n"
        "Seq Number: 1\n\n"
        "Name: Agriculture share of GDP\n\n"
        "Description: The share of agriculture in GDP.\n\n"
        "Example: Agriculture contributed 24.3% of GDP in 2023.\n\n"
        "Unit: %\n\n"
        "## Direct evidence\n\n"
        "### Direct Evidence 1\n\n"
        "Source: Agriculture value added\n\n"
        "Indicator: NV.AGR.TOTL.ZS\n\n"
        "Latest value: 14 % (2025)\n\n"
        "Plot: ![Agriculture value added](plots/0001-worldbank-1.png)\n\n"
        "## Indirect evidence\n\nNone.\n\n"
        "## References\n\n"
        "1. [Agriculture value added](https://data.worldbank.org/indicator/X)\n",
        encoding="utf-8",
    )
    (tmp_path / "0002.md").write_text(
        "## Metric info\nSeq Number: 2\nName: Pastureland\n\n"
        "## Queries\n\n1. [vectorstore] query\n",
        encoding="utf-8",
    )
    combined = combine_metric_reports(
        [tmp_path / "0001.md", tmp_path / "0002.md"],
        human=True,
    )
    assert "# 1. Agriculture share of GDP" in combined
    assert "Description: The share of agriculture in GDP." in combined
    assert "Example: Agriculture contributed 24.3% of GDP in 2023." in combined
    assert "## Plots" in combined
    assert "![Agriculture value added](plots/0001-worldbank-1.png)" in combined
    assert "Seq Number: 1" not in combined
    assert "Latest value:" not in combined
    assert "Unit: %" not in combined
    assert "Seq Number: 2" in combined
    assert "## Queries" in combined


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
        "## Metric info\nSeq Number: 2\nName: Pastureland\n\n"
        "## Direct evidence\n\nNone.\n\n## References\n\nNone.\n",
        encoding="utf-8",
    )
    (reports / "0001.md").write_text(
        "## Metric info\nSeq Number: 1\nName: Cropland\n\n"
        f"{formatted}\n\n## References\n\nNone.\n",
        encoding="utf-8",
    )
    output = tmp_path / "el-nino-KEN.pdf"
    written = build_research_pdf(input_dir=reports, output_path=output)
    assert written == output
    assert output.is_file()
    assert output.read_bytes()[:4] == b"%PDF"
    pdf_bytes = output.read_bytes()
    import pymupdf

    document = pymupdf.open(output)  # type: ignore[no-untyped-call]
    pdf_text = "\n".join(page.get_text() for page in cast(Any, document))
    assert document.page_count >= 2
    assert "Cropland" in pdf_text
    assert "Pastureland" in pdf_text
    assert b"/Subtype /Image" in pdf_bytes


def test_markdown_to_pdf_renders_fourteen_column_table(tmp_path: Path) -> None:
    import pymupdf

    headers = [
        "DisNo.",
        "Start Year",
        "Start Month",
        "Start Day",
        "End Year",
        "End Month",
        "End Day",
        "Disaster Type",
        "Disaster Subtype",
        "Event Name",
        "Location",
        "Regions",
        "Value",
        "Unit",
    ]
    values = [
        "2025-0471-GTM",
        "2025",
        "6",
        "17",
        "2025",
        "6",
        "20",
        "Storm",
        "Tropical cyclone",
        "Hurricane Erik",
        "Jalpatagua and Chacaya municipalities",
        "Guatemala; Escuintla; Alta Verapaz",
        "16",
        "persons",
    ]
    markdown_table = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            "| " + " | ".join(values) + " |",
        ]
    )
    output = tmp_path / "wide-table.pdf"

    written = markdown_to_pdf(markdown_table, output)

    assert written == output
    assert output.read_bytes()[:4] == b"%PDF"
    assert output.stat().st_size > 0
    document = pymupdf.open(output)  # type: ignore[no-untyped-call]
    pdf_text = "\n".join(page.get_text() for page in cast(Any, document))
    assert document.page_count == 1
    assert "Event" in pdf_text
    assert "2025-6-17 to" in pdf_text
    assert "2025-6-20" in pdf_text
    assert "Hurricane Erik" in pdf_text
    assert "16 persons" in pdf_text


def test_build_research_pdf_makes_local_links_clickable(tmp_path: Path) -> None:
    import pymupdf

    reports = tmp_path / "reports"
    fao_data = reports / "fao_data"
    fao_data.mkdir(parents=True)
    local_pdf = fao_data / "My Doc.pdf"
    local_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    (reports / "0001.md").write_text(
        "## Metric info\nSeq Number: 1\nName: Metric\n\n"
        "## References\n\n1. [local](fao_data/My%20Doc.pdf)\n"
        "2. [web](https://example.com/a).\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.pdf"
    build_research_pdf(input_dir=reports, output_path=output)

    from urllib.parse import unquote

    document = pymupdf.open(output)  # type: ignore[no-untyped-call]
    links = [link for page in cast(Any, document) for link in page.get_links()]
    uris = {link.get("uri") for link in links if link.get("uri")}
    files = {unquote(str(link.get("file"))) for link in links if link.get("file")}
    assert "https://example.com/a" in uris
    # Portable relative target (fao_data/ beside report.pdf), not absolute file://.
    assert "fao_data/My Doc.pdf" in files
    assert not any(path.startswith("/") for path in files)
    assert not any(str(uri).startswith("file:") for uri in uris)
    assert b"/GoToR" not in output.read_bytes()
    assert str(local_pdf.resolve()).encode() not in output.read_bytes()


def test_build_research_pdf_ascii_folds_non_ascii_local_pdf_names(
    tmp_path: Path,
) -> None:
    from urllib.parse import quote, unquote

    import pymupdf

    reports = tmp_path / "reports"
    fao_data = reports / "fao_data"
    fao_data.mkdir(parents=True)
    original = "Haiti \u2012 El Nin\u0303o Response Plan.pdf"
    local_pdf = fao_data / original
    local_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    href = quote(f"fao_data/{original}", safe="/")
    (reports / "0001.md").write_text(
        "## Metric info\nSeq Number: 1\nName: Metric\n\n"
        f"## References\n\n1. [local]({href})\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.pdf"
    build_research_pdf(input_dir=reports, output_path=output)

    document = pymupdf.open(output)  # type: ignore[no-untyped-call]
    files = {
        unquote(str(link.get("file")))
        for page in cast(Any, document)
        for link in page.get_links()
        if link.get("file")
    }
    assert "fao_data/Haiti - El Nino Response Plan.pdf" in files


def test_format_researcher_result_has_parseable_evidence_and_numbered_citations() -> (
    None
):
    output = ResearcherOutput(
        status="answered",
        country="Kenya",
        metric_name="Cropland",
        final_summary="unused",
        statements=[
            AnswerStatement(
                statement_id="stmt_001",
                text="Cropland affected by drought reached 35%.",
                statement_type="answer",
                supporting_claim_ids=["claim_001"],
            )
        ],
        claims=[
            EvidenceClaim(
                claim_id="claim_001",
                source_type="vectorstore",
                source_id="vs:1:2",
                quoted_text="Cropland affected by drought reached 35%.",
                country="Kenya",
                relevance="direct",
                statement_type="answer",
                answer_fit="direct_requested_unit",
                url="https://fao.org/doc.pdf",
                page_number=3,
            ),
            EvidenceClaim(
                claim_id="claim_002",
                source_type="web",
                source_id="web:1",
                quoted_text="The drought also delayed planting.",
                country="Kenya",
                relevance="context",
                statement_type="context",
                url="https://example.org/drought",
            ),
            EvidenceClaim(
                claim_id="claim_003",
                source_type="vectorstore",
                source_id="vs:1:2",
                quoted_text="Planting was also delayed.",
                country="Kenya",
                relevance="context",
                statement_type="context",
                url="https://fao.org/doc.pdf",
            ),
        ],
        sources=[
            SourceReference(
                source_id="vs:1:2",
                source_type="vectorstore",
                document_uri="https://fao.org/doc.pdf",
                document_name="Kenya Report",
                page_number=3,
                document_source="PdfEvidencePipeline",
                source_text=(
                    "  - 5 -  \n"
                    "\n"
                    "\n"
                    "  By contrast, five out of nine.  \n"
                    "\n"
                    "\n"
                    "\n"
                    "Figure 3: caption\n"
                    "\n"
                    "The case of 1997.  "
                ),
                evidence_id="evidence-001",
                physical_pages=[3, 4],
                printed_pages=["1", "2"],
                events=[
                    PdfEvidenceEvent(
                        event_id="el_nino_2015_16",
                        relationship="associated",
                    )
                ],
                verified_visual_facts=[
                    PdfVerifiedVisualFact(text="The chart labels 35%.")
                ],
            ),
            SourceReference(
                source_id="web:1",
                source_type="web",
                document_uri="https://example.org/drought",
                document_name="Drought update",
                source_text="The drought also delayed planting.",
            ),
        ],
        open_gaps=[],
        research_iterations=1,
    )
    body, refs = format_researcher_result(output)
    assert body.startswith("## Direct evidence")
    assert "### Direct Evidence 1" in body
    assert "Evidence id: evidence-001" in body
    assert (
        "Events: el_nino_2015_16, associated\n\n"
        "Source text:\n\n```\n"
        "- 5 -\n"
        "\n"
        "By contrast, five out of nine.\n"
        "\n"
        "Figure 3: caption\n"
        "\n"
        "The case of 1997.\n"
        "```\n\n"
        "Verified visual facts:\n"
        "- The chart labels 35%."
    ) in body
    assert body.index("Events: el_nino_2015_16, associated") < body.index(
        "Source text:"
    )
    assert body.index("Source text:") < body.index("Verified visual facts:")
    assert body.index("Source physical pages: 3, 4") < body.index("Source text:")
    assert "Source url: https://fao.org/doc.pdf" in body
    assert "### Indirect Evidence 1" in body
    assert "### Indirect Evidence 2" not in body
    assert "Source: https://example.org/drought" in body
    assert "Source title: Drought update" in body
    answer = body.split("## Answer\n\n", maxsplit=1)[1]
    assert answer == "Cropland affected by drought reached 35%. [1]"
    assert "](https://" not in answer
    assert refs == [
        (
            "[Kenya Report, physical pages 3, 4](https://fao.org/doc.pdf) "
            "(PdfEvidencePipeline)"
        ),
        "[Drought update](https://example.org/drought) (web search)",
    ]


def test_format_researcher_result_uses_relative_local_pdf_link() -> None:
    output = ResearcherOutput(
        status="answered",
        country="Somalia",
        metric_name="Flood impacts",
        final_summary="Flooding affected farms.",
        statements=[
            AnswerStatement(
                statement_id="stmt_001",
                text="Flooding affected 12% of farms.",
                supporting_claim_ids=["claim_001"],
            )
        ],
        claims=[
            EvidenceClaim(
                claim_id="claim_001",
                source_type="vectorstore",
                source_id="pdf:1",
                quoted_text="Flooding affected 12% of farms.",
                country="Somalia",
                relevance="direct",
                statement_type="answer",
                answer_fit="direct_requested_unit",
                url="file://fao_data/El%20Ni%C3%B1o%20Plan.pdf",
            )
        ],
        sources=[
            SourceReference(
                source_id="pdf:1",
                source_type="vectorstore",
                document_uri="file://fao_data/El%20Ni%C3%B1o%20Plan.pdf",
                document_name="El Niño Plan",
                page_number=2,
                document_source="PdfEvidencePipeline",
                source_text="Flooding affected 12% of farms.",
                evidence_id="e1",
                physical_pages=[2],
            )
        ],
        open_gaps=[],
        research_iterations=1,
    )

    _, refs = format_researcher_result(output)

    assert refs == [
        (
            "[El Niño Plan, physical pages 2]"
            "(fao_data/El%20Ni%C3%B1o%20Plan.pdf) "
            "(PdfEvidencePipeline)"
        )
    ]


def test_format_researcher_result_cannot_answer() -> None:
    output = ResearcherOutput(
        status="cannot_answer",
        country="Kenya",
        metric_name="Cropland",
        final_summary="unused",
        statements=[],
        claims=[],
        sources=[],
        open_gaps=[],
        research_iterations=1,
    )
    body, _refs = format_researcher_result(output)
    assert "## Direct evidence\n\nNone." in body
    assert "## Indirect evidence\n\nNone." in body
    assert "cannot answer this metric quantitatively for Kenya" in body


def test_format_researcher_result_rejects_legacy_vector_provenance() -> None:
    output = ResearcherOutput(
        status="cannot_answer",
        country="Kenya",
        metric_name="Cropland",
        final_summary="unused",
        statements=[],
        claims=[
            EvidenceClaim(
                claim_id="claim_001",
                source_type="vectorstore",
                source_id="legacy:1",
                quoted_text="Cropland losses reached 18%.",
                country="Kenya",
                relevance="direct",
                statement_type="answer",
                url="file://legacy.pdf",
            )
        ],
        sources=[
            SourceReference(
                source_id="legacy:1",
                source_type="vectorstore",
                document_uri="file://legacy.pdf",
                document_name="Legacy chunk",
            )
        ],
        open_gaps=[],
        research_iterations=1,
    )
    with pytest.raises(ValueError, match="rerun with `pdf-research`"):
        format_researcher_result(output)


def test_format_metric_section_and_report() -> None:
    metric = _metric(name="Pastureland")
    queries = format_queries_section(
        [
            QueryRunStat(
                query="Kenya El Nino cropland percentage",
                destination="vectorstore",
                results_returned=12,
                results_accepted=3,
            ),
            QueryRunStat(
                query="Kenya drought cropland affected",
                destination="web",
                results_returned=8,
                results_accepted=2,
            ),
        ]
    )
    section = format_metric_section(
        section_number=1,
        metric=metric,
        result_markdown="Some answer.",
        reference_lines=["- [Doc, p. 1](https://example.org/a.pdf)"],
        queries_markdown=queries,
    )
    assert section.startswith(
        "## Metric info\n\n"
        "Seq Number: 1\n\n"
        "Name: Pastureland\n\n"
        "Description: Desc\n\n"
        "Example: Example text\n\n"
        "Unit: %\n"
    )
    assert "Seq Number: 1" in section
    assert "Name: Pastureland" in section
    assert "Description: Desc" in section
    assert "Example: Example text" in section
    assert "Unit: %" in section
    assert "Some answer." in section
    assert "## References\n\n1. [Doc, p. 1](https://example.org/a.pdf)" in section
    assert section.index("## Metric info") < section.index("Some answer.")
    assert section.index("Some answer.") < section.index("## References")
    assert section.index("## References") < section.index("## Queries")
    assert (
        "1. [vectorstore] Kenya El Nino cropland percentage — returned: 12, accepted: 3"
    ) in section
    assert (
        "2. [web] Kenya drought cropland affected — returned: 8, accepted: 2"
    ) in section
    assert section.rstrip().endswith(
        "2. [web] Kenya drought cropland affected — returned: 8, accepted: 2"
    )

    report = build_report(
        title="El Nino research",
        country_iso3="ken",
        sections=[section],
    )
    assert report == section
    assert "# El Nino research - KEN" not in report


def test_format_queries_section_empty() -> None:
    assert format_queries_section([]) == "## Queries\n\nNone."


def test_el_nino_routing_matches_plan() -> None:
    metrics = Metric.from_use_case(Path("use-cases/el-nino.json"))
    irrigated_index = next(
        index
        for index, metric in enumerate(metrics)
        if metric.name == "Irrigated cropland"
    )
    cropland_index = next(
        index for index, metric in enumerate(metrics) if metric.name == "Cropland"
    )
    emdat_index = next(
        index
        for index, metric in enumerate(metrics)
        if metric.name == "Deaths and missing persons"
    )

    assert metric_path(metrics[0]) == "worldbank"
    assert metric_path(metrics[1]) == "worldbank"
    assert all(
        metric_path(metric) == "faostat"
        for metric in metrics[irrigated_index:cropland_index]
    )
    assert all(
        metric_path(metric) == "researcher"
        for metric in metrics[cropland_index:emdat_index]
    )
    assert all(metric_path(metric) == "emdat" for metric in metrics[emdat_index:])
