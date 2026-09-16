"""Unit tests for metric-report parsing and impact-analysis helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fao_impact_monitor.data_source.faostat import FAOSTATDataResult
from fao_impact_monitor.data_source.world_bank import WorldBankDataResult
from fao_impact_monitor.impact_report import (
    construct_reference_line,
    default_impact_analysis_md_path,
    default_impact_analysis_pdf_path,
    default_undrr_report_md_path,
    default_undrr_report_pdf_path,
    impact_analysis_stem,
    load_plot_image_bytes,
    parse_metric_report_directory,
    parse_metric_report_file,
    render_impact_markdown,
    undrr_metric_seq_numbers,
    undrr_report_stem,
)
from fao_impact_monitor.research_report import (
    format_structured_result,
    format_worldbank_result,
)


def test_impact_analysis_paths_start_with_iso3() -> None:
    assert impact_analysis_stem("eth", use_case="el-nino") == (
        "ETH impact analysis El Nino"
    )
    md = default_impact_analysis_md_path("eth", use_case="el-nino")
    pdf = default_impact_analysis_pdf_path("ETH", use_case="el-nino")
    assert md.name == "ETH impact analysis El Nino.md"
    assert pdf.name == "ETH impact analysis El Nino.pdf"
    assert md.parent.name == "ETH"


def test_undrr_report_paths_ascii_fold_use_case_name() -> None:
    assert undrr_report_stem("eth", use_case="el-nino") == "ETH UNDRR El Nino"
    md = default_undrr_report_md_path("eth", use_case="el-nino")
    pdf = default_undrr_report_pdf_path("ETH", use_case="el-nino")
    assert md.name == "ETH UNDRR El Nino.md"
    assert pdf.name == "ETH UNDRR El Nino.pdf"
    assert md.parent.name == "ETH"


def test_undrr_metric_seq_numbers_use_undrr_tag() -> None:
    selected = undrr_metric_seq_numbers("use-cases/el-nino.json")
    assert selected == {26, 27, 28, 29, 30, 31}


def test_truncate_markdown_table_keeps_header_and_twenty_rows() -> None:
    from fao_impact_monitor.impact_report import truncate_markdown_table

    header = "| A | B | C |"
    sep = "| --- | --- | --- |"
    rows = [f"| {i} | x | y |" for i in range(25)]
    table = "\n".join([header, sep, *rows])
    truncated = truncate_markdown_table(table, max_data_rows=20)
    lines = truncated.splitlines()
    assert lines[0] == header
    assert lines[1] == sep
    assert len(lines) == 2 + 20 + 1  # header, sep, 20 data, ending
    assert lines[-1] == "| ... | ... | ... |"
    assert "| 19 | x | y |" in truncated
    assert "| 20 | x | y |" not in truncated


def test_append_undrr_source_tables_includes_each_dataset(
    tmp_path: Path,
) -> None:
    from fao_impact_monitor.impact_report import append_undrr_source_tables

    plot = _tiny_png(tmp_path / "plots" / "0026-desinventar-1.png")
    report = _write(
        tmp_path / "0026.md",
        f"""## Metric info

Seq Number: 26

Name: Deaths and missing persons

Description: d

Example: e

Unit: persons

## Direct evidence

### Direct Evidence 1

Source: muertos

Dataset: DesInventar

Indicator: muertos

Source data:

| Year | Value |
| --- | --- |
| 2024 | 1 |
| 2023 | 2 |
| 2022 | 3 |

Plot: ![{plot.stem}](plots/{plot.name})

### Direct Evidence 2

Source: Total Deaths

Dataset: EM-DAT

Indicator: Total Deaths

Source data:

| Year | Total Deaths |
| --- | --- |
| 2024 | 10 |
| 2023 | 20 |

## Indirect evidence

None.
""",
    )
    parsed = parse_metric_report_file(report)
    markdown = (
        "# Ethiopia: UNDRR El Nino\n\n"
        "## Deaths and missing persons\n\n"
        "Floods killed people. [1]\n\n"
        "## References\n\n"
        "1. example\n"
    )
    combined = append_undrr_source_tables(markdown, [parsed], max_data_rows=20)
    assert "**DesInventar / muertos**" in combined
    assert "**EM-DAT / Total Deaths**" in combined
    assert "| 2024 | 1 |" in combined
    assert "| 2024 | 10 |" in combined
    assert combined.index("DesInventar") < combined.index("## References")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _tiny_png(path: Path) -> Path:
    # Minimal valid 1x1 PNG.
    data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_parse_worldbank_block_and_construct_reference(tmp_path: Path) -> None:
    plot = _tiny_png(tmp_path / "plots" / "0001-worldbank-1.png")
    report = _write(
        tmp_path / "0001.md",
        f"""## Metric info

Seq Number: 1

Name: Agriculture share of GDP

Description: The share of agriculture in the total GDP of a country.

Example: Agriculture contributed 24.3% of GDP in 2023.

Unit: %

## Direct evidence

### Direct Evidence 1

Source: Agriculture, forestry, and fishing, value added (% of GDP)

Indicator: NV.AGR.TOTL.ZS

Source url: https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=ET

Latest value: 32.8 % (2025)

Plot: ![Agriculture, forestry, and fishing, value added (% of GDP)](plots/{plot.name})

## Indirect evidence

None.

## References

1. ignore this line
""",
    )
    parsed = parse_metric_report_file(report)
    assert parsed.meta.name == "Agriculture share of GDP"
    assert len(parsed.direct) == 1
    evidence = parsed.direct[0]
    assert evidence.source_type == "worldbank"
    assert evidence.indicator == "NV.AGR.TOTL.ZS"
    assert evidence.latest_value == "32.8 %"
    assert evidence.latest_year == 2025
    assert evidence.plot_path == plot.resolve()
    assert load_plot_image_bytes(evidence) == plot.read_bytes()
    ref = construct_reference_line(evidence, country_iso3="ETH")
    assert ref.startswith(
        "[Agriculture, forestry, and fishing, value added (% of GDP)]("
    )
    assert "World Bank indicator `NV.AGR.TOTL.ZS`" in ref
    assert "ignore this line" not in ref


def test_parse_pdf_and_web_blocks_construct_refs_without_references_section(
    tmp_path: Path,
) -> None:
    report = _write(
        tmp_path / "0016.md",
        """## Metric info

Seq Number: 16

Name: Crop yield loss

Description: Crop yield loss

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Evidence id: abc:s001:e001

Source: Ethiopia SITUATION REPORT ? April 2016

Source url: file://fao_data/Ethiopia%20-%20Situation%20Report%20April%202016.pdf

Source physical pages: 1

Source printed pages: 1

Events: el_nino_2015_16, attributed

Source text:

```
Some regions experiencing between 50 and 90 percent crop loss.
```

### Direct Evidence 2

Source: https://fews.net/east-africa/ethiopia

Source text:

```
Belg production was estimated at 35% below average.
```

## Indirect evidence

None.

## Answer

Some regions reported crop losses of 50% to 90%. [1]

## References

1. DO NOT USE THIS
""",
    )
    parsed = parse_metric_report_file(report)
    assert parsed.answer is not None
    assert "50% to 90%" in parsed.answer
    pdf = parsed.direct[0]
    web = parsed.direct[1]
    assert pdf.source_type == "pdf"
    assert pdf.source_text is not None
    assert "50 and 90 percent" in pdf.source_text
    assert pdf.physical_pages == [1]
    pdf_ref = construct_reference_line(pdf)
    assert (
        "[Ethiopia SITUATION REPORT ? April 2016, p. 1]"
        "(fao_data/Ethiopia%20-%20Situation%20Report%20April%202016.pdf)"
    ) == pdf_ref
    assert "DO NOT USE" not in pdf_ref
    assert web.source_type == "web"
    web_ref = construct_reference_line(web)
    assert web_ref == (
        "[https://fews.net/east-africa/ethiopia](https://fews.net/east-africa/ethiopia)"
    )


def test_scraped_markdown_images_are_not_plots(tmp_path: Path) -> None:
    report = _write(
        tmp_path / "0016.md",
        """## Metric info

Seq Number: 16

Name: Crop yield loss

Description: Crop yield loss

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Source: https://www.fao.org/4/x1101e/x1101e00.htm

Source title: FAO/GIEWS Special report on Nicaragua, 5 February 1999

Source text:

```
# SPECIAL REPORT
![](blubvsps.gif)
Losses incurred represent about 35 percent of expected output.
```

## Indirect evidence

None.
""",
    )
    parsed = parse_metric_report_file(report)
    evidence = parsed.direct[0]
    assert evidence.plot_path is None
    assert evidence.source_type == "web"


def test_unfenced_scraped_image_is_not_a_plot(tmp_path: Path) -> None:
    report = _write(
        tmp_path / "0016.md",
        """## Metric info

Seq Number: 16

Name: Crop yield loss

Description: Crop yield loss

Example: Example

Unit: %

## Direct evidence

### Direct Evidence 1

Source: https://www.fao.org/4/x1101e/x1101e00.htm

Source title: FAO/GIEWS Special report on Nicaragua, 5 February 1999

Source text:

```
![](blubvsps.gif)
truncated without a closing fence

## Indirect evidence

None.
""",
    )
    parsed = parse_metric_report_file(report)
    evidence = parsed.direct[0]
    assert evidence.plot_path is None
    assert evidence.source_type == "web"


def test_missing_plot_raises(tmp_path: Path) -> None:
    report = _write(
        tmp_path / "0001.md",
        """## Metric info

Seq Number: 1

Name: Agriculture share of GDP

Description: d

Example: e

Unit: %

## Direct evidence

### Direct Evidence 1

Source: Agriculture share of GDP

Indicator: NV.AGR.TOTL.ZS

Plot: ![Agriculture share of GDP](plots/missing.png)

## Indirect evidence

None.
""",
    )
    parsed = parse_metric_report_file(report)
    with pytest.raises(FileNotFoundError, match="Linked plot missing"):
        load_plot_image_bytes(parsed.direct[0])


def test_format_worldbank_includes_latest_value_and_source_url(tmp_path: Path) -> None:
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
    body, _refs = format_worldbank_result(
        [result],
        plot_dir=tmp_path / "plots",
        plot_stem="gdp",
    )
    assert (
        "Source url: https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=KE"
        in body
    )
    assert "Latest value: 20.5 % (2023)" in body


def test_format_faostat_includes_latest_value(tmp_path: Path) -> None:
    result = FAOSTATDataResult(
        source="FAOSTAT",
        title="Land area equipped for irrigation; share in cropland (%)",
        url="https://www.fao.org/faostat/en/#data/RL",
        citation="cite",
        metadata={
            "indicator": "Land area equipped for irrigation; share in cropland (%)",
            "unit": "%",
        },
        data=pd.DataFrame({"year": [2023, 2024], "value": [3.5, 3.67]}),
    )
    body, _refs = format_structured_result(
        [result],
        plot_dir=tmp_path / "plots",
        plot_stem="irrig",
    )
    assert "Source url: https://www.fao.org/faostat/en/#data/RL" in body
    assert "Latest value: 3.67 % (2024)" in body


def test_parse_directory_and_render_markdown(tmp_path: Path) -> None:
    _tiny_png(tmp_path / "plots" / "0001-worldbank-1.png")
    _write(
        tmp_path / "0001.md",
        """## Metric info

Seq Number: 1

Name: Agriculture share of GDP

Description: d

Example: e

Unit: %

## Direct evidence

### Direct Evidence 1

Source: Agriculture share of GDP

Indicator: NV.AGR.TOTL.ZS

Source url: https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS?locations=MW

Latest value: 30.0 % (2025)

Plot: ![Agriculture share of GDP](plots/0001-worldbank-1.png)

## Indirect evidence

None.
""",
    )
    reports = parse_metric_report_directory(tmp_path)
    assert len(reports) == 1
    markdown = render_impact_markdown(
        country_name="Malawi",
        sections={
            "Past impacts": "Agriculture employed many people. [1]",
            "Expected impacts": "Risks remain elevated. [1]",
        },
        references=["World Bank indicator"],
    )
    assert markdown.startswith("# Malawi: El Nino Risk Outlook")
    assert "## Past impacts" in markdown
    assert "## Expected impacts" in markdown
    assert "## Preparedness Considerations" not in markdown
    assert "## References" in markdown
