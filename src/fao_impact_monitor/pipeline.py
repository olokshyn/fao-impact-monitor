"""CLI entry points for data-lake pipeline runs."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from fao_impact_monitor.agent.researcher_agent import research
from fao_impact_monitor.data_lake.mongo import connect_data_lake
from fao_impact_monitor.data_lake.vectorstore import VectorStore
from fao_impact_monitor.data_source.tellus import TellusDataSource
from fao_impact_monitor.data_source.world_bank import WorldBank
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.research_report import (
    build_report,
    build_research_pdf,
    default_research_dir,
    default_research_pdf_path,
    ensure_research_output_dir,
    format_metric_section,
    format_researcher_result,
    format_worldbank_result,
    metric_path,
    metric_report_path,
    select_metrics,
    write_metric_report,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_DEFAULT_USE_CASE = Path("use-cases/el-nino.json")


@app.callback()
def main() -> None:
    """Run data-lake pipeline commands."""


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _use_case_title(use_case_path: Path) -> str:
    try:
        payload = json.loads(use_case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return use_case_path.stem
    name = payload.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return use_case_path.stem


async def _run_tellus(
    use_case_path: Path,
    country_iso3: str,
    max_requests: int,
) -> int:
    metrics = Metric.from_use_case(use_case_path)
    typer.echo(
        f"Loaded {len(metrics)} metric(s) from {use_case_path}; "
        f"country={country_iso3}; max_requests={max_requests}"
    )

    client = await connect_data_lake()
    try:
        source = TellusDataSource()
        results = await source.get_data_for_metrics(
            metrics,
            country_iso3,
            tellus_max_requests=max_requests,
        )
    finally:
        await client.close()

    typer.echo(
        f"Tellus finished: {len(results)} DataResult(s) from {len(metrics)} metric(s)"
    )
    for result in results:
        typer.echo(f"  - {result.title or '(no title)'} | {result.url}")
    return len(results)


async def _run_one_metric(
    *,
    position: int,
    total: int,
    index: int,
    metric: Metric,
    country_iso3: str,
    title: str,
    output_dir: Path,
    world_bank: WorldBank,
    vector_store: VectorStore | None,
    semaphore: asyncio.Semaphore,
) -> Path:
    """Run one metric and write ``output_dir/{index:04d}.md``."""
    async with semaphore:
        path = metric_path(metric)
        report_path = metric_report_path(output_dir, index)
        logger.info(
            "CLI metric %s/%s index=%s name=%r path=%s output=%s",
            position,
            total,
            index,
            metric.name,
            path,
            report_path,
        )
        if path == "worldbank":
            typer.echo(f"[{position}/{total}] WorldBank: {metric.name}")
            all_results: list[Any] = []
            for config in metric.data_sources:
                results = await world_bank.get_data(metric, config, country_iso3)
                all_results.extend(results)
            result_md, refs = format_worldbank_result(all_results)
        else:
            typer.echo(f"[{position}/{total}] ResearcherAgent: {metric.name}")
            assert vector_store is not None
            output = await research(
                metric=metric,
                country_iso3=country_iso3,
                vector_store=vector_store,
            )
            result_md, refs = format_researcher_result(output)
            logger.info(
                "CLI research metric done index=%s name=%r status=%s statements=%s",
                index,
                metric.name,
                output.status,
                len(output.statements),
            )

        section = format_metric_section(
            section_number=index,
            metric=metric,
            result_markdown=result_md,
            reference_lines=refs,
        )
        report = build_report(
            title=f"{title} research",
            country_iso3=country_iso3,
            sections=[section],
        )
        write_metric_report(report_path, report)
        logger.info(
            "CLI metric %s/%s complete index=%s name=%r wrote=%s",
            position,
            total,
            index,
            metric.name,
            report_path,
        )
        typer.echo(f"Wrote report: {report_path}")
        return report_path


async def _run_research(
    *,
    use_case_path: Path,
    country_iso3: str,
    metric_indices: list[int] | None,
    output_dir: Path,
    max_parallel: int,
) -> Path:
    metrics = Metric.from_use_case(use_case_path)
    try:
        selected = select_metrics(metrics, metric_indices)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    wb_count = sum(1 for _, m in selected if metric_path(m) == "worldbank")
    research_count = len(selected) - wb_count
    title = _use_case_title(use_case_path)
    logger.info(
        "Research plan: use_case=%s country=%s selected=%s "
        "worldbank=%s researcher=%s output_dir=%s max_parallel=%s",
        use_case_path,
        country_iso3,
        [(i, m.name, metric_path(m)) for i, m in selected],
        wb_count,
        research_count,
        output_dir,
        max_parallel,
    )
    typer.echo(
        f"Loaded {len(metrics)} metric(s); running {len(selected)} "
        f"({wb_count} WorldBank, {research_count} ResearcherAgent) "
        f"for {country_iso3} with max_parallel={max_parallel}"
    )

    needs_mongo = research_count > 0
    client: Any | None = None
    vector_store: VectorStore | None = None
    if needs_mongo:
        logger.info("Connecting to data lake / vector store for ResearcherAgent")
        client = await connect_data_lake()
        vector_store = VectorStore()

    ensure_research_output_dir(output_dir)
    semaphore = asyncio.Semaphore(max_parallel)
    try:
        world_bank = WorldBank()
        total = len(selected)
        tasks = [
            _run_one_metric(
                position=position,
                total=total,
                index=index,
                metric=metric,
                country_iso3=country_iso3,
                title=title,
                output_dir=output_dir,
                world_bank=world_bank,
                vector_store=vector_store,
                semaphore=semaphore,
            )
            for position, (index, metric) in enumerate(selected, start=1)
        ]
        await asyncio.gather(*tasks)
    finally:
        if client is not None:
            await client.close()

    logger.info("Wrote %s research report(s) under %s", len(selected), output_dir)
    typer.echo(f"Wrote {len(selected)} report(s) under: {output_dir}")
    return output_dir


@app.command()
def tellus(
    country: Annotated[
        str,
        typer.Option("--country", help="ISO3 country code for Tellus search."),
    ],
    use_case: Annotated[
        Path,
        typer.Option(
            "--use-case",
            help="Path to use-case JSON with a metrics list.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            path_type=Path,
        ),
    ] = _DEFAULT_USE_CASE,
    max_requests: Annotated[
        int,
        typer.Option(
            "--max-requests",
            min=1,
            help="Max concurrent Tellus search / pipeline requests.",
        ),
    ] = 4,
) -> None:
    """Run Tellus ingest for every metric in a use-case JSON file."""
    _configure_logging()
    asyncio.run(_run_tellus(use_case, country.upper(), max_requests))


@app.command("research")
def research_command(
    country: Annotated[
        str,
        typer.Option("--country", help="ISO3 country code to research."),
    ],
    metric: Annotated[
        list[int] | None,
        typer.Option(
            "--metric",
            help=(
                "1-based metric number to run (repeatable). "
                "When omitted, run all metrics."
            ),
        ),
    ] = None,
    use_case: Annotated[
        Path,
        typer.Option(
            "--use-case",
            help="Path to use-case JSON with metrics and default data_sources.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            path_type=Path,
        ),
    ] = _DEFAULT_USE_CASE,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=(
                "Directory for per-metric markdown reports "
                "(default: reports/el-nino-<COUNTRY>/). "
                "Each metric is written as {metric_index:04d}.md."
            ),
            path_type=Path,
        ),
    ] = None,
    max_parallel: Annotated[
        int,
        typer.Option(
            "--max-parallel",
            min=1,
            help="Max number of metrics to research concurrently.",
        ),
    ] = 2,
) -> None:
    """Run WorldBank and/or ResearcherAgent for use-case metrics; write markdown."""
    _configure_logging()
    country_iso3 = country.upper()
    output_dir = output or default_research_dir(country_iso3)
    asyncio.run(
        _run_research(
            use_case_path=use_case,
            country_iso3=country_iso3,
            metric_indices=metric,
            output_dir=output_dir,
            max_parallel=max_parallel,
        )
    )


@app.command("report-pdf")
def report_pdf_command(
    country: Annotated[
        str,
        typer.Option("--country", help="ISO3 country code for the report set."),
    ],
    input_dir: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help=(
                "Directory with per-metric markdown "
                "(default: reports/el-nino-<COUNTRY>/)."
            ),
            exists=False,
            file_okay=False,
            dir_okay=True,
            path_type=Path,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=("Output PDF path (default: reports/el-nino-<COUNTRY>.pdf)."),
            path_type=Path,
        ),
    ] = None,
) -> None:
    """Combine per-metric markdown reports into a single PDF."""
    _configure_logging()
    country_iso3 = country.upper()
    source_dir = input_dir or default_research_dir(country_iso3)
    output_path = output or default_research_pdf_path(country_iso3)
    try:
        written = build_research_pdf(input_dir=source_dir, output_path=output_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        logger.error("%s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    logger.info("Wrote research PDF to %s from %s", written, source_dir)
    typer.echo(f"Wrote PDF: {written}")


if __name__ == "__main__":
    app()
