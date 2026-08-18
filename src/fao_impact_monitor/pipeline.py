"""CLI entry points for data-lake pipeline runs."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, cast

import typer

from fao_impact_monitor.agent.impact_analyzer_agent import analyze_impact
from fao_impact_monitor.agent.researcher_agent import (
    ResearchVectorStore,
    format_human_markdown,
    research,
)
from fao_impact_monitor.agent.undrr_summarizer_agent import summarize_undrr
from fao_impact_monitor.config import get_config
from fao_impact_monitor.data_lake.mongo import connect_data_lake
from fao_impact_monitor.data_lake.vectorstore import VectorStore
from fao_impact_monitor.data_source import get_data_source
from fao_impact_monitor.data_source.tellus import TellusDataSource
from fao_impact_monitor.impact_report import (
    append_undrr_source_tables,
    default_impact_analysis_md_path,
    default_undrr_report_md_path,
    enrich_structured_latest_values,
    filter_undrr_reports,
    parse_metric_report_directory,
    write_impact_analysis_files,
    write_undrr_report_files,
)
from fao_impact_monitor.metric.metric import Metric
from fao_impact_monitor.pdf_pipeline.mongo import connect_pdf_pipeline
from fao_impact_monitor.pdf_pipeline.retrieval import (
    PdfEvidenceVectorStore,
    ensure_pdf_pipeline_indexes,
)
from fao_impact_monitor.research_report import (
    MetricProcessJob,
    build_metric_process_jobs,
    build_report,
    build_research_pdf,
    default_research_dir,
    default_research_pdf_path,
    ensure_research_output_dir,
    format_metric_section,
    format_queries_section,
    format_researcher_result,
    format_structured_result,
    metric_human_report_path,
    metric_path,
    metric_report_path,
    missing_researcher_process_jobs,
    parse_countries_iso3,
    parse_metric_option,
    select_metrics,
    use_case_data_filter,
    use_case_display_name,
    write_metric_report,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_DEFAULT_USE_CASE = Path("use-cases/el-nino.json")
_DEFAULT_USE_CASES_DIR = Path("use-cases")


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
    vector_store: ResearchVectorStore | None,
    semaphore: asyncio.Semaphore,
    web_research_enabled: bool = True,
    data_source: str | None = None,
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
        if path in {
            "worldbank",
            "faostat",
            "emdat",
            "desinventar",
            "structured",
        }:
            configs = list(metric.data_sources)
            if data_source is not None:
                source_key = data_source.casefold()
                configs = [
                    config
                    for config in configs
                    if config.source.casefold() == source_key
                ]
            source_names = ", ".join(dict.fromkeys(config.source for config in configs))
            typer.echo(f"[{position}/{total}] {source_names}: {metric.name}")
            all_results: list[Any] = []
            for config in configs:
                source = get_data_source(config.source)
                results = await source.get_data(metric, config, country_iso3)
                all_results.extend(results)
            result_md, refs = format_structured_result(
                all_results,
                plot_dir=output_dir / "plots",
                plot_stem=f"{index:04d}",
            )
            queries_markdown = None
        else:
            typer.echo(f"[{position}/{total}] ResearcherAgent: {metric.name}")
            assert vector_store is not None
            output = await research(
                metric=metric,
                country_iso3=country_iso3,
                vector_store=vector_store,
                web_research_enabled=web_research_enabled,
            )
            result_md, refs = format_researcher_result(output)
            queries_markdown = format_queries_section(output.query_runs)
            human_path = metric_human_report_path(output_dir, index)
            write_metric_report(
                human_path,
                format_human_markdown(output, metric=metric, section_number=index),
            )
            typer.echo(f"Wrote human report: {human_path}")
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
            queries_markdown=queries_markdown,
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
    use_pdf_vector_store: bool = True,
    web_research_enabled: bool = True,
    data_source: str | None = None,
    ensure_indexes: bool = True,
) -> Path:
    metrics = Metric.from_use_case(use_case_path)
    try:
        selected = select_metrics(metrics, metric_indices)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if data_source is not None:
        source_key = data_source.casefold()
        selected = [
            (index, metric)
            for index, metric in selected
            if metric.data_sources
            and any(
                config.source.casefold() == source_key for config in metric.data_sources
            )
        ]
        if not selected:
            raise typer.BadParameter(f"No metrics use the {data_source!r} data source")

    structured_count = sum(
        1 for _, metric in selected if metric_path(metric) != "researcher"
    )
    research_count = len(selected) - structured_count
    title = _use_case_title(use_case_path)
    logger.info(
        "Research plan: use_case=%s country=%s selected=%s "
        "structured=%s researcher=%s output_dir=%s max_parallel=%s",
        use_case_path,
        country_iso3,
        [(i, m.name, metric_path(m)) for i, m in selected],
        structured_count,
        research_count,
        output_dir,
        max_parallel,
    )
    typer.echo(
        f"Loaded {len(metrics)} metric(s); running {len(selected)} "
        f"({structured_count} structured data, {research_count} ResearcherAgent) "
        f"for {country_iso3} with max_parallel={max_parallel}"
    )

    needs_mongo = research_count > 0
    client: Any | None = None
    vector_store: ResearchVectorStore | None = None
    if needs_mongo:
        if use_pdf_vector_store:
            logger.info("Connecting to PDF evidence vector store for ResearcherAgent")
            client = await connect_pdf_pipeline()
            if ensure_indexes and get_config().ensure_indexes:
                await ensure_pdf_pipeline_indexes()
            vector_store = PdfEvidenceVectorStore()
        else:
            logger.info("Connecting to data lake / vector store for ResearcherAgent")
            client = await connect_data_lake()
            vector_store = VectorStore()

    ensure_research_output_dir(output_dir)
    semaphore = asyncio.Semaphore(max_parallel)
    try:
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
                vector_store=vector_store,
                semaphore=semaphore,
                web_research_enabled=web_research_enabled,
                data_source=data_source,
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


def _research_process_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one metric batch in a spawned worker process."""
    _configure_logging()
    country_iso3 = str(payload["country_iso3"])
    kind = str(payload["kind"])
    metric_indices = list(payload["metric_indices"])
    started_at = perf_counter()
    try:
        asyncio.run(
            _run_research(
                use_case_path=Path(payload["use_case_path"]),
                country_iso3=country_iso3,
                metric_indices=metric_indices,
                output_dir=Path(payload["output_dir"]),
                max_parallel=max(1, len(metric_indices)),
                use_pdf_vector_store=kind == "researcher",
                data_source=payload.get("data_source"),
                ensure_indexes=False,
            )
        )
    except Exception as exc:
        logger.exception(
            "Research worker failed country=%s kind=%s metrics=%s",
            country_iso3,
            kind,
            metric_indices,
        )
        return {
            "country_iso3": country_iso3,
            "kind": kind,
            "metric_indices": metric_indices,
            "ok": False,
            "error": str(exc),
            "pid": os.getpid(),
            "elapsed_seconds": perf_counter() - started_at,
        }
    return {
        "country_iso3": country_iso3,
        "kind": kind,
        "metric_indices": metric_indices,
        "ok": True,
        "error": None,
        "pid": os.getpid(),
        "elapsed_seconds": perf_counter() - started_at,
    }


async def _ensure_pdf_pipeline_indexes_once() -> None:
    """Create PDF search indexes in the parent process before workers start."""
    client = await connect_pdf_pipeline()
    try:
        await ensure_pdf_pipeline_indexes()
    finally:
        await client.close()


def _run_research_parallel(
    *,
    use_case_path: Path,
    countries_iso3: list[str],
    output_root: Path,
    jobs: list[MetricProcessJob] | None = None,
    continue_incomplete: bool = False,
    metric_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Spawn one process per (country, metric-process-job) pair."""
    metrics = Metric.from_use_case(use_case_path)
    if not countries_iso3:
        raise typer.BadParameter("At least one ISO3 country code is required")

    shared_jobs: list[MetricProcessJob] | None = None
    if not continue_incomplete:
        shared_jobs = jobs if jobs is not None else build_metric_process_jobs(metrics)
        if metric_indices is not None:
            allowed = set(metric_indices)
            shared_jobs = [
                MetricProcessJob(
                    kind=job.kind,
                    metric_indices=[i for i in job.metric_indices if i in allowed],
                    data_source=job.data_source,
                )
                for job in shared_jobs
            ]
            shared_jobs = [job for job in shared_jobs if job.metric_indices]
        if not shared_jobs:
            raise typer.BadParameter(f"No metrics found in use-case {use_case_path}")

    payloads: list[dict[str, Any]] = []
    for country_iso3 in countries_iso3:
        output_dir = output_root / use_case_path.stem / country_iso3.upper()
        if continue_incomplete:
            country_jobs = missing_researcher_process_jobs(metrics, output_dir)
            logger.info(
                "Continue country=%s missing_text_metrics=%s",
                country_iso3.upper(),
                [job.metric_indices[0] for job in country_jobs],
            )
        else:
            country_jobs = shared_jobs or []
        for job in country_jobs:
            payloads.append(
                {
                    "use_case_path": str(use_case_path),
                    "country_iso3": country_iso3.upper(),
                    "kind": job.kind,
                    "metric_indices": list(job.metric_indices),
                    "data_source": job.data_source,
                    "output_dir": str(output_dir),
                }
            )

    if continue_incomplete:
        typer.echo(
            f"Continue: launching {len(payloads)} missing text metric process(es) "
            f"for use-case {use_case_path}"
        )
        logger.info(
            "Research-parallel continue: countries=%s total_processes=%s",
            countries_iso3,
            len(payloads),
        )
    else:
        typer.echo(
            f"Launching {len(payloads)} process(es) "
            f"({len(countries_iso3)} country(ies) × {len(shared_jobs or [])} job(s)) "
            f"for use-case {use_case_path}"
        )
        logger.info(
            "Research-parallel plan: countries=%s jobs=%s total_processes=%s",
            countries_iso3,
            [
                (job.kind, job.metric_indices, job.data_source)
                for job in (shared_jobs or [])
            ],
            len(payloads),
        )

    if not payloads:
        return []

    if get_config().ensure_indexes and any(
        payload["kind"] == "researcher" for payload in payloads
    ):
        logger.info("Ensuring PDF search indexes before spawning research workers")
        typer.echo("Ensuring PDF search indexes")
        asyncio.run(_ensure_pdf_pipeline_indexes_once())

    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(payloads),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_research_process_worker, payload): payload
            for payload in payloads
        }
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - a worker process may crash
                result = {
                    "country_iso3": payload["country_iso3"],
                    "kind": payload["kind"],
                    "metric_indices": payload["metric_indices"],
                    "ok": False,
                    "error": str(exc),
                    "pid": None,
                    "elapsed_seconds": None,
                }
            results.append(result)
            status = "ok" if result["ok"] else f"FAILED: {result['error']}"
            typer.echo(
                f"[{len(results)}/{len(payloads)}] "
                f"{result['country_iso3']} {result['kind']} "
                f"metrics={result['metric_indices']} pid={result['pid']} "
                f"{status}"
            )
    return results


async def _run_pdf_research_all_use_cases(
    *,
    use_cases_dir: Path,
    country_iso3: str,
    output_root: Path,
    max_parallel: int,
    metric_indices: list[int] | None = None,
    web_research_enabled: bool = True,
) -> list[Path]:
    """Run every discovered use-case with PDF evidence for researcher metrics."""
    use_case_paths = sorted(
        path for path in use_cases_dir.rglob("*.json") if path.is_file()
    )
    if not use_case_paths:
        raise typer.BadParameter(f"No JSON use-cases found under {use_cases_dir}")

    typer.echo(
        f"Discovered {len(use_case_paths)} use-case(s) under {use_cases_dir}; "
        f"country={country_iso3}; vector_store=pdf"
    )
    outputs: list[Path] = []
    for position, use_case_path in enumerate(use_case_paths, start=1):
        output_dir = output_root / use_case_path.stem / country_iso3.upper()
        typer.echo(f"[{position}/{len(use_case_paths)}] PDF research: {use_case_path}")
        outputs.append(
            await _run_research(
                use_case_path=use_case_path,
                country_iso3=country_iso3,
                metric_indices=metric_indices,
                output_dir=output_dir,
                max_parallel=max_parallel,
                use_pdf_vector_store=True,
                web_research_enabled=web_research_enabled,
            )
        )
    typer.echo(f"Completed {len(outputs)} use-case(s) under: {output_root}")
    return outputs


async def _run_pdf_embedding_search(
    *,
    query: str,
    country_iso3: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Search and print raw PDF embedding representations."""
    client = await connect_pdf_pipeline()
    try:
        if get_config().ensure_indexes:
            await ensure_pdf_pipeline_indexes()
        rows = await PdfEvidenceVectorStore().search_embeddings(
            query,
            countries_iso3=[country_iso3] if country_iso3 else None,
            limit=limit,
        )
    finally:
        await client.close()

    typer.echo(f"Matched {len(rows)} embedding representation(s)")
    for rank, row in enumerate(rows, start=1):
        printable = {
            "rank": rank,
            "score": row.get("score"),
            "representation_kind": row.get("representation_kind"),
            "owner_kind": row.get("owner_kind"),
            "owner_id": row.get("owner_id"),
            "evidence_id": row.get("evidence_id"),
            "section_id": row.get("section_id"),
            "document_id": row.get("document_id"),
            "document_title": row.get("document_title"),
            "document_url": row.get("document_url"),
            "chunk_index": row.get("chunk_index"),
            "countries_iso3": row.get("countries_iso3"),
            "event_ids": row.get("event_ids"),
            "enso_relationships": row.get("enso_relationships"),
            "assertion_mode": row.get("assertion_mode"),
            "embedding_text": row.get("embedding_text"),
            "chunk_text": row.get("chunk_text"),
        }
        typer.echo(json.dumps(printable, ensure_ascii=False, default=str, indent=2))
    return rows


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
        str | None,
        typer.Option(
            "--country",
            help="ISO3 country code to research (use --countries for several).",
        ),
    ] = None,
    countries: Annotated[
        str | None,
        typer.Option(
            "--countries",
            help="Comma-separated ISO3 country codes (e.g. ETH,KEN,MWI).",
        ),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            help=(
                "1-based metric number to run (repeatable), or 'undrr' for all "
                "EM-DAT / DesInventar metrics. When omitted, run all metrics."
            ),
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help=(
                "Run only metrics whose resolved data sources include this name, "
                "and fetch only that source (for example, FAOSTAT, emdat, "
                "desinventar)."
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
                "Directory for per-metric markdown reports when using --country "
                "(default: reports/<USE_CASE>/<COUNTRY>/). "
                "Ignored with --countries (uses reports/<USE_CASE>/<COUNTRY>/)."
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
    """Run structured sources and/or ResearcherAgent; write markdown reports."""
    _configure_logging()
    if country and countries:
        raise typer.BadParameter("Use --country or --countries, not both")
    if not country and not countries:
        raise typer.BadParameter("Provide --country or --countries")
    try:
        countries_iso3 = (
            [country.upper()] if country else parse_countries_iso3(cast(str, countries))
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    metrics = Metric.from_use_case(use_case)
    try:
        metric_indices = parse_metric_option(metrics, metric)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def _run_all() -> None:
        for country_iso3 in countries_iso3:
            output_dir = (
                output
                if output is not None and len(countries_iso3) == 1
                else default_research_dir(country_iso3, use_case=use_case)
            )
            await _run_research(
                use_case_path=use_case,
                country_iso3=country_iso3,
                metric_indices=metric_indices,
                output_dir=output_dir,
                max_parallel=max_parallel,
                use_pdf_vector_store=True,
                data_source=source,
            )

    asyncio.run(_run_all())


@app.command("research-parallel")
def research_parallel_command(
    countries: Annotated[
        str,
        typer.Option(
            "--countries",
            help="Comma-separated ISO3 country codes (e.g. ETH,KEN,MWI).",
        ),
    ],
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            help=(
                "1-based metric number to run (repeatable), or 'undrr' for all "
                "EM-DAT / DesInventar metrics. When omitted, run all metrics."
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
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=(
                "Root directory for per-country reports "
                "(default: reports/; results under "
                "<root>/<USE_CASE>/<COUNTRY>/)."
            ),
            path_type=Path,
        ),
    ] = Path("reports"),
    continue_incomplete: Annotated[
        bool,
        typer.Option(
            "--continue",
            help=(
                "Run only researcher (text) metrics that do not yet have a "
                "report file. Skips World Bank, FAOSTAT, EM-DAT, and "
                "DesInventar, and spawns processes only for the missing "
                "text metrics."
            ),
        ),
    ] = False,
) -> None:
    """Run all metrics across countries in separate OS processes."""
    _configure_logging()
    try:
        countries_iso3 = parse_countries_iso3(countries)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    metrics = Metric.from_use_case(use_case)
    try:
        metric_indices = parse_metric_option(metrics, metric)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    results = _run_research_parallel(
        use_case_path=use_case,
        countries_iso3=countries_iso3,
        output_root=output_root,
        continue_incomplete=continue_incomplete,
        metric_indices=metric_indices,
    )
    failed = [result for result in results if not result["ok"]]
    typer.echo(
        f"Completed {len(results) - len(failed)}/{len(results)} process(es) "
        f"under: {output_root / use_case.stem}"
    )
    if failed:
        raise typer.Exit(code=1)


@app.command("pdf-research")
def pdf_research_command(
    country: Annotated[
        str,
        typer.Option("--country", help="ISO3 country code to research."),
    ],
    metric: Annotated[
        list[int] | None,
        typer.Option(
            "--metric",
            min=1,
            help=(
                "1-based metric number to run (repeatable). "
                "When omitted, run all metrics."
            ),
        ),
    ] = None,
    use_cases_dir: Annotated[
        Path,
        typer.Option(
            "--use-cases-dir",
            help="Directory recursively containing use-case JSON files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            path_type=Path,
        ),
    ] = _DEFAULT_USE_CASES_DIR,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=(
                "Root directory for per-use-case metric reports "
                "(default: reports/; results are written to "
                "reports/<USE_CASE>/<COUNTRY>/)."
            ),
            path_type=Path,
        ),
    ] = None,
    max_parallel: Annotated[
        int,
        typer.Option(
            "--max-parallel",
            min=1,
            help="Max number of metrics to research concurrently per use-case.",
        ),
    ] = 2,
    web_research: Annotated[
        bool,
        typer.Option(
            "--web-research/--no-web-research",
            help=(
                "Append bounded WebScout evidence after PDF research "
                "(maximum five searches per metric)."
            ),
        ),
    ] = True,
) -> None:
    """Run all use-cases using PDF evidence for ResearcherAgent metrics."""
    _configure_logging()
    country_iso3 = country.upper()
    output_root = output or Path("reports")
    asyncio.run(
        _run_pdf_research_all_use_cases(
            use_cases_dir=use_cases_dir,
            country_iso3=country_iso3,
            output_root=output_root,
            max_parallel=max_parallel,
            metric_indices=metric,
            web_research_enabled=web_research,
        )
    )


@app.command("pdf-search")
def pdf_search_command(
    query: Annotated[str, typer.Argument(help="Query to search in PDF embeddings.")],
    country: Annotated[
        str | None,
        typer.Option(
            "--country",
            help="Optional ISO3 country filter.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum raw embedding representations to print.",
        ),
    ] = 20,
) -> None:
    """Print raw ranked PDF embedding matches before evidence deduplication."""
    _configure_logging()
    asyncio.run(
        _run_pdf_embedding_search(
            query=query,
            country_iso3=country.upper() if country else None,
            limit=limit,
        )
    )


async def _run_impact_report(
    *,
    use_case_path: Path,
    countries_iso3: list[str],
    input_root: Path,
    output_root: Path,
) -> list[Path]:
    """Parse metric reports and write impact-analysis markdown + PDF per country."""
    written: list[Path] = []
    failures: list[str] = []
    for country_iso3 in countries_iso3:
        iso3 = country_iso3.upper()
        input_dir = input_root / use_case_path.stem / iso3
        output_md = default_impact_analysis_md_path(
            iso3, use_case=use_case_path, output_root=output_root
        )
        try:
            reports = parse_metric_report_directory(input_dir)
            await enrich_structured_latest_values(
                reports,
                country_iso3=iso3,
                use_case_path=use_case_path,
            )
            output = await analyze_impact(
                country_iso3=iso3,
                reports=reports,
            )
            md_path, pdf_path = write_impact_analysis_files(
                markdown_text=output.markdown,
                output_md=output_md,
            )
        except Exception as exc:
            logger.exception("Impact report failed for %s", iso3)
            failures.append(f"{iso3}: {exc}")
            typer.echo(f"FAILED {iso3}: {exc}", err=True)
            continue
        written.extend([md_path, pdf_path])
        typer.echo(f"Wrote impact analysis: {md_path}")
        typer.echo(f"Wrote impact PDF: {pdf_path}")
    if failures:
        raise RuntimeError("Impact report failed for " + "; ".join(failures))
    return written


@app.command("impact-report")
def impact_report_command(
    countries: Annotated[
        str,
        typer.Option(
            "--countries",
            help="Comma-separated ISO3 country codes (e.g. ETH,KEN,MWI).",
        ),
    ],
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
    input_root: Annotated[
        Path,
        typer.Option(
            "--input-root",
            help=(
                "Root directory containing per-country metric reports "
                "(default: reports/; reads <root>/<USE_CASE>/<COUNTRY>/)."
            ),
            path_type=Path,
        ),
    ] = Path("reports"),
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=(
                "Root directory for <ISO3> impact analysis <name>.md/.pdf "
                "(default: reports/; writes <root>/<USE_CASE>/<COUNTRY>/)."
            ),
            path_type=Path,
        ),
    ] = Path("reports"),
) -> None:
    """Compile cited El Niño impact analyses from per-metric research reports."""
    _configure_logging()
    try:
        countries_iso3 = parse_countries_iso3(countries)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        written = asyncio.run(
            _run_impact_report(
                use_case_path=use_case,
                countries_iso3=countries_iso3,
                input_root=input_root,
                output_root=output_root,
            )
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {len(written)} file(s) under: {output_root / use_case.stem}")


async def _run_undrr_report(
    *,
    use_case_path: Path,
    countries_iso3: list[str],
    input_root: Path,
    output_root: Path,
) -> list[Path]:
    """Parse EM-DAT/DesInventar metric reports and write UNDRR markdown + PDF."""
    use_case_name = use_case_display_name(use_case_path)
    data_filter = use_case_data_filter(use_case_path)
    written: list[Path] = []
    failures: list[str] = []
    for country_iso3 in countries_iso3:
        iso3 = country_iso3.upper()
        input_dir = input_root / use_case_path.stem / iso3
        output_md = default_undrr_report_md_path(
            iso3, use_case=use_case_path, output_root=output_root
        )
        try:
            reports = parse_metric_report_directory(input_dir)
            undrr_reports = filter_undrr_reports(reports, use_case_path=use_case_path)
            if not undrr_reports:
                raise ValueError(
                    f"No EM-DAT/DesInventar metric reports found under {input_dir}"
                )
            output = await summarize_undrr(
                country_iso3=iso3,
                reports=undrr_reports,
                use_case_name=use_case_name,
                data_filter=data_filter,
            )
            markdown = append_undrr_source_tables(output.markdown, undrr_reports)
            md_path, pdf_path = write_undrr_report_files(
                markdown_text=markdown,
                output_md=output_md,
            )
        except Exception as exc:
            logger.exception("UNDRR report failed for %s", iso3)
            failures.append(f"{iso3}: {exc}")
            typer.echo(f"FAILED {iso3}: {exc}", err=True)
            continue
        written.extend([md_path, pdf_path])
        typer.echo(f"Wrote UNDRR report: {md_path}")
        typer.echo(f"Wrote UNDRR PDF: {pdf_path}")
    if failures:
        raise RuntimeError("UNDRR report failed for " + "; ".join(failures))
    return written


@app.command("undrr-report")
def undrr_report_command(
    countries: Annotated[
        str,
        typer.Option(
            "--countries",
            help="Comma-separated ISO3 country codes (e.g. ETH,KEN,MWI).",
        ),
    ],
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
    input_root: Annotated[
        Path,
        typer.Option(
            "--input-root",
            help=(
                "Root directory containing per-country metric reports "
                "(default: reports/; reads <root>/<USE_CASE>/<COUNTRY>/)."
            ),
            path_type=Path,
        ),
    ] = Path("reports"),
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=(
                "Root directory for <ISO3> UNDRR <name>.md/.pdf "
                "(default: reports/; writes <root>/<USE_CASE>/<COUNTRY>/)."
            ),
            path_type=Path,
        ),
    ] = Path("reports"),
) -> None:
    """Compile cited UNDRR summaries from EM-DAT / DesInventar metric reports."""
    _configure_logging()
    try:
        countries_iso3 = parse_countries_iso3(countries)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        written = asyncio.run(
            _run_undrr_report(
                use_case_path=use_case,
                countries_iso3=countries_iso3,
                input_root=input_root,
                output_root=output_root,
            )
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {len(written)} file(s) under: {output_root / use_case.stem}")


@app.command("report-pdf")
def report_pdf_command(
    country: Annotated[
        str,
        typer.Option("--country", help="ISO3 country code for the report set."),
    ],
    use_case: Annotated[
        Path,
        typer.Option(
            "--use-case",
            help="Path to use-case JSON (used for report directory and PDF name).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            path_type=Path,
        ),
    ] = _DEFAULT_USE_CASE,
    input_dir: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help=(
                "Directory with per-metric markdown "
                "(default: reports/<USE_CASE>/<COUNTRY>/)."
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
            help=(
                "Output PDF path (default from use-case "
                "report_pdf_template, e.g. reports/el-nino/ETH/ETH metrics El Nino.pdf; "
                "with --human, '... - human.pdf')."
            ),
            path_type=Path,
        ),
    ] = None,
    human: Annotated[
        bool,
        typer.Option(
            "--human",
            help=(
                "Use NNNN-H.md for researcher metrics. World Bank / FAOSTAT "
                "keep plots but drop machine Seq Number fields. "
                "Writes '... - human.pdf' by default."
            ),
        ),
    ] = False,
) -> None:
    """Combine per-metric markdown reports into a single PDF."""
    _configure_logging()
    country_iso3 = country.upper()
    source_dir = input_dir or default_research_dir(country_iso3, use_case=use_case)
    output_path = output or default_research_pdf_path(
        country_iso3, use_case=use_case, human=human
    )
    try:
        written = build_research_pdf(
            input_dir=source_dir, output_path=output_path, human=human
        )
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
