"""CLI entry points for research, reports, and vector-store search."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import typer

from fao_impact_monitor.agent.impact_analyzer_agent import analyze_impact
from fao_impact_monitor.agent.researcher_agent import (
    ResearchVectorStore,
    format_human_markdown,
    research,
)
from fao_impact_monitor.agent.undrr_summarizer_agent import summarize_undrr
from fao_impact_monitor.config import get_config
from fao_impact_monitor.data_lake.mongo import (
    DATA_LAKE_DOCUMENT_MODELS,
    create_async_mongo_client,
    get_mongo_config,
    init_data_lake_beanie,
)
from fao_impact_monitor.data_lake.vectorstore import ChunkHit, VectorStore
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
from fao_impact_monitor.pdf_pipeline.mongo import (
    PDF_PIPELINE_MODELS,
    init_pdf_pipeline_beanie,
)
from fao_impact_monitor.pdf_pipeline.retrieval import (
    PdfEvidenceVectorStore,
    ensure_pdf_pipeline_indexes,
)
from fao_impact_monitor.research_report import (
    build_report,
    build_research_pdf,
    default_research_dir,
    default_research_pdf_path,
    ensure_research_output_dir,
    filter_missing_metric_indices,
    format_metric_section,
    format_queries_section,
    format_researcher_result,
    format_structured_result,
    merge_countries_iso3,
    metric_human_report_path,
    metric_path,
    metric_report_path,
    parse_tags_option,
    resolve_metric_indices,
    select_metrics,
    use_case_data_filter,
    use_case_display_name,
    write_metric_report,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_DEFAULT_USE_CASE = Path("use-cases/el-nino.json")
_DEFAULT_REPORTS_ROOT = Path("reports")

CountriesOption = Annotated[
    str | None,
    typer.Option(
        "--countries",
        help="Comma/whitespace-separated ISO3 country codes (e.g. ETH,KEN,MWI).",
    ),
]
CountriesFileOption = Annotated[
    Path | None,
    typer.Option(
        "--countries-file",
        help="File with ISO3 codes (one per line or comma/whitespace-separated).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
]
UseCaseOption = Annotated[
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
]
MetricOption = Annotated[
    list[str] | None,
    typer.Option(
        "--metric",
        help=(
            "1-based metric number to run (repeatable). "
            "When omitted, run all metrics (subject to --tags)."
        ),
    ),
]
TagsOption = Annotated[
    str | None,
    typer.Option(
        "--tags",
        help=(
            "Comma/whitespace metric tags (any-match). "
            "When omitted, do not filter by tag."
        ),
    ),
]
OutputRootOption = Annotated[
    Path,
    typer.Option(
        "--output-root",
        help=(
            "Root directory for per-country outputs "
            "(default: reports/; writes <root>/<USE_CASE>/<COUNTRY>/)."
        ),
        path_type=Path,
    ),
]
InputRootOption = Annotated[
    Path,
    typer.Option(
        "--input-root",
        help=(
            "Root directory containing per-country metric reports "
            "(default: reports/; reads <root>/<USE_CASE>/<COUNTRY>/)."
        ),
        path_type=Path,
    ),
]
MaxParallelOption = Annotated[
    int,
    typer.Option(
        "--max-parallel",
        min=1,
        help="Max number of metrics to research concurrently per country.",
    ),
]
MaxCountriesOption = Annotated[
    int | None,
    typer.Option(
        "--max-countries",
        min=1,
        help=(
            "Max countries to process concurrently (default: all requested countries)."
        ),
    ),
]


@app.callback()
def main() -> None:
    """Run FAO Impact Monitor pipeline commands."""


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


def _resolve_countries(
    *,
    countries: str | None,
    countries_file: Path | None,
) -> list[str]:
    try:
        return merge_countries_iso3(countries=countries, countries_file=countries_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_metric_selection(
    metrics: Sequence[Metric],
    *,
    metric: list[str] | None,
    tags: str | None,
) -> list[int] | None:
    try:
        tag_list = parse_tags_option(tags)
        return resolve_metric_indices(metrics, metric_specs=metric, tags=tag_list)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


class CompositeVectorStore:
    """Search several ResearchVectorStore backends and concatenate hits."""

    def __init__(self, stores: Sequence[ResearchVectorStore]) -> None:
        if not stores:
            raise ValueError("At least one vector store is required")
        self._stores = list(stores)

    async def search(
        self,
        query: str,
        *,
        countries_iso3: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[ChunkHit]:
        results = await asyncio.gather(
            *(
                store.search(query, countries_iso3=countries_iso3, limit=limit)
                for store in self._stores
            )
        )
        hits: list[ChunkHit] = []
        for batch in results:
            hits.extend(batch)
        return hits


async def _connect_research_mongo(
    *,
    use_fao_repo: bool,
    use_tellus: bool,
    ensure_indexes: bool,
) -> tuple[Any, ResearchVectorStore | None]:
    """Open one Mongo client and initialize the requested Beanie model sets."""
    if not use_fao_repo and not use_tellus:
        return None, None

    cfg = get_mongo_config()
    client = create_async_mongo_client(cfg)
    database = client[cfg.db_name]
    models: list[type[Any]] = []
    if use_fao_repo:
        models.extend(PDF_PIPELINE_MODELS)
    if use_tellus:
        models.extend(DATA_LAKE_DOCUMENT_MODELS)
    # Prefer dedicated init helpers when only one store is needed so their
    # skip_indexes / side effects stay consistent with the rest of the codebase.
    if use_fao_repo and use_tellus:
        from beanie import init_beanie

        await init_beanie(database=database, document_models=models)
    elif use_fao_repo:
        await init_pdf_pipeline_beanie(database)
    else:
        await init_data_lake_beanie(database)

    if use_fao_repo and ensure_indexes and get_config().ensure_indexes:
        await ensure_pdf_pipeline_indexes()

    stores: list[ResearchVectorStore] = []
    if use_fao_repo:
        stores.append(PdfEvidenceVectorStore())
    if use_tellus:
        stores.append(VectorStore())
    vector_store: ResearchVectorStore | None
    if len(stores) == 1:
        vector_store = stores[0]
    elif stores:
        vector_store = CompositeVectorStore(stores)
    else:
        vector_store = None
    return client, vector_store


async def _run_tellus_ingest(
    *,
    metrics: Sequence[Metric],
    country_iso3: str,
    max_requests: int,
) -> int:
    """Ingest Tellus documents for the given metrics into the data lake."""
    mongo_client, _ = await _connect_research_mongo(
        use_fao_repo=False,
        use_tellus=True,
        ensure_indexes=False,
    )
    assert mongo_client is not None
    try:
        source = TellusDataSource()
        results = await source.get_data_for_metrics(
            list(metrics),
            country_iso3,
            tellus_max_requests=max_requests,
        )
    finally:
        await mongo_client.close()
    typer.echo(
        f"Tellus ingest finished for {country_iso3}: "
        f"{len(results)} DataResult(s) from {len(metrics)} metric(s)"
    )
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
    use_fao_repo: bool = True,
    use_tellus: bool = False,
    web_research_enabled: bool = True,
    tellus_max_requests: int = 4,
    continue_incomplete: bool = False,
    ensure_indexes: bool = True,
) -> Path:
    metrics = Metric.from_use_case(use_case_path)
    try:
        selected = select_metrics(metrics, metric_indices)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if continue_incomplete:
        before = len(selected)
        selected = filter_missing_metric_indices(selected, output_dir)
        typer.echo(
            f"Continue: {len(selected)}/{before} metric(s) still missing under "
            f"{output_dir}"
        )
        if not selected:
            typer.echo(f"Nothing to do for {country_iso3}: all selected reports exist")
            return output_dir

    structured_count = sum(
        1 for _, metric in selected if metric_path(metric) != "researcher"
    )
    research_count = len(selected) - structured_count
    title = _use_case_title(use_case_path)
    logger.info(
        "Research plan: use_case=%s country=%s selected=%s "
        "structured=%s researcher=%s fao_repo=%s tellus=%s "
        "output_dir=%s max_parallel=%s",
        use_case_path,
        country_iso3,
        [(i, m.name, metric_path(m), m.tags) for i, m in selected],
        structured_count,
        research_count,
        use_fao_repo,
        use_tellus,
        output_dir,
        max_parallel,
    )
    typer.echo(
        f"Loaded {len(metrics)} metric(s); running {len(selected)} "
        f"({structured_count} structured data, {research_count} ResearcherAgent) "
        f"for {country_iso3} with max_parallel={max_parallel}"
    )

    if use_tellus:
        await _run_tellus_ingest(
            metrics=[metric for _, metric in selected],
            country_iso3=country_iso3,
            max_requests=tellus_max_requests,
        )

    needs_mongo = research_count > 0 and (use_fao_repo or use_tellus)
    if research_count > 0 and not use_fao_repo and not use_tellus:
        raise typer.BadParameter(
            "Researcher metrics require --fao-repo and/or --tellus"
        )

    client: Any | None = None
    vector_store: ResearchVectorStore | None = None
    if needs_mongo:
        client, vector_store = await _connect_research_mongo(
            use_fao_repo=use_fao_repo,
            use_tellus=use_tellus,
            ensure_indexes=ensure_indexes,
        )

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


async def _run_countries(
    *,
    countries_iso3: list[str],
    max_countries: int | None,
    runner: Callable[[str], Awaitable[None]],
) -> list[dict[str, Any]]:
    """Run ``runner(iso3)`` for each country with bounded concurrency."""
    limit = max_countries if max_countries is not None else len(countries_iso3)
    semaphore = asyncio.Semaphore(max(1, limit))
    results: list[dict[str, Any]] = []

    async def _one(iso3: str) -> dict[str, Any]:
        started = perf_counter()
        async with semaphore:
            try:
                await runner(iso3)
            except Exception as exc:
                logger.exception("Country %s failed", iso3)
                return {
                    "country_iso3": iso3,
                    "ok": False,
                    "error": str(exc),
                    "elapsed_seconds": perf_counter() - started,
                }
            return {
                "country_iso3": iso3,
                "ok": True,
                "error": None,
                "elapsed_seconds": perf_counter() - started,
            }

    gathered = await asyncio.gather(*(_one(iso3) for iso3 in countries_iso3))
    results.extend(gathered)
    return results


def _print_country_summary(results: list[dict[str, Any]]) -> None:
    failed = [result for result in results if not result["ok"]]
    for result in results:
        status = "ok" if result["ok"] else f"FAILED: {result['error']}"
        elapsed = result.get("elapsed_seconds")
        elapsed_text = f" ({elapsed:.1f}s)" if isinstance(elapsed, float) else ""
        typer.echo(f"  {result['country_iso3']}: {status}{elapsed_text}")
    typer.echo(f"Completed {len(results) - len(failed)}/{len(results)} country(ies)")
    if failed:
        raise typer.Exit(code=1)


def _echo_research_dry_run(
    *,
    countries_iso3: list[str],
    use_case: Path,
    metrics: Sequence[Metric],
    metric_indices: list[int] | None,
    output_root: Path,
    use_fao_repo: bool,
    use_tellus: bool,
    web_research_enabled: bool,
    continue_incomplete: bool,
    max_parallel: int,
    max_countries: int | None,
) -> None:
    selected = select_metrics(list(metrics), metric_indices)
    typer.echo("Dry-run research plan:")
    typer.echo(f"  use-case: {use_case}")
    typer.echo(f"  countries: {', '.join(countries_iso3)}")
    typer.echo(f"  max-countries: {max_countries or len(countries_iso3)}")
    typer.echo(f"  max-parallel: {max_parallel}")
    typer.echo(f"  fao-repo: {use_fao_repo}")
    typer.echo(f"  tellus: {use_tellus}")
    typer.echo(f"  web-research: {web_research_enabled}")
    typer.echo(f"  continue: {continue_incomplete}")
    typer.echo(f"  metrics ({len(selected)}):")
    for index, metric in selected:
        tags = ", ".join(metric.tags) if metric.tags else "(none)"
        typer.echo(f"    {index:04d} {metric.name} [{tags}]")
    for iso3 in countries_iso3:
        output_dir = output_root / use_case.stem / iso3
        typer.echo(f"  output: {output_dir}")


@app.command("research")
def research_command(
    countries: CountriesOption = None,
    countries_file: CountriesFileOption = None,
    metric: MetricOption = None,
    tags: TagsOption = None,
    use_case: UseCaseOption = _DEFAULT_USE_CASE,
    output_root: OutputRootOption = _DEFAULT_REPORTS_ROOT,
    max_parallel: MaxParallelOption = 2,
    max_countries: MaxCountriesOption = None,
    fao_repo: Annotated[
        bool,
        typer.Option(
            "--fao-repo/--no-fao-repo",
            help="Search FAO Knowledge Repository PDF evidence (default: on).",
        ),
    ] = True,
    tellus: Annotated[
        bool,
        typer.Option(
            "--tellus/--no-tellus",
            help="Ingest Tellus and search Tellus embeddings (default: off).",
        ),
    ] = False,
    tellus_max_requests: Annotated[
        int,
        typer.Option(
            "--tellus-max-requests",
            min=1,
            help="Max concurrent Tellus search / pipeline requests.",
        ),
    ] = 4,
    web_research: Annotated[
        bool,
        typer.Option(
            "--web-research/--no-web-research",
            help="Append bounded WebScout evidence after vector-store research.",
        ),
    ] = True,
    continue_incomplete: Annotated[
        bool,
        typer.Option(
            "--continue",
            help="Skip metrics that already have a non-empty NNNN.md report.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the research plan without calling APIs.",
        ),
    ] = False,
) -> None:
    """Run structured sources and/or ResearcherAgent; write markdown reports."""
    _configure_logging()
    countries_iso3 = _resolve_countries(
        countries=countries, countries_file=countries_file
    )
    metrics = Metric.from_use_case(use_case)
    metric_indices = _resolve_metric_selection(metrics, metric=metric, tags=tags)

    if dry_run:
        _echo_research_dry_run(
            countries_iso3=countries_iso3,
            use_case=use_case,
            metrics=metrics,
            metric_indices=metric_indices,
            output_root=output_root,
            use_fao_repo=fao_repo,
            use_tellus=tellus,
            web_research_enabled=web_research,
            continue_incomplete=continue_incomplete,
            max_parallel=max_parallel,
            max_countries=max_countries,
        )
        return

    async def _runner(iso3: str) -> None:
        output_dir = output_root / use_case.stem / iso3
        await _run_research(
            use_case_path=use_case,
            country_iso3=iso3,
            metric_indices=metric_indices,
            output_dir=output_dir,
            max_parallel=max_parallel,
            use_fao_repo=fao_repo,
            use_tellus=tellus,
            web_research_enabled=web_research,
            tellus_max_requests=tellus_max_requests,
            continue_incomplete=continue_incomplete,
        )

    results = asyncio.run(
        _run_countries(
            countries_iso3=countries_iso3,
            max_countries=max_countries,
            runner=_runner,
        )
    )
    typer.echo(f"Research summary under: {output_root / use_case.stem}")
    _print_country_summary(results)


async def _run_impact_for_country(
    *,
    use_case_path: Path,
    country_iso3: str,
    input_root: Path,
    output_root: Path,
) -> list[Path]:
    iso3 = country_iso3.upper()
    input_dir = input_root / use_case_path.stem / iso3
    output_md = default_impact_analysis_md_path(
        iso3, use_case=use_case_path, output_root=output_root
    )
    reports = parse_metric_report_directory(input_dir)
    await enrich_structured_latest_values(
        reports,
        country_iso3=iso3,
        use_case_path=use_case_path,
    )
    output = await analyze_impact(country_iso3=iso3, reports=reports)
    md_path, pdf_path = write_impact_analysis_files(
        markdown_text=output.markdown,
        output_md=output_md,
    )
    typer.echo(f"Wrote impact analysis: {md_path}")
    typer.echo(f"Wrote impact PDF: {pdf_path}")
    return [md_path, pdf_path]


async def _run_undrr_for_country(
    *,
    use_case_path: Path,
    country_iso3: str,
    input_root: Path,
    output_root: Path,
) -> list[Path]:
    iso3 = country_iso3.upper()
    input_dir = input_root / use_case_path.stem / iso3
    output_md = default_undrr_report_md_path(
        iso3, use_case=use_case_path, output_root=output_root
    )
    reports = parse_metric_report_directory(input_dir)
    undrr_reports = filter_undrr_reports(reports, use_case_path=use_case_path)
    if not undrr_reports:
        raise ValueError(f"No UNDRR-tagged metric reports found under {input_dir}")
    output = await summarize_undrr(
        country_iso3=iso3,
        reports=undrr_reports,
        use_case_name=use_case_display_name(use_case_path),
        data_filter=use_case_data_filter(use_case_path),
    )
    markdown = append_undrr_source_tables(output.markdown, undrr_reports)
    md_path, pdf_path = write_undrr_report_files(
        markdown_text=markdown,
        output_md=output_md,
    )
    typer.echo(f"Wrote UNDRR report: {md_path}")
    typer.echo(f"Wrote UNDRR PDF: {pdf_path}")
    return [md_path, pdf_path]


async def _run_metrics_pdf_for_country(
    *,
    use_case_path: Path,
    country_iso3: str,
    input_root: Path,
    human: bool,
) -> Path:
    iso3 = country_iso3.upper()
    source_dir = input_root / use_case_path.stem / iso3
    if not source_dir.is_dir():
        # Fall back to default layout helper for clearer errors / legacy paths.
        source_dir = default_research_dir(iso3, use_case=use_case_path)
    output_path = default_research_pdf_path(iso3, use_case=use_case_path, human=human)
    # Keep PDFs under the requested input/output country dir when possible.
    if input_root != _DEFAULT_REPORTS_ROOT:
        output_path = source_dir / output_path.name
    written = await asyncio.to_thread(
        build_research_pdf,
        input_dir=source_dir,
        output_path=output_path,
        human=human,
    )
    label = "human metrics PDF" if human else "technical metrics PDF"
    typer.echo(f"Wrote {label}: {written}")
    return written


def _echo_report_dry_run(
    *,
    countries_iso3: list[str],
    use_case: Path,
    input_root: Path,
    output_root: Path,
    do_impact: bool,
    do_technical: bool,
    do_human: bool,
    do_undrr: bool,
    max_countries: int | None,
) -> None:
    kinds = [
        name
        for name, enabled in (
            ("impact", do_impact),
            ("technical", do_technical),
            ("human", do_human),
            ("undrr", do_undrr),
        )
        if enabled
    ]
    typer.echo("Dry-run report plan:")
    typer.echo(f"  use-case: {use_case}")
    typer.echo(f"  countries: {', '.join(countries_iso3)}")
    typer.echo(f"  max-countries: {max_countries or len(countries_iso3)}")
    typer.echo(f"  reports: {', '.join(kinds)}")
    for iso3 in countries_iso3:
        typer.echo(f"  input:  {input_root / use_case.stem / iso3}")
        typer.echo(f"  output: {output_root / use_case.stem / iso3}")


@app.command("report")
def report_command(
    countries: CountriesOption = None,
    countries_file: CountriesFileOption = None,
    use_case: UseCaseOption = _DEFAULT_USE_CASE,
    input_root: InputRootOption = _DEFAULT_REPORTS_ROOT,
    output_root: OutputRootOption = _DEFAULT_REPORTS_ROOT,
    max_countries: MaxCountriesOption = None,
    impact: Annotated[
        bool,
        typer.Option("--impact", help="Generate the cited impact analysis report."),
    ] = False,
    technical: Annotated[
        bool,
        typer.Option(
            "--technical",
            help="Combine per-metric NNNN.md reports into a technical metrics PDF.",
        ),
    ] = False,
    human: Annotated[
        bool,
        typer.Option(
            "--human",
            help="Combine human-readable metric summaries into a metrics PDF.",
        ),
    ] = False,
    undrr: Annotated[
        bool,
        typer.Option("--undrr", help="Generate the UNDRR disaster-loss summary."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the report plan without writing files."),
    ] = False,
) -> None:
    """Compile impact, UNDRR, and/or combined metrics reports."""
    _configure_logging()
    countries_iso3 = _resolve_countries(
        countries=countries, countries_file=countries_file
    )

    any_flag = impact or technical or human or undrr
    do_impact = impact or not any_flag
    do_technical = technical
    do_human = human or not any_flag
    do_undrr = undrr

    if dry_run:
        _echo_report_dry_run(
            countries_iso3=countries_iso3,
            use_case=use_case,
            input_root=input_root,
            output_root=output_root,
            do_impact=do_impact,
            do_technical=do_technical,
            do_human=do_human,
            do_undrr=do_undrr,
            max_countries=max_countries,
        )
        return

    async def _runner(iso3: str) -> None:
        if do_impact:
            await _run_impact_for_country(
                use_case_path=use_case,
                country_iso3=iso3,
                input_root=input_root,
                output_root=output_root,
            )
        if do_undrr:
            await _run_undrr_for_country(
                use_case_path=use_case,
                country_iso3=iso3,
                input_root=input_root,
                output_root=output_root,
            )
        if do_technical:
            await _run_metrics_pdf_for_country(
                use_case_path=use_case,
                country_iso3=iso3,
                input_root=input_root,
                human=False,
            )
        if do_human:
            await _run_metrics_pdf_for_country(
                use_case_path=use_case,
                country_iso3=iso3,
                input_root=input_root,
                human=True,
            )

    results = asyncio.run(
        _run_countries(
            countries_iso3=countries_iso3,
            max_countries=max_countries,
            runner=_runner,
        )
    )
    typer.echo(f"Report summary under: {output_root / use_case.stem}")
    _print_country_summary(results)


async def _run_vs_search(
    *,
    query: str,
    countries_iso3: list[str] | None,
    limit: int,
    use_fao_repo: bool,
    use_tellus: bool,
) -> list[dict[str, Any]]:
    if not use_fao_repo and not use_tellus:
        raise typer.BadParameter("Enable --fao-repo and/or --tellus for vs-search")

    client, _ = await _connect_research_mongo(
        use_fao_repo=use_fao_repo,
        use_tellus=use_tellus,
        ensure_indexes=True,
    )
    assert client is not None
    printed: list[dict[str, Any]] = []
    try:
        if use_fao_repo:
            rows = await PdfEvidenceVectorStore().search_embeddings(
                query,
                countries_iso3=countries_iso3,
                limit=limit,
            )
            typer.echo(f"FAO repo: matched {len(rows)} embedding representation(s)")
            for rank, row in enumerate(rows, start=1):
                printable = {
                    "store": "fao_repo",
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
                printed.append(printable)
                typer.echo(
                    json.dumps(printable, ensure_ascii=False, default=str, indent=2)
                )
        if use_tellus:
            hits = await VectorStore().search(
                query,
                countries_iso3=countries_iso3,
                limit=limit,
            )
            typer.echo(f"Tellus: matched {len(hits)} chunk hit(s)")
            for rank, hit in enumerate(hits, start=1):
                printable = {
                    "store": "tellus",
                    "rank": rank,
                    **hit.model_dump(mode="json"),
                }
                printed.append(printable)
                typer.echo(
                    json.dumps(printable, ensure_ascii=False, default=str, indent=2)
                )
    finally:
        await client.close()
    return printed


@app.command("vs-search")
def vs_search_command(
    query: Annotated[str, typer.Argument(help="Query to search in vector stores.")],
    countries: CountriesOption = None,
    countries_file: CountriesFileOption = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum hits to print per enabled store.",
        ),
    ] = 20,
    fao_repo: Annotated[
        bool,
        typer.Option(
            "--fao-repo/--no-fao-repo",
            help="Search FAO Knowledge Repository PDF embeddings (default: on).",
        ),
    ] = True,
    tellus: Annotated[
        bool,
        typer.Option(
            "--tellus/--no-tellus",
            help="Also search Tellus / data-lake embeddings (default: off).",
        ),
    ] = False,
) -> None:
    """Print ranked vector-store matches (FAO repo and/or Tellus)."""
    _configure_logging()
    countries_iso3: list[str] | None = None
    if countries is not None or countries_file is not None:
        countries_iso3 = _resolve_countries(
            countries=countries, countries_file=countries_file
        )
    asyncio.run(
        _run_vs_search(
            query=query,
            countries_iso3=countries_iso3,
            limit=limit,
            use_fao_repo=fao_repo,
            use_tellus=tellus,
        )
    )


if __name__ == "__main__":
    app()
