"""CLI entry points for data-lake pipeline runs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer

from fao_impact_monitor.data_lake.mongo import connect_data_lake
from fao_impact_monitor.data_source.tellus import TellusDataSource
from fao_impact_monitor.metric.metric import Metric

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_DEFAULT_USE_CASE = Path("use-cases/el-nino.json")


@app.callback()
def main() -> None:
    """Run data-lake pipeline commands."""


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(_run_tellus(use_case, country.upper(), max_requests))


if __name__ == "__main__":
    app()
