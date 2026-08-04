"""Drop all collections in the local/debug MongoDB database and clear fetched_data."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from fao_impact_monitor.config import DATA_DIR
from fao_impact_monitor.data_lake.mongo import create_mongo_client, get_mongo_config

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _clear_fetched_data(data_dir: Path) -> None:
    if not data_dir.exists():
        typer.echo(f"No fetched_data directory at {data_dir}.")
        return
    removed = 0
    for child in data_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    if removed == 0:
        typer.echo(f"fetched_data already empty: {data_dir}")
    else:
        typer.echo(
            f"Cleared {removed} entr{'y' if removed == 1 else 'ies'} in {data_dir}"
        )


@app.command()
def main(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    host: Annotated[
        str | None,
        typer.Option(help="Override MONGO_HOST."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(help="Override MONGO_PORT."),
    ] = None,
    db_name: Annotated[
        str | None,
        typer.Option("--db", help="Override MONGO_DB_NAME."),
    ] = None,
) -> None:
    """Drop every collection in the target database and clear fetched_data (debug use)."""
    overrides: dict[str, str | int] = {}
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if db_name is not None:
        overrides["db_name"] = db_name
    mongo = get_mongo_config(**overrides)

    clear_mongo = yes or typer.confirm(
        f"Drop ALL collections in database {mongo.db_name!r} "
        f"at {mongo.host}:{mongo.port}?",
        default=False,
    )
    clear_data = yes or typer.confirm(
        f"Clear fetched_data directory {DATA_DIR}?",
        default=False,
    )
    if not clear_mongo and not clear_data:
        typer.echo("Nothing to do.")
        raise typer.Exit(0)

    if clear_mongo:
        client = create_mongo_client(mongo)
        try:
            db = client[mongo.db_name]
            names = db.list_collection_names()
            if not names:
                typer.echo(f"No collections in {mongo.db_name!r}.")
            else:
                for name in names:
                    if name.startswith("system."):
                        typer.echo(f"Skipping {name}")
                        continue
                    db.drop_collection(name)
                    typer.echo(f"Dropped {name}")
                typer.echo(f"Done. Cleared {mongo.db_name!r}.")
        finally:
            client.close()

    if clear_data:
        _clear_fetched_data(DATA_DIR)


if __name__ == "__main__":
    app()
