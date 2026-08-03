"""Run format, lint, type-check, and tests in one command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    no_tests: Annotated[
        bool, typer.Option("--no-tests", help="Do not run tests.")
    ] = False,
    unit_only: Annotated[
        bool,
        typer.Option(
            "--unit-only",
            help='Skip integration tests (pytest -m "not integration").',
        ),
    ] = False,
    test_stage: Annotated[
        str | None,
        typer.Option(
            "--test-stage",
            help="Run only tests for this stage (matches tests/**/test_<stage>*.py).",
        ),
    ] = None,
) -> None:
    pytest_command = _pytest_command(test_stage, unit_only=unit_only)
    commands = [
        ["ruff", "format", "."],
        ["ruff", "check", "--fix", "."],
        ["mypy", "."],
    ]
    if not no_tests:
        commands.append(pytest_command)
    for command in commands:
        print(f"+ {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise typer.Exit(result.returncode)


def _pytest_command(
    test_stage: str | None,
    *,
    unit_only: bool = False,
) -> list[str]:
    if test_stage is None:
        command = ["pytest", "tests"]
    else:
        stage = test_stage.strip()
        if not stage:
            print(
                "error: --test-stage must be a non-empty stage name",
                file=sys.stderr,
            )
            raise typer.Exit(2)
        test_files = sorted(Path("tests").rglob(f"test_{stage}*.py"))
        if not test_files:
            print(
                f"error: no tests found for stage {stage!r} "
                f"(expected tests/**/test_{stage}*.py)",
                file=sys.stderr,
            )
            raise typer.Exit(2)
        command = ["pytest", *[str(path) for path in test_files]]
    if unit_only:
        command.extend(["-m", "not integration"])
    return command


if __name__ == "__main__":
    app()
