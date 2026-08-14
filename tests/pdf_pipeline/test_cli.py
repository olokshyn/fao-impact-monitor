from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Self

from typer.testing import CliRunner

from fao_impact_monitor.pdf_pipeline import cli


def test_ingest_help_exposes_process_worker_option() -> None:
    result = CliRunner().invoke(cli.app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "--workers" in result.stdout
    assert "worker processes" in result.stdout


def test_ensure_indexes_help_exposes_recreate_option() -> None:
    result = CliRunner().invoke(cli.app, ["ensure-indexes", "--help"])

    assert result.exit_code == 0
    assert "--recreate" in result.stdout


def test_directory_runner_uses_spawned_process_pool(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {"submissions": []}

    class FakeFuture:
        def __init__(self, result: cli._WorkerResult) -> None:
            self._result = result

        def result(self) -> cli._WorkerResult:
            return self._result

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: Any) -> None:
            captured["max_workers"] = max_workers
            captured["start_method"] = mp_context.get_start_method()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self,
            function: Any,
            path: str,
            force: bool,
            position: int,
            total: int,
        ) -> FakeFuture:
            captured["submissions"].append((function, path, force, position, total))
            return FakeFuture(
                {
                    "processed": 1,
                    "skipped": 0,
                    "failed": 0,
                    "path": path,
                    "pid": 1000 + position,
                    "elapsed_seconds": 1.0,
                    "error": None,
                }
            )

    monkeypatch.setattr(cli, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(cli, "as_completed", lambda futures: futures)
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]

    result = cli._run_directory_workers(pdfs, force=True, workers=4)

    assert result == {"processed": 2, "skipped": 0, "failed": 0}
    assert captured["max_workers"] == 2
    assert captured["start_method"] == "spawn"
    assert [submission[1] for submission in captured["submissions"]] == [
        str(path) for path in pdfs
    ]
    assert all(
        submission[0] is cli._ingest_pdf_worker
        for submission in captured["submissions"]
    )


def test_worker_entrypoint_is_picklable_for_spawn() -> None:
    assert pickle.loads(pickle.dumps(cli._ingest_pdf_worker)) is cli._ingest_pdf_worker
