# FAO Impact Monitor

**FAO Impact Monitor (FIM)** estimates how a given event affects livelihood metrics chosen by the user.

It combines FAO statistics and documentary evidence with complementary open data to produce grounded, evidence-based impact estimates—not generic model outputs detached from sources.

## Evidence sources

| Source                                                     | Role                                                          |
| ---------------------------------------------------------- | ------------------------------------------------------------- |
| [FAO Knowledge Repository](https://openknowledge.fao.org/) | Technical publications, reports, and other FAO documents      |
| [FAOSTAT](https://www.fao.org/faostat/)                    | FAO’s corporate statistical database for food and agriculture |
| [Tellus](https://tellus.fao.org/)                          | FAO AI research agent over the Knowledge Repository           |
| AIDA                                                       | FAO AI / analytics evidence source                            |
| [World Bank Open Data](https://data.worldbank.org/)        | Development indicators and related country statistics         |

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python >= 3.13 (see `.python-version`)

## Setup

Install the project and its development tools into a local virtual environment:

```bash
uv sync
uv run install-browsers
```

This creates `.venv` if needed and installs runtime plus `dev` dependencies (`ruff`, `mypy`, `pytest`).
`install-browsers` downloads Chromium for Scrapling's browser fetcher (`uv` cannot run this as a post-install hook).
`browser_fetch` also calls it automatically if Chromium is missing.

## Running

Run the CLI entry point defined in `pyproject.toml`:

```bash
uv run fao-impact-monitor
```

Always use `uv run` for Python commands and scripts so they execute in the project environment.
Do not use system `python` or `pip`.

### Local MongoDB (debug)

For interactive debugging of Atlas Search / Vector Search (`$search`, `$vectorSearch`, `$rankFusion`), start a persistent local Atlas-compatible MongoDB:

```bash
docker compose up -d
```

Uses the `mongodb/mongodb-atlas-local` image (mongod + mongot). Data and search indexes persist in Docker volumes. Connection settings live in `MongoConfig` (`src/fao_impact_monitor/config.py`) / helpers in `data_lake/mongo.py`. Set in `.env`:

- `MONGO_USERNAME` / `MONGO_PASSWORD` (also mapped by Compose to the image init vars)
- optional `MONGO_HOST` (default `127.0.0.1`), `MONGO_PORT` (default `27018`), `MONGO_DB_NAME` (default `fao_impact_monitor`)

(Host port **27018** avoids conflicts with tools that bind `localhost:27017`, e.g. Cursor port forwarding.)

Clear the debug database and/or `fetched_data`:

```bash
uv run clear-datalake --yes
```

Unit tests do **not** use this container; they run against mongomock.

## Dependencies

```bash
uv add <package>       # add a runtime dependency
uv add --dev <package> # add a development dependency
uv remove <package>    # remove a dependency
uv sync                # reinstall from lockfile / pyproject.toml
```

## Tooling

Run these after editing code. Prefer this order: format → lint → type-check → tests.

Or run format, lint, type-check and tests together:

```bash
uv run check
```

This applies `ruff format`, runs `ruff check`, then `mypy`, then `pytest` (stops on the first failure).

Skip integration tests:

```bash
uv run check --unit-only
```

### Format and lint (`ruff`)

Apply formatting:

```bash
uv run ruff format .
```

Check formatting without writing files:

```bash
uv run ruff format --check .
```

Lint:

```bash
uv run ruff check .
```

Auto-fix lint issues where possible:

```bash
uv run ruff check --fix .
```

### Type checking (`mypy`)

Strict mode is enabled in `pyproject.toml`:

```bash
uv run mypy .
```

### Tests (`pytest`)

Place tests under `./tests`. Run them after `ruff` and `mypy`:

```bash
uv run pytest
```

Integration tests (live external APIs) are marked with `@pytest.mark.integration`. Run only unit tests with:

```bash
uv run pytest -m "not integration"
```

## Data providers vs data sources

Two layers sit under `src/fao_impact_monitor/`:

| Layer | Path | Role |
| --- | --- | --- |
| **Data providers** | `data_provider/` | Thin wrappers over external APIs. They call services such as Tellus and return raw payloads (search hits, document chunks, etc.). No metric-level interpretation. |
| **Data sources** | `data_source/` | Higher-level abstractions that answer the questions needed to compute a `Metric` (`metric/metric.py`). A `DataSource` may use AI to understand data from providers (Tellus) and other evidence pipelines (for example the PDF data lake) and return `DataResult` evidence for a metric. |

Example: `tellus_provider` searches and fetches Tellus chunks; `TellusDataSource` builds a metric/country query, starts the Tellus process pipeline on matching documents, and returns results suitable for metric computation.

## Adding a data source

New metric-facing sources implement the `DataSource` interface in
`src/fao_impact_monitor/data_source/data_source.py`. There is one `DataSource`
instance per source type (for example one `WorldBank`, one `Tellus`). Source
parameters such as a World Bank indicator live on a `DataSourceConfig` subclass
read from the metrics definition, not on the `DataSource` itself.

Concrete subclasses are registered automatically by the `source` class attribute
and constructed with `get_data_source(source)`.

Low-level HTTP/API clients belong in `data_provider/` (see
`data_provider/tellus_provider.py`), not inside a `DataSource` module, when the
same API is reused by stages or multiple sources.

### Implement the interface

1. Add a module under `src/fao_impact_monitor/data_source/` (see
   `world_bank.py` as the reference implementation).
2. Subclass `DataSource` and set a unique `source: str = "..."` class attribute.
3. Subclass `DataSourceConfig` for any source-specific fields (for example
   `indicator` on `WorldBankDataSourceConfig`).
4. Implement
   `async def get_data(self, metric, data_source_config, country_iso3, *, year_start=None, year_end=None) -> list[DataResult]`.
   Validate `data_source_config` into your config subclass inside `get_data`.
5. Return `DataResult` (or a subclass) with `source`, `citation`, `metadata`,
   and optional `document` / `url`. Attach source-specific payload fields on a
   subclass when needed (for example `WorldBankDataResult.data`).
6. Export the classes from `src/fao_impact_monitor/data_source/__init__.py` so
   the source is imported and registered.

Minimal shape:

```python
from fao_impact_monitor.data_source import DataResult, DataSource, DataSourceConfig
from fao_impact_monitor.metric import Metric


class MySourceConfig(DataSourceConfig):
    # source-specific fields...
    ...


class MySource(DataSource):
    source: str = "MySource"

    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        config = MySourceConfig.model_validate(data_source_config.model_dump())
        ...
```

### Required tests

Every new data source must have unit tests under `./tests`. Follow
`tests/test_world_bank.py`:

- Cover registration via `get_data_source` (no provider kwargs on the source).
- Cover `get_data` behaviour with a `Metric`, a source config, and mocked HTTP /
  SDK calls (no network).
- Include **at least one** `@pytest.mark.integration` test that calls the live
  API and asserts a real response shape (non-empty data, expected metadata,
  citations, etc.).

## Adding a data lake document

Documents are Beanie models stored in MongoDB. The base class lives in
`src/fao_impact_monitor/data_lake/document.py`. Concrete document types are
registered via Beanie inheritance (`Settings.class_id` / `class_id_value`).

**Required:** every Beanie document model (including `Document` subclasses,
`StageVersion` subclasses, `Pipeline` subclasses, `ChunkEmbedding`, and any
future root collections) must be listed in
`DATA_LAKE_DOCUMENT_MODELS` in
`src/fao_impact_monitor/data_lake/mongo.py`. That list is what
`init_data_lake_beanie` / `connect_data_lake` pass to `init_beanie`. Omitting a
model means Beanie will not initialize it in apps, notebooks, or tests that use
those helpers.

### Implement the document

1. Add a module under `src/fao_impact_monitor/data_lake/documents/` (see
   `web_page_document.py` as the reference).
2. Add a new value to `DocumentType` in `document.py` if the type does not
   already exist.
3. Define a module-level importable constant for the type (for example
   `MY_DOCUMENT_TYPE`) and use it for `Settings.class_id_value`. Do not
   scatter string / enum literals; other modules should import the constant
   to avoid typos.
4. Subclass `Document` and implement the `citation` computed field.
5. Export the constant and class from
   `src/fao_impact_monitor/data_lake/documents/__init__.py`.
6. **Register the class in**
   `DATA_LAKE_DOCUMENT_MODELS` in
   `src/fao_impact_monitor/data_lake/mongo.py` (import it there and append it to
   the list).

Minimal shape:

```python
from pydantic import computed_field

from fao_impact_monitor.data_lake.document import Document, DocumentType

MY_DOCUMENT_TYPE = DocumentType.WEB_PAGE  # or your new DocumentType value


class MyDocument(Document):
    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        return f"{self.title} ({self.url})"

    class Settings:
        class_id_value = MY_DOCUMENT_TYPE
```

### Required tests

Every new document type must have unit tests under `./tests` (for example
`tests/data_lake/`). Tests are mandatory. Cover construction, `type` /
`citation`, and any type-specific fields or behaviour.

## Adding a data lake stage

Stages transform documents in a pipeline. The base classes live in
`src/fao_impact_monitor/data_lake/stage.py`:

| Class          | Role                                             |
| -------------- | ------------------------------------------------ |
| `Stage`        | Runnable pipeline step, looked up by `name`      |
| `StageResult`  | Result written onto the document after a run     |
| `StageVersion` | Immutable provenance record for stage parameters |

Concrete `Stage` and `StageResult` subclasses are registered automatically by
their `name` attribute (`get_stage(name)` resolves a `Stage`).

### Pipeline philosophy

A document tracks progress **per pipeline** via `Document.pipeline_statuses`
(a map of pipeline name → `Status`). Pipelines adopt these rules:

1. A document can participate in multiple pipelines with independent progress.
   Status is always per pipeline, never a single global flag.
2. Each document starts each pipeline in `PENDING`. `PENDING` means not
   processed. Any code that finds a document in MongoDB must also require
   `COMPLETED` for that pipeline (or stage) before treating it as done;
   otherwise it must reprocess.
3. A single `Pipeline.run()` call must drive the document to `COMPLETED` for
   that pipeline (or `FAILED` if stages do not all complete). The document
   still begins as `PENDING` / `RUNNING` because the call may fail or yield.
4. A pipeline or stage may start from one seed and discover a tree of child
   documents. `Pipeline.run()` on the root must process that whole tree before
   returning.
5. Before processing a document, a pipeline or stage looks it up in MongoDB. If
   it exists and its status for that pipeline/stage is `COMPLETED`, it is not
   reprocessed; any other status means it must be reprocessed. When a run
   starts child pipelines (for example crawl enrolling PDFs into
   `pdf_process`), that same `run()` cascades into those child pipelines and
   finishes them before returning.

### Implement the stage

1. Add a module under `src/fao_impact_monitor/data_lake/stages/`.
2. Define a module-level importable constant for the stage name (for example
   `MY_STAGE_NAME`). Use that constant for both `Stage.name` and
   `StageResult.name`. Other modules (pipelines, tests) should import the
   constant instead of repeating string literals, to avoid typos.
3. Subclass `StageResult` and set `name: str = MY_STAGE_NAME` **with a
   default value** (required for registration). Set the base `status` field
   when constructing results. Add fields for the stage’s output as needed.
4. Subclass `Stage` and set `name = MY_STAGE_NAME`. Implement `get_version`
   and `run`.
5. Optionally subclass `StageVersion` with `Settings.class_id_value` when the
   stage needs custom provenance parameters. If you add a `StageVersion`
   subclass, also register it in `DATA_LAKE_DOCUMENT_MODELS` in
   `src/fao_impact_monitor/data_lake/mongo.py`.
6. Export the constant and classes from
   `src/fao_impact_monitor/data_lake/stages/__init__.py` so they are imported
   and registered.

Minimal shape:

```python
from typing import Any

from fao_impact_monitor.data_lake.document import Document
from fao_impact_monitor.data_lake.common import Status
from fao_impact_monitor.data_lake.stage import (
    Stage,
    StageResult,
    StageVersion,
)

MY_STAGE_NAME = "my_stage"


class MyStageResult(StageResult):
    name: str = MY_STAGE_NAME
    # stage-specific output fields...


class MyStage(Stage):
    name = MY_STAGE_NAME

    async def get_version(self) -> StageVersion: ...

    async def run(
        self,
        document: Document,
        stage_params: dict[str, Any],
        prev_stages: list[StageResult],
    ) -> StageResult: ...
```

### Required tests

Every new stage must have unit tests under `./tests` (for example
`tests/data_lake/` or `tests/test_pipeline.py`). Tests are mandatory. Cover:

- Registration via `get_stage` (and `StageResult` registration by `name`).
- `run` behaviour with a document, params, and previous stage results (mock
  external I/O; no network).
- Failure / `Status` handling where relevant.

## Use-case metrics and data sources

Use cases live under `use-cases/` as JSON (see `use-cases/el-nino.json`).
Each file describes an event scenario, its metrics, and which data sources feed
each metric. Metrics map to the `Metric` model in
`src/fao_impact_monitor/metric/metric.py`; each `data_sources` entry maps to
`DataSourceConfig` (or a subclass).

Top-level fields:

| Field         | Meaning                                                 |
| ------------- | ------------------------------------------------------- |
| `name`        | Use-case title                                          |
| `description` | Short description of the event / scenario               |
| `config`      | Optional flags (for example `use_all_fao_data_sources`) |
| `metrics`     | List of livelihood metrics to estimate                  |

Each metric object:

| Field          | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `name`         | Metric display name                                           |
| `description`  | What the metric measures                                      |
| `example`      | Example phrasing of an impact finding                         |
| `unit`         | Unit for the metric (for example `%`)                         |
| `data_sources` | List of source configs used to fetch evidence for this metric |

Each entry in `data_sources` must include `source` matching a registered
`DataSource.source` value (for example `"WorldBank"`). Additional keys are
provider config fields on that source’s `DataSourceConfig` subclass—for
`WorldBank`, that is typically `indicator` and optional `unit`. Those configs
are passed into `get_data` together with the parent `Metric`:

```json
{
  "name": "Agriculture share of GDP",
  "description": "The share of agriculture in the total GDP of a country.",
  "example": "Agriculture contributed 24.3% of GDP in 2023.",
  "unit": "%",
  "data_sources": [
    {
      "source": "WorldBank",
      "indicator": "NV.AGR.TOTL.ZS",
      "unit": "%"
    }
  ]
}
```

To add a metric: append an object to `metrics` with the fields above and point
`data_sources` at one or more registered sources (and their parameters). To wire
a newly implemented source into a use case, use its `source` string and any
fields its config class declares.

Sources that are not yet implemented in code (for example `"GIEWS"` or
`"FAO Drought Portal"` in `el-nino.json`) can still appear in the JSON as
placeholders until their `DataSource` classes exist.

## Editor

Recommended VS Code / Cursor extensions (see `.vscode/extensions.json`):

- Ruff (format on save and import organization)
- Mypy Type Checker
