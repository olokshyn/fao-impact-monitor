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
```

This creates `.venv` if needed and installs runtime plus `dev` dependencies (`ruff`, `mypy`, `pytest`).

## Running

Run the CLI entry point defined in `pyproject.toml`:

```bash
uv run fao-impact-monitor
```

Always use `uv run` for Python commands and scripts so they execute in the project environment.
Do not use system `python` or `pip`.

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

## Adding a data source

New providers implement the `DataSource` interface in
`src/fao_impact_monitor/data_source/data_source.py`. There is one `DataSource`
instance per provider type (for example one `WorldBank`, one FAOSTAT). Provider
parameters such as a World Bank indicator live on a `DataSourceConfig` subclass
read from the metrics definition, not on the `DataSource` itself.

Concrete subclasses are registered automatically by the `source` class attribute
and constructed with `build_data_source(source)`.

### Implement the interface

1. Add a module under `src/fao_impact_monitor/data_source/` (see
   `world_bank.py` as the reference implementation).
2. Subclass `DataSource` and set a unique `source: str = "..."` class attribute.
3. Subclass `DataSourceConfig` for any provider-specific fields (for example
   `indicator` on `WorldBankDataSourceConfig`).
4. Implement
   `async def get_data(self, metric, data_source_config, country_iso3, *, year_start=None, year_end=None) -> list[DataResult]`.
   Validate `data_source_config` into your config subclass inside `get_data`.
5. Return `DataResult` (or a subclass) with `source`, `citation`, `metadata`,
   and optional `document` / `url`. Attach provider-specific payload fields on a
   subclass when needed (for example `WorldBankDataResult.data`).
6. Export the classes from `src/fao_impact_monitor/data_source/__init__.py` so
   the source is imported and registered.

Minimal shape:

```python
from fao_impact_monitor.data_source import DataResult, DataSource, DataSourceConfig
from fao_impact_monitor.metric import Metric


class MySourceConfig(DataSourceConfig):
    # provider-specific fields...
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

- Cover registration via `build_data_source` (no provider kwargs on the source).
- Cover `get_data` behaviour with a `Metric`, a source config, and mocked HTTP /
  SDK calls (no network).
- Include **at least one** `@pytest.mark.integration` test that calls the live
  API and asserts a real response shape (non-empty data, expected metadata,
  citations, etc.).

## Use-case metrics and data sources

Use cases live under `use-cases/` as JSON (see `use-cases/el-nino.json`).
Each file describes an event scenario, its metrics, and which data sources feed
each metric. Metrics map to the `Metric` model in
`src/fao_impact_monitor/metric/metric.py`; each `data_sources` entry maps to
`DataSourceConfig` (or a subclass).

Top-level fields:

| Field         | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| `name`        | Use-case title                                       |
| `description` | Short description of the event / scenario            |
| `config`      | Optional flags (for example `use_all_fao_data_sources`) |
| `metrics`     | List of livelihood metrics to estimate               |

Each metric object:

| Field          | Meaning                                                         |
| -------------- | --------------------------------------------------------------- |
| `name`         | Metric display name                                             |
| `description`  | What the metric measures                                        |
| `example`      | Example phrasing of an impact finding                           |
| `unit`         | Unit for the metric (for example `%`)                           |
| `data_sources` | List of source configs used to fetch evidence for this metric   |

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
