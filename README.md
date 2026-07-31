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

## Editor

Recommended VS Code / Cursor extensions (see `.vscode/extensions.json`):

- Ruff (format on save and import organization)
- Mypy Type Checker
