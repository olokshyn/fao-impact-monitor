import asyncio
from collections.abc import Iterator
from typing import cast

import pytest

from fao_impact_monitor.data_source.data_source import (
    _DATA_SOURCE_CLS_REGISTRY,
    DataResult,
    DataSource,
    get_data_source,
)
from fao_impact_monitor.data_source.data_source_config import DataSourceConfig
from fao_impact_monitor.metric import Metric


class MockDataSourceConfig(DataSourceConfig):
    extra_field: str
    optional_flag: bool = False


class MockSource(DataSource):
    source: str = "MockSource"

    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        config = MockDataSourceConfig.model_validate(data_source_config.model_dump())
        return [
            DataResult(
                source=self.source,
                citation=f"{config.extra_field}:{country_iso3}:{metric.name}",
                metadata={
                    "extra_field": config.extra_field,
                    "optional_flag": config.optional_flag,
                    "unit": config.unit,
                    "year_start": year_start,
                    "year_end": year_end,
                },
            )
        ]


@pytest.fixture(autouse=True)
def _cleanup_transient_registry_entries() -> Iterator[None]:
    yield
    _DATA_SOURCE_CLS_REGISTRY.pop("AbstractMock", None)


def test_metaclass_registers_concrete_data_source() -> None:
    assert _DATA_SOURCE_CLS_REGISTRY["MockSource"] is MockSource


def test_metaclass_skips_abstract_data_source() -> None:
    class AbstractMock(DataSource):
        source: str = "AbstractMock"

    assert "AbstractMock" not in _DATA_SOURCE_CLS_REGISTRY


def test_metaclass_skips_annotation_only_source() -> None:
    class AnnotationOnly(DataSource):
        source: str

        async def get_data(
            self,
            metric: Metric,
            data_source_config: DataSourceConfig,
            country_iso3: str,
            *,
            year_start: int | None = None,
            year_end: int | None = None,
        ) -> list[DataResult]:
            return []

    assert "AnnotationOnly" not in _DATA_SOURCE_CLS_REGISTRY
    assert AnnotationOnly not in _DATA_SOURCE_CLS_REGISTRY.values()


def test_metaclass_requires_source_string() -> None:
    with pytest.raises(TypeError, match="Source must be a string"):

        class NoDefaultSource(DataSource):
            source = 123  # type: ignore[assignment]

            async def get_data(
                self,
                metric: Metric,
                data_source_config: DataSourceConfig,
                country_iso3: str,
                *,
                year_start: int | None = None,
                year_end: int | None = None,
            ) -> list[DataResult]:
                return []


def test_get_data_source_returns_instance() -> None:
    source = get_data_source("MockSource")

    assert isinstance(source, MockSource)
    assert source.source == "MockSource"


def test_get_data_source_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown data source: MissingSource"):
        get_data_source("MissingSource")


def test_get_data_uses_metric_and_config() -> None:
    source = cast(MockSource, get_data_source("MockSource"))
    metric = Metric(
        name="Test metric",
        description="A test metric",
        example="Example finding",
        unit="%",
        data_sources=[
            MockDataSourceConfig(
                source="MockSource",
                extra_field="indicator-code",
                optional_flag=True,
                unit="%",
            )
        ],
    )
    config = metric.data_sources[0]

    results = asyncio.run(
        source.get_data(
            metric,
            config,
            "KEN",
            year_start=2020,
            year_end=2021,
        )
    )

    assert len(results) == 1
    assert results[0].citation == "indicator-code:KEN:Test metric"
    assert results[0].metadata == {
        "extra_field": "indicator-code",
        "optional_flag": True,
        "unit": "%",
        "year_start": 2020,
        "year_end": 2021,
    }
