from collections.abc import Iterator
from typing import cast

import pytest

from fao_impact_monitor.data_source.data_source import (
    _DATA_SOURCE_REGISTRY,
    DataResult,
    DataSource,
    build_data_source,
)


class MockSource(DataSource):
    source: str = "MockSource"
    extra_field: str
    optional_flag: bool = False

    async def get_data(
        self,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        return [
            DataResult(
                source=self.source,
                citation=f"{self.extra_field}:{country_iso3}",
                metadata={
                    "extra_field": self.extra_field,
                    "optional_flag": self.optional_flag,
                    "year_start": year_start,
                    "year_end": year_end,
                },
            )
        ]


@pytest.fixture(autouse=True)
def _cleanup_transient_registry_entries() -> Iterator[None]:
    yield
    _DATA_SOURCE_REGISTRY.pop("AbstractMock", None)


def test_metaclass_registers_concrete_data_source() -> None:
    assert _DATA_SOURCE_REGISTRY["MockSource"] is MockSource


def test_metaclass_skips_abstract_data_source() -> None:
    class AbstractMock(DataSource):
        source: str = "AbstractMock"

    assert "AbstractMock" not in _DATA_SOURCE_REGISTRY


def test_metaclass_skips_source_without_default() -> None:
    class NoDefaultSource(DataSource):
        async def get_data(
            self,
            country_iso3: str,
            *,
            year_start: int | None = None,
            year_end: int | None = None,
        ) -> list[DataResult]:
            return []

    assert NoDefaultSource not in _DATA_SOURCE_REGISTRY.values()


def test_build_data_source_with_additional_fields() -> None:
    source = build_data_source(
        "MockSource",
        extra_field="indicator-code",
        optional_flag=True,
        unit="%",
    )

    mock_source = cast(MockSource, source)
    assert isinstance(source, MockSource)
    assert mock_source.source == "MockSource"
    assert mock_source.extra_field == "indicator-code"
    assert mock_source.optional_flag is True
    assert mock_source.unit == "%"


def test_build_data_source_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown data source: MissingSource"):
        build_data_source("MissingSource")
