"""Unit tests for country helpers."""

import pytest

from fao_impact_monitor.utils.country import iso3_to_country_name


def test_iso3_to_country_name_kenya() -> None:
    assert iso3_to_country_name("KEN") == "Republic of Kenya"


def test_iso3_to_country_name_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown ISO3"):
        iso3_to_country_name("ZZZ")
