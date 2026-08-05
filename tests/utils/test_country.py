"""Unit tests for country helpers."""

import pytest

from fao_impact_monitor.utils.country import iso3_to_country_name, iso3_to_iso2


def test_iso3_to_country_name_kenya() -> None:
    assert iso3_to_country_name("KEN") == "Republic of Kenya"


def test_iso3_to_iso2_kenya() -> None:
    assert iso3_to_iso2("KEN") == "KE"


def test_iso3_to_country_name_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown ISO3"):
        iso3_to_country_name("ZZZ")
