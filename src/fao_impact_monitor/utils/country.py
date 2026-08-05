"""Country code / name helpers."""

from __future__ import annotations

import pycountry


def iso3_to_country_name(iso3: str) -> str:
    """Return the official English country name for an ISO 3166-1 alpha-3 code."""
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    official = getattr(country, "official_name", None)
    return str(official or country.name)


def iso3_to_iso2(iso3: str) -> str:
    """Return the ISO 3166-1 alpha-2 code for an alpha-3 code."""
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    alpha_2 = getattr(country, "alpha_2", None)
    if not isinstance(alpha_2, str) or not alpha_2:
        raise ValueError(f"No ISO2 code for ISO3 country code: {iso3!r}")
    return alpha_2
