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
