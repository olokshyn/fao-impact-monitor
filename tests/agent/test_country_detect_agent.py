"""Unit and integration tests for the country detection agent."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fao_impact_monitor.agent.country_detect_agent import (
    CountryMention,
    CountryMentionList,
    ValidationIssue,
    detect_countries,
    lookup_country,
    mentions_to_iso3,
    validate_mentions,
)
from fao_impact_monitor.config import get_config

DEBUG_ABSTRACT = (
    "This study examines how smallholder farmers in Kenya and Uganda "
    "adopt drought-tolerant crops, conservation agriculture, and improved "
    "irrigation practices. Household survey data were compared with evidence "
    "from northern Tanzania, Zambia, Malawi, and Mozambique to assess whether "
    "climate-smart agriculture improves food availability and resilience "
    "during drought. A complementary island case study from São Tomé and "
    "Príncipe evaluates agroforestry and cocoa production, while policy "
    "lessons from Brazil and India provide broader international context."
)

DEBUG_EXPECTED_ISO3 = {
    "KEN",
    "UGA",
    "TZA",
    "ZMB",
    "MWI",
    "MOZ",
    "STP",
    "BRA",
    "IND",
}

ALIAS_TEXT = (
    "Trade links Ivory Coast, Russia, the UK, Turkey, Palestine, DR Congo, "
    "Laos, Cape Verde, East Timor, Syria, Moldova, Brunei, and Micronesia."
)

ALIAS_EXPECTED_ISO3 = {
    "CIV",
    "RUS",
    "GBR",
    "TUR",
    "PSE",
    "COD",
    "LAO",
    "CPV",
    "TLS",
    "SYR",
    "MDA",
    "BRN",
    "FSM",
}


def test_lookup_country_accepts_official_names() -> None:
    assert lookup_country("Côte d'Ivoire") is not None
    assert lookup_country("Russian Federation") is not None
    assert lookup_country("Congo, The Democratic Republic of the") is not None
    assert lookup_country("not a country") is None


def test_validate_mentions_reports_substring_and_official_name_errors() -> None:
    text = "Farmers in Kenya and Uganda adopted climate-smart practices."
    mentions = [
        CountryMention(substring="Kenya", official_name="Kenya"),
        CountryMention(substring="Narnia", official_name="Kenya"),
        CountryMention(substring="Uganda", official_name="NotARealCountry"),
        CountryMention(substring="Atlantis", official_name="AlsoFake"),
    ]
    valid, issues = validate_mentions(mentions, text)
    assert [m.substring for m in valid] == ["Kenya"]
    assert len(issues) == 3
    by_sub = {issue.substring: issue for issue in issues}
    assert "substring not found in text" in by_sub["Narnia"].reasons
    assert any("not found in pycountry" in r for r in by_sub["Uganda"].reasons)
    assert len(by_sub["Atlantis"].reasons) == 2


def test_mentions_to_iso3_maps_official_names() -> None:
    iso3s, detections = mentions_to_iso3(
        [
            CountryMention(substring="Ivory Coast", official_name="Côte d'Ivoire"),
            CountryMention(
                substring="Russia",
                official_name="Russian Federation",
            ),
        ]
    )
    assert iso3s == ["CIV", "RUS"]
    assert detections == ["Ivory Coast", "Russia"]


def test_validation_reprompts_for_missing_substring() -> None:
    class FakeStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: Any) -> CountryMentionList:
            self.calls += 1
            if self.calls == 1:
                return CountryMentionList(
                    mentions=[
                        CountryMention(
                            substring="Narnia",
                            official_name="Kenya",
                        ),
                        CountryMention(
                            substring="Kenya",
                            official_name="Kenya",
                        ),
                    ]
                )
            return CountryMentionList(
                mentions=[
                    CountryMention(substring="Kenya", official_name="Kenya"),
                ]
            )

    class FakeModel:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return self.structured

    text = "Evidence from Kenya shows adoption gains."
    model = FakeModel()
    iso3s, detections = asyncio.run(
        detect_countries(text, max_retries=3, model=model)  # type: ignore[arg-type]
    )
    assert iso3s == ["KEN"]
    assert detections == ["Kenya"]
    assert model.structured.calls == 2


def test_validation_reprompts_for_bad_official_name() -> None:
    class FakeStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: Any) -> CountryMentionList:
            self.calls += 1
            if self.calls == 1:
                return CountryMentionList(
                    mentions=[
                        CountryMention(
                            substring="Kenya",
                            official_name="Republic of MadeUp",
                        ),
                    ]
                )
            return CountryMentionList(
                mentions=[
                    CountryMention(substring="Kenya", official_name="Kenya"),
                ]
            )

    class FakeModel:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return self.structured

    text = "Evidence from Kenya shows adoption gains."
    model = FakeModel()
    iso3s, detections = asyncio.run(
        detect_countries(text, max_retries=3, model=model)  # type: ignore[arg-type]
    )
    assert iso3s == ["KEN"]
    assert detections == ["Kenya"]
    assert model.structured.calls == 2


def test_validation_reports_both_error_types_together() -> None:
    text = "Links between Kenya and Uganda."
    mentions = [
        CountryMention(substring="Narnia", official_name="Kenya"),
        CountryMention(substring="Uganda", official_name="Fakeland"),
    ]
    valid, issues = validate_mentions(mentions, text)
    assert valid == []
    assert len(issues) == 2
    assert all(isinstance(issue, ValidationIssue) for issue in issues)


def test_detect_countries_empty_text() -> None:
    iso3s, detections = asyncio.run(detect_countries("   ", max_retries=1))
    assert iso3s == []
    assert detections == []


def test_drops_invalid_mentions_after_retries_exhausted() -> None:
    class FakeStructured:
        async def ainvoke(self, _messages: Any) -> CountryMentionList:
            return CountryMentionList(
                mentions=[
                    CountryMention(substring="Narnia", official_name="Kenya"),
                    CountryMention(substring="Kenya", official_name="Kenya"),
                ]
            )

    class FakeModel:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, _schema: Any) -> FakeStructured:
            return self.structured

    text = "Evidence from Kenya shows adoption gains."
    iso3s, detections = asyncio.run(
        detect_countries(
            text,
            max_retries=0,
            model=FakeModel(),  # type: ignore[arg-type]
        )
    )
    assert iso3s == ["KEN"]
    assert detections == ["Kenya"]


def _require_bedrock() -> None:
    if not get_config().aws_bedrock.api_key.get_secret_value():
        pytest.skip("AWS_BEDROCK_API_KEY not configured")


def _assert_detections_in_text(
    text: str,
    iso3s: list[str],
    detections: list[str],
) -> None:
    assert len(iso3s) == len(detections)
    for span in detections:
        assert span
        assert span in text


@pytest.mark.integration
def test_live_model_debug_abstract() -> None:
    _require_bedrock()
    iso3s, detections = asyncio.run(detect_countries(DEBUG_ABSTRACT, max_retries=3))
    _assert_detections_in_text(DEBUG_ABSTRACT, iso3s, detections)
    assert DEBUG_EXPECTED_ISO3.issubset(set(iso3s))
    # Regional qualifier must not invent a non-Tanzania country for that span.
    tza_spans = [
        span for iso3, span in zip(iso3s, detections, strict=True) if iso3 == "TZA"
    ]
    assert tza_spans
    assert any("Tanzania" in span for span in tza_spans)


@pytest.mark.integration
def test_live_model_alias_names() -> None:
    _require_bedrock()
    iso3s, detections = asyncio.run(detect_countries(ALIAS_TEXT, max_retries=3))
    _assert_detections_in_text(ALIAS_TEXT, iso3s, detections)
    assert ALIAS_EXPECTED_ISO3.issubset(set(iso3s))


@pytest.mark.integration
def test_live_model_bahamas_and_gambia() -> None:
    _require_bedrock()
    text = (
        "Fisheries governance lessons from the Bahamas and the Gambia "
        "informed coastal livelihood programmes."
    )
    iso3s, detections = asyncio.run(detect_countries(text, max_retries=3))
    _assert_detections_in_text(text, iso3s, detections)
    assert {"BHS", "GMB"}.issubset(set(iso3s))


@pytest.mark.integration
def test_live_model_no_countries() -> None:
    _require_bedrock()
    text = "This methodology section describes sampling weights and survey design."
    iso3s, detections = asyncio.run(detect_countries(text, max_retries=2))
    assert iso3s == []
    assert detections == []


@pytest.mark.integration
def test_live_model_many_countries_parallel_lists() -> None:
    _require_bedrock()
    text = (
        "Comparative evidence spans Kenya, Uganda, Tanzania, Zambia, Malawi, "
        "Mozambique, Brazil, India, and Sao Tome and Principe in one synthesis."
    )
    iso3s, detections = asyncio.run(detect_countries(text, max_retries=3))
    _assert_detections_in_text(text, iso3s, detections)
    expected = {"KEN", "UGA", "TZA", "ZMB", "MWI", "MOZ", "BRA", "IND", "STP"}
    assert expected.issubset(set(iso3s))
