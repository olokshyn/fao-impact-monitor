"""LangGraph agent that detects country mentions and maps them to ISO3 codes."""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, Literal

import pycountry
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from fao_impact_monitor.config import AwsBedrockConfig, CountryDetectConfig, get_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a country-detection agent for the FAO Impact Monitor data lake.

Your job is to read a piece of text and list every sovereign country that is
explicitly mentioned.

For each country mention, return:
1. substring — the minimal exact span copied VERBATIM from the text where the
   country name appears. Prefer the shortest country name span present in the
   text (e.g. "Tanzania" not "northern Tanzania"). The substring may be a
   colloquial form that appears in the text (e.g. "Ivory Coast", "UK", "Russia").
2. official_name — the official English country name as used by ISO 3166 /
   pycountry.lookup (NOT the colloquial alias). Examples:
   - Ivory Coast → Côte d'Ivoire
   - Russia → Russian Federation
   - UK / U.K. → United Kingdom
   - Turkey → Türkiye
   - Palestine → Palestine, State of
   - DR Congo / DRC / Democratic Republic of Congo → Congo, The Democratic Republic of the
   - Laos → Lao People's Democratic Republic
   - Cape Verde → Cabo Verde
   - East Timor → Timor-Leste
   - São Tomé and Príncipe / Sao Tome → Sao Tome and Principe
   - the Bahamas → Bahamas
   - the Gambia → Gambia
   - Syria → Syrian Arab Republic
   - Moldova → Moldova, Republic of
   - Brunei → Brunei Darussalam
   - Micronesia → Micronesia, Federated States of
   - Tanzania / northern Tanzania → Tanzania, United Republic of
     (or United Republic of Tanzania / Tanzania — any name pycountry accepts)

Critical rules:
1. Copy each substring EXACTLY as it appears in the text. Do not invent,
   normalize, translate, or paraphrase substrings.
2. official_name must be a name that pycountry.countries.lookup accepts.
3. Include every distinct country mention; if the same country appears twice
   with different spans, return both.
4. Skip continents, regions, cities, demonyms alone, and non-sovereign entities.
5. If no countries are mentioned, return an empty list.
"""


class CountryMention(BaseModel):
    substring: str = Field(
        description="Exact minimal span copied verbatim from the text",
    )
    official_name: str = Field(
        description="Official English country name resolvable by pycountry",
    )


class CountryMentionList(BaseModel):
    mentions: list[CountryMention] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    substring: str
    official_name: str
    reasons: list[str]


class AgentState(BaseModel):
    text: str
    mentions: list[CountryMention] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    retries_left: int = 0
    countries_iso3: list[str] = Field(default_factory=list)
    detections: list[str] = Field(default_factory=list)


def build_chat_model(
    country_detect_config: CountryDetectConfig | None = None,
    aws_bedrock_config: AwsBedrockConfig | None = None,
) -> BaseChatModel:
    config = get_config()
    country_detect_config = country_detect_config or config.country_detect
    aws_bedrock_config = aws_bedrock_config or config.aws_bedrock
    return init_chat_model(
        country_detect_config.llm_model,
        api_key=aws_bedrock_config.api_key.get_secret_value(),
        base_url=aws_bedrock_config.base_url,
        use_responses_api=True,
    )


def _normalize_country_name(name: str) -> str:
    return (
        unicodedata.normalize("NFKC", name)
        .replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
        .strip(" \t\r\n,.;:()[]{}")
    )


def lookup_country(official_name: str) -> Any | None:
    """Return a pycountry country object if ``official_name`` resolves."""
    normalized = _normalize_country_name(official_name)
    if not normalized:
        return None
    candidates = [normalized]
    if normalized.casefold().startswith("the "):
        candidates.append(normalized[len("the ") :])
    for candidate in candidates:
        try:
            return pycountry.countries.lookup(candidate)
        except LookupError:
            continue
    return None


def _user_prompt(*, text: str, issues: list[ValidationIssue]) -> str:
    parts = [
        "Detect all countries mentioned in the following text.",
        "Text follows:",
        text,
    ]
    if issues:
        lines: list[str] = []
        for issue in issues:
            reasons = "; ".join(issue.reasons)
            lines.append(
                f"- substring={issue.substring!r}, "
                f"official_name={issue.official_name!r}: {reasons}"
            )
        parts.insert(
            0,
            "Your previous country list had validation errors. Produce a new "
            "list. Fix these problems:\n" + "\n".join(lines) + "\n"
            "Only include substrings that appear verbatim in the text, and only "
            "official_name values that pycountry.countries.lookup accepts.",
        )
    return "\n\n".join(parts)


def validate_mentions(
    mentions: list[CountryMention],
    text: str,
) -> tuple[list[CountryMention], list[ValidationIssue]]:
    """Split mentions into validated vs invalid with reasons."""
    valid: list[CountryMention] = []
    issues: list[ValidationIssue] = []
    for mention in mentions:
        reasons: list[str] = []
        substring = mention.substring
        if not substring or not substring.strip():
            reasons.append("substring is empty")
        elif substring not in text:
            reasons.append("substring not found in text")
        if lookup_country(mention.official_name) is None:
            reasons.append(
                f"official_name {mention.official_name!r} not found in pycountry"
            )
        if reasons:
            issues.append(
                ValidationIssue(
                    substring=mention.substring,
                    official_name=mention.official_name,
                    reasons=reasons,
                )
            )
        else:
            valid.append(mention)
    return valid, issues


def mentions_to_iso3(
    mentions: list[CountryMention],
) -> tuple[list[str], list[str]]:
    """Map validated mentions to parallel ISO3 and substring lists."""
    iso3s: list[str] = []
    detections: list[str] = []
    for mention in mentions:
        country = lookup_country(mention.official_name)
        if country is None:
            continue
        alpha_3 = getattr(country, "alpha_3", None)
        if not isinstance(alpha_3, str):
            continue
        iso3s.append(alpha_3)
        detections.append(mention.substring)
    return iso3s, detections


def build_country_detect_agent(model: BaseChatModel) -> Any:
    """Compile a LangGraph that detects, validates, and maps country mentions."""

    structured = model.with_structured_output(CountryMentionList)

    async def detect(state: AgentState) -> dict[str, Any]:
        prompt = _user_prompt(text=state.text, issues=state.issues)
        result = await structured.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        if not isinstance(result, CountryMentionList):
            result = CountryMentionList.model_validate(result)
        logger.info("Country agent extracted %s mention(s)", len(result.mentions))
        return {"mentions": result.mentions, "issues": []}

    async def validate(state: AgentState) -> dict[str, Any]:
        valid, issues = validate_mentions(state.mentions, state.text)
        if issues:
            logger.warning(
                "Country agent produced %s invalid mention(s)",
                len(issues),
            )
            return {
                "mentions": valid,
                "issues": issues,
                "retries_left": state.retries_left - 1,
            }
        return {"mentions": valid, "issues": []}

    async def map_to_iso3(state: AgentState) -> dict[str, Any]:
        # Drop any still-invalid mentions after retries are exhausted.
        valid, _issues = validate_mentions(state.mentions, state.text)
        iso3s, detections = mentions_to_iso3(valid)
        return {
            "mentions": valid,
            "issues": [],
            "countries_iso3": iso3s,
            "detections": detections,
        }

    def route_after_validate(
        state: AgentState,
    ) -> Literal["detect", "map_to_iso3"]:
        if state.issues and state.retries_left > 0:
            return "detect"
        return "map_to_iso3"

    graph = StateGraph(AgentState)
    graph.add_node("detect", detect)
    graph.add_node("validate", validate)
    graph.add_node("map_to_iso3", map_to_iso3)
    graph.set_entry_point("detect")
    graph.add_edge("detect", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"detect": "detect", "map_to_iso3": "map_to_iso3"},
    )
    graph.add_edge("map_to_iso3", END)
    return graph.compile()


async def detect_countries(
    text: str,
    *,
    max_retries: int,
    model: BaseChatModel | None = None,
) -> tuple[list[str], list[str]]:
    """Detect countries in ``text`` and return (iso3 codes, substrings)."""
    if not text.strip():
        return [], []

    chat_model = model or build_chat_model()
    agent = build_country_detect_agent(chat_model)
    final_state = AgentState.model_validate(
        await agent.ainvoke(
            {
                "text": text,
                "mentions": [],
                "issues": [],
                "retries_left": max_retries,
                "countries_iso3": [],
                "detections": [],
            }
        )
    )
    return final_state.countries_iso3, final_state.detections
