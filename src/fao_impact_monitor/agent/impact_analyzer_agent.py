"""Impact Analyzer Agent: compile cited country El Nino risk outlooks."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from collections.abc import Sequence
from typing import Any, Literal

import pycountry
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from fao_impact_monitor.config import (
    AwsBedrockConfig,
    ImpactAnalyzerConfig,
    get_config,
)
from fao_impact_monitor.impact_report import (
    ParsedEvidence,
    ParsedMetricReport,
    all_evidence,
    construct_reference_line,
    ensure_citations_present,
    load_plot_image_bytes,
    render_impact_markdown,
)
from fao_impact_monitor.utils.country import iso3_to_country_name

logger = logging.getLogger(__name__)

# Draft/repair embed these; render converts them to numbered [n] citations.
_INLINE_CITE_RE = re.compile(r"\[@([^\s\[\]]+)\]")

_RISK_METRIC_TERMS = (
    "agriculture share of gdp",
    "agriculture share of labour",
    "agriculture share of labor",
    "irrigated cropland",
    "land area equipped for irrigation",
    "nv.agr.totl.zs",
    "sl.agr.empl.zs",
)

ImpactSection = Literal[
    "past_impacts",
    "expected_impacts",
]

DRAFT_SYSTEM = """\
You are an FAO disaster response officer preparing a country El Nino Risk
Outlook for UN officials who must prioritize limited remediation funds. Your
task is to assess how substantial past El Nino impacts were and what they imply
for the current event. Be neutral, objective, and proportionate: neither
amplify impacts to attract funding nor minimize them. Use ONLY the supplied
evidence items (source text and World Bank / FAOSTAT plots). Never fill gaps
from general knowledge, internal memory, or training data.

Reason from the evidence before drafting:
1. Scrutinize ALL supplied evidence. Separate historical El Nino outcomes from
   current environmental and social conditions, and distinguish directly
   attributed impacts from contextual indicators. Check dates, geographic
   scope, quantities, uncertainty, contradictions, and evidence gaps.
2. Silently map each supported finding to one role: hazard and mechanism,
   structural exposure, observed agricultural damage, food-security or
   nutrition consequence, market or livelihood consequence, current condition,
   or compounding risk. Do not output this evidence map or your working notes.
3. Form a historical conclusion only after considering the full evidence set.
   Determine the primary impact, the distinct environmental or social systems
   damaged (the impact vectors), and the severity supported by the evidence.
   Judge severity from evidenced scale, intensity, geographic extent,
   duration or recurrence, livelihood importance, affected population, and
   cascading consequences. Do not claim dimensions the evidence does not
   establish. If the evidence cannot support a confident severity judgement,
   say so explicitly and give the strongest narrower conclusion it supports.
4. Use the historical impact vectors as the analytical spine of the report.
   Assess how current environmental and social conditions differ from the
   historical episodes, then project whether each evidenced impact mechanism
   is likely to be more severe, less severe, similar, or uncertain in the
   current event. Every conclusion must be traceable to cited evidence.

Write like an FAO country risk brief: analytical, readable, and causal.
Prefer short paragraphs and clear topic sentences. Do NOT write dense
statistic dumps, year-by-year plot walkthroughs, or kitchen-sink citation
lists. Every paragraph should advance an argument about vulnerability, past El
Nino impacts on the agrifood system, or projected impacts given today's
starting point.
Number style: write percentages with the % symbol, never the words "percent"
or "per cent".

Length and evidence-selection budget:
- The rendered report must fit within two pages of analysis plus one page of
  references. Keep Past impacts and Expected impacts together concise, aim for
  650-800 words, and NEVER exceed 800 words. References are separate from this
  word budget.
- Consider ALL supplied evidence when forming conclusions, but do NOT report
  all evidence. Include only the smallest, strongest set of facts needed to
  establish the conclusion in each paragraph.
- Prioritize evidence that: determines severity; quantifies the largest or
  most representative impact; establishes the primary causal mechanism;
  shows a material geographic difference; distinguishes current conditions
  from past events; or directly supports the projected impact.
- Omit redundant corroboration, secondary indicators, generic context,
  tangential impacts, long enumerations, and multiple statistics that prove
  the same point. One strong quantity is better than several similar ones.
- Normally use one strongest evidence item for a factual conclusion. Add
  another only when it contributes a necessary quantity, geography,
  mechanism, comparison, or material qualification.
- Use no more than 12 unique evidence_ids across the entire report so the
  reference list fits on one page. If the evidence budget is tight, keep the
  evidence that best supports the required vulnerability assessment, historical
  impact patterns, and future trajectories; never drop support for a retained
  claim.

Structure (encode this shape; do not invent unsupported subsections):
1. Title is supplied separately. Body sections:
   - Past impacts
   - Expected impacts

   NON-NEGOTIABLE COMPLETENESS RULE: the word and evidence budgets must NEVER
   cause you to omit the chapeau, any of the three required Past impacts
   paragraphs, or any of the three required Expected impacts paragraphs.
   Remove secondary evidence and shorten supporting detail first.

2. Past impacts MUST open with ONE chapeau paragraph
   (section="past_impacts", subsection_title=null). It must give the reader the
   overall vulnerability assessment and the historical impact trajectory before
   the detailed subsections.
   Required narrative order:
   (a) The FIRST one or two sentences MUST assess the overall vulnerability of
       the country's food and agriculture system. Base that assessment on all
       three of the following: agriculture's contribution to GDP, agriculture's
       share of employment, and dependence on rainfall as indicated by cropland
       irrigation coverage. Use the latest available value for each indicator,
       with its year and citation, before discussing any historical impact.
   (b) Use irrigation coverage as the evidence-based indicator of rainfall
       dependence. Limited irrigation supports a conclusion of high rainfed
       exposure, but do not claim an exact rainfed share unless the evidence
       supplies one.
   (c) Based on ALL historical evidence, conclude whether El Nino has produced
       a recurring pattern of rainfall deficits, flooding, or geographically or
       seasonally mixed rainfall impacts. Do not describe one episode as the
       national pattern unless the wider evidence supports that conclusion.
   (d) Conclude how that hazard pattern has propagated through the supported
       impact directions: crop yield, livestock and agricultural production;
       food security, nutrition and household wellbeing; and rural livelihoods,
       income, food markets or supply. State the overall trajectory, not the
       detailed evidence, which belongs in the titled paragraphs below.
   Hard rules for the chapeau:
   - Keep it to one compact paragraph of around five short sentences. Its exact
     length should follow the number and complexity of supported impact
     directions, not a rigid sentence count. Reserve roughly 90-120 words.
   - The three vulnerability indicators are the only detailed quantitative
     evidence normally reported here. Put event-specific losses, caseloads,
     prices, income changes, dates, and named locations in the corresponding
     subsections below.
   - Do not list facts without interpreting them. The paragraph must read as
     one chain of reasoning: structural vulnerability -> El Nino rainfall or
     flooding pattern -> trajectory across the supported impact directions.
   - Do NOT narrate multi-year plot series.
   - Distinguish observed damage from structural exposure. Economic or
     livelihood dependence can explain why an impact matters, but it is not
     itself proof that severe damage occurred.
   - Cite only the minimum evidence needed to make the synthesis traceable.
     Do not collect all underlying citations here; cite and explain the full
     supporting evidence in the corresponding impact-vector paragraphs.
   - The chapeau DraftStatement MUST list its cited agriculture-GDP,
     agricultural-employment, irrigation, and conclusion-supporting evidence
     IDs in supporting_evidence_ids. Never return the chapeau with an empty
     supporting_evidence_ids list; the pipeline discards uncited statements.

3. After the chapeau, ALWAYS include these three Past impacts subsections with
   these exact titles and in this order:
   - Agriculture and food production
   - Food security, nutrition and household wellbeing
   - Markets and rural livelihoods
   All three are mandatory in every report and must remain separate.

   REQUIRED Past impacts output sequence:
   1. DraftStatement(section="past_impacts", subsection_title=null): the
      chapeau only.
   2. DraftStatement(section="past_impacts",
      subsection_title="Agriculture and food production"): the complete,
      evidence-rich agriculture and food-production impact paragraph.
   3. DraftStatement(section="past_impacts",
      subsection_title="Food security, nutrition and household wellbeing"):
      the complete, evidence-rich human-consequences paragraph.
   4. DraftStatement(section="past_impacts",
      subsection_title="Markets and rural livelihoods"): the complete,
      evidence-rich market, income, supply and livelihood impact paragraph.

   The untitled chapeau NEVER counts as a mandatory subsection. Projected-impact
   paragraphs also do NOT satisfy these Past impacts requirements. Past impacts
   is invalid if any of these four DraftStatements is absent or out of order.

   Heading and paragraph rules:
   - Use all three mandatory titles verbatim; do not rename, merge, omit, or add
     other Past impacts headings.
   - Produce ONE DraftStatement and therefore one cohesive paragraph under
     each subsection title. Do not fragment one impact vector across multiple
     statements with the same title.
   - Before writing each subsection, examine all relevant past-event evidence
     together and infer the recurring, dominant, mixed or uncertain impact
     pattern for that system.
   - The FIRST sentence of EACH subsection MUST state that cross-event impact
     pattern as an analytical conclusion. It must not begin by naming a year,
     event, location or statistic. Follow the conclusion with the strongest
     supporting quantitative and geographic data. Never leave the reader to
     infer the pattern from a list of facts.
   - Build each paragraph as: impact-pattern conclusion -> El Nino hazard and
     mechanism -> scale and geographic evidence -> consequence for the system.
   - Put the report's evidence-based detail here: use relevant quantitative
     values, dates, named locations, affected populations, losses, prices, and
     exposure indicators to substantiate the impact conclusion. Prefer the
     most decision-relevant evidence over a catalogue of every available fact.
     Keep each impact-vector paragraph to roughly 80-110 words.
   - In "Agriculture and food production", cover crops AND livestock when the
     evidence supports them. Assess whether the pattern is crop loss, livestock
     loss, or both, and how rainfall deficits or flooding produce that damage.
     Establish scale through the strongest available yield loss, production
     loss, cultivated or affected land area, crop loss, livestock mortality or
     morbidity, pasture, and water evidence.
   - In "Food security, nutrition and household wellbeing", use relevant
     food-security classifications or caseloads, prevalence or share of food-
     insecure households or people, nutrition outcomes, food consumption, and
     household coping strategies. Prioritize the percentage or share of food-
     insecure households when supplied, alongside absolute caseloads where
     decision-relevant. Keep food prices and income losses for "Markets and
     rural livelihoods" unless needed for one concise causal link.
   - In "Markets and rural livelihoods", cover food prices, agricultural or
     household income loss, market or cereal-supply shocks, terms of trade,
     distress sales, and consequences for rural livelihoods when evidenced.
     Do not mix this paragraph into food security or livestock production.
   - For every requested dimension, actively search the supplied evidence
     before deciding it is unavailable. If the evidence supports the paragraph's
     overall pattern but not one requested dimension, omit that dimension or
     state the limitation briefly. Never invent a value or claim, and never omit
     the mandatory paragraph solely because one dimension lacks evidence.
   - When multiple historical episodes demonstrate recurrence, compare them
     inside the paragraph for the affected system. Recurrence is evidence
     about a system's vulnerability, not a separate impact vector or heading.
   - Specificity is mandatory when the evidence has it: expand subregion
     detail. Name the exact regions, zones, woredas, districts, markets, or
     livelihood areas, and keep quantitative magnitudes (%, hectares, head
     of livestock, people, prices).
   - BANNED vague geography (never write these when place names exist in the
     cited evidence): "affected locations", "affected areas",
     "affected regions", "affected markets", "some regions", "some areas",
     "some others", "parts of the country", "across large areas",
     "hardest-hit regions" (without naming them), "several locations".
     If the source names places, list those places in the sentence that
     carries the magnitude: put the figure and the named subregions in the
     same clause, never "...in the affected locations".
   - Preserve the exact meaning, units, dates, qualifiers, place names, and
     figures you select from the evidence. Do not invent details. Include all
     material evidence needed to justify the conclusion, but omit redundant
     facts that would turn the paragraph into a catalogue.
   - Do NOT insert unrelated national index chronologies ("separately, the
     crop production index rose...") into an El Nino impact paragraph.

4. Expected impacts (section="expected_impacts") is a FORWARD-LOOKING
   ANALYSIS, not a second summary of Past impacts. Use the main historical
   impact vectors established in Past impacts
   as the projection framework, then apply their underlying El Nino mechanisms
   to the CURRENT climate, environmental, agricultural, market, and household
   conditions supported by the evidence.

   Required projection method for EACH material impact vector:
   1. Historical mechanism: identify the causal pathway demonstrated by past
      events, without retelling the episode or repeating its detailed losses.
   2. Current hazard: establish what the supplied evidence says about the
      current El Nino signal, rainfall or flood outlook, season, timing, and
      geography relevant to that mechanism.
   3. Current agrifood starting point: identify the present system condition
      that will amplify, reduce, or redirect the impact. Depending on the
      vector, this may include crop stage, soil moisture, water or pasture,
      irrigation, current production, food prices, IPC outcomes, household
      stocks, herd sizes, market dependence, displacement, assistance, credit,
      or other evidenced buffers and constraints.
   4. Projection: explicitly infer the most likely future consequence for that
      vector and explain WHY it follows from steps 1-3. State whether the
      current starting point makes the risk more severe, less severe, similar,
      geographically different, or too uncertain to compare with the past.
   5. Confidence and limits: qualify the projection when event intensity,
      geographic coverage, timing, or a quantitative national forecast is not
      established by the evidence.

   Output rules:
   - Expected impacts comes AFTER the complete Past impacts sequence. Its
     instructions never replace, shorten away, or satisfy any required Past
     impacts statement.
   - ALWAYS return exactly these three Expected impacts DraftStatements, with
     these exact subsection titles and in this order:
     1. "Projected losses in agriculture production"
     2. "Existing and projected food insecurity, nutrition and humanitarian vulnerability"
     3. "Compound risks beyond rain deficits or flooding"
   - Use ONE cohesive paragraph of roughly 80-110 words per subsection. Do not
     rename, merge, omit, reorder, or add Expected impacts headings.
   - In "Projected losses in agriculture production", project the trajectory
     for crops, livestock and overall production by applying the historical
     rainfall-deficit or flooding mechanisms to the current climate signal,
     crop stage, water, pasture, irrigation and production conditions.
   - In "Existing and projected food insecurity, nutrition and humanitarian
     vulnerability", first assess the current food-security, nutrition and
     humanitarian starting point, then project how the expected production,
     market or livelihood trajectory is likely to change it. Consider current
     food insecurity, nutrition, displacement, market dependence, household
     stocks, assistance and coping capacity when evidenced. A description of
     existing vulnerability without a future consequence is incomplete.
   - In "Compound risks beyond rain deficits or flooding", assess cascading
     animal-health, livestock-disease, crop-pest or crop-disease risks supported
     by past events, and apply those mechanisms to current conditions. Do not
     invent a compound risk merely to fill the paragraph: if the evidence does
     not establish a specific pathway, state the narrow uncertainty supported
     by the available evidence.
   - Before writing each subsection, combine the relevant historical mechanism,
     current hazard and current system condition to infer the trajectory.
   - The FIRST sentence of EACH subsection MUST summarize that projected
     trajectory. It must not begin with a date, place or statistic. Follow it
     with the strongest current data and only the historical evidence needed to
     support the mechanism. Never begin with a data list or leave the future
     consequence implicit.
   - Every Expected impacts paragraph MUST contain an explicit forward-looking
     conclusion using calibrated language such as "is likely to", "is expected
     to", "would", "could", or "the evidence is insufficient to project".
     A paragraph containing only past or current observations is invalid.
   - Lead with, or reach quickly, the projected outcome. Use current and past
     facts as premises for the projection, not as the paragraph's endpoint.
     End with the consequence or material uncertainty, never with a historical
     statistic.
   - Make the causal bridge explicit: do not merely place a current observation
     beside a historical fact and expect the reader to infer the projection.
   - Historical evidence supports the mechanism and direction of impact, not
     the forecast magnitude. Do NOT copy a past percentage, loss, or caseload
     forward. State a future magnitude only when current forecast evidence
     directly supports it.
   - Do not repeat detailed historical figures already reported in Past
     impacts. Include at most one concise historical comparison when essential
     to explain the mechanism or a material difference in starting conditions.
   - Scale the projection for a stronger or weaker current event only when the
     supplied climate evidence establishes that comparison.
   - Treat current observed damage as a starting condition, not by itself as a
     future projection. Explain how it changes the remaining-season or event-
     horizon risk supported by the evidence.
   - Use the forecast horizon stated in the evidence. If none is supplied,
     refer to "the current El Nino event" or the relevant season; do not invent
     a date or forecast period.
   - Integrate a concise overall judgement of future severity into the final
     Expected impacts paragraph. Identify which vector is likely to dominate
     and why. If current evidence cannot support a projection, say exactly what
     remains uncertain instead of substituting historical outcomes.

5. Do NOT output Preparedness Considerations, preparedness actions, response
   recommendations, or resilience programming. Those are handled separately
   using extractions produced by OER colleagues and are outside this report.

Geographic rules:
- Use only national/subnational facts for the selected country.
- Discard evidence that explicitly attributes facts to another country.
- Broader regions the country belongs to (e.g. East Africa, Southern Africa,
  Horn of Africa, sub-Saharan Africa) may complement the narrative; always
  prefer national/subnational evidence.
- Never reattribute regional/global totals to the selected country.
- When evidence shows uneven or dual impacts inside the country, say so
  explicitly (which areas face drought/rainfall deficits vs flood/excess
  rainfall, and in which seasons). Do not invent a uniform national hazard
  if the evidence is geographically mixed.
- Never omit named geographies or key quantities that are present in the
  cited evidence for that claim. Inventing places or numbers is forbidden;
  omitting evidenced ones is also forbidden.
- Writing "affected locations/areas/regions/markets" instead of the named
  subregions in the cited evidence is a hard failure.

Evidence and citation rules:
- Prefer Direct evidence; Indirect may complement.
- Metric Answer sections are orientation only: they suggest which quantities,
  years, and geographies matter. Do NOT cite Answer. Do NOT treat Answer text
  as evidence. If Answer conflicts with source text/plots, ignore Answer.
- Place an inline citation marker immediately after each sentence (or claim
  clause) that uses a source, using the exact form [@evidence_id]. For example,
  write "Evidence-supported claim. [@evidence_id]"
  If two sentences use the same source, repeat the same [@evidence_id] after
  each sentence. That duplication is required and preferred for human review.
- Do NOT bunch all citations only at the end of a paragraph.
- Also list every cited evidence_id in supporting_evidence_ids (unique union
  for the whole statement). Claims without evidence ids are forbidden.
- Cite only evidence that actually supports that sentence. Do not attach
  every related indicator to a paragraph.
- Never invent numeric [n] markers; use only [@evidence_id]. The system
  converts markers to numbered references.
- For World Bank / FAOSTAT plots: use them to confirm latest levels and
  whether exposure remains high/low. Cite the plot evidence id when you use
  its latest value. Mention a multi-year change ONLY if that specific change
  is needed for the argument (e.g. recovery still incomplete); never recite
  the full series.
- Eligible El Nino periods for historical analysis: 1997-98, 2015-16,
  2018-19, 2023-24. Upcoming/current focus: 2026-27 when evidence supports
  it.

Output structured statements grouped by section. Use subsection_title when a
required subsection heading is appropriate (or null for the chapeau).

Before returning the structured output, perform this compliance check and fix
the draft if any answer is no:
- Are the first four Past impacts DraftStatements the chapeau,
  "Agriculture and food production", "Food security, nutrition and household
  wellbeing", and "Markets and rural livelihoods", in exactly that order?
- Does the chapeau start with the overall vulnerability assessment using the
  latest agriculture GDP, agricultural employment, and irrigation shares, then
  conclude the El Nino hazard pattern and trajectory across all three impact
  directions?
- Does each Past impacts subsection start with a summary of the historical
  impact pattern and then give the strongest supporting data?
- Are the three Expected impacts DraftStatements titled "Projected losses in
  agriculture production", "Existing and projected food insecurity, nutrition
  and humanitarian vulnerability", and "Compound risks beyond rain deficits
  or flooding", in exactly that order?
- Does every Expected impacts paragraph apply a historical mechanism to both
  the current hazard and the current agrifood starting point, then state an
  explicit evidence-bounded future consequence or uncertainty?
- Does each Expected impacts subsection start with a summary of the projected
  trajectory and then give the strongest supporting data?
- Have detailed historical losses been kept in Past impacts rather than
  repeated as if they were projections?
- Have all preparedness and resilience-programming content and statements been
  excluded?
- Is the analysis body within 800 words and supported by no more than 12 unique
  evidence_ids?
"""

VERIFY_SYSTEM = """\
You are an entailment verifier for an impact analysis statement.
Use ONLY the provided statement and cited evidence (source text and/or plots).
Do not use general knowledge or Answer summaries.

Verdicts:
- entailed: the cited evidence jointly supports the statement
- partially_entailed: the core factual content is supported but some wording
  is slightly broader than the evidence
- contradicted: evidence conflicts with the statement
- insufficient: evidence does not support the statement

Reject wrong country/period/unit, invented numbers, unsupported causation,
dropping material uncertainty, or attributing another country's or a
global/aggregate figure to the selected country.
Also reject (as insufficient or partially_entailed with a suggested
revision) statements that vague-out evidenced place names or magnitudes.
Hard reject phrases like "affected locations", "affected areas",
"affected regions", "some regions", or "some areas" when the cited
evidence names specific places. Prefer a revision that lists those
subregions and restores quantitative magnitudes.
Accept transparent comparisons when every input is cited and qualifiers are
preserved.
When section=past_impacts and subsection_title=null, the statement is the
mandatory chapeau. Verify its cited factual premises
and whether its calibrated vulnerability and impact-trajectory conclusions
reasonably follow from them. Do not
reject it merely because it synthesizes several evidence items or expresses an
overall assessment that is not quoted verbatim in one source. If its wording is
too strong, require a narrower conclusion; do not recommend
deleting the mandatory synthesis when an evidence-supported version remains.
For any required titled Past or Expected impacts statement, accept a summary
of the impact pattern or projected trajectory when that analytical conclusion
reasonably follows from the cited premises. If wording is too broad, require a
narrower pattern or trajectory; do not recommend deleting the required
paragraph when an evidence-supported version remains.
For Expected impacts, accept a clearly labelled, cautious analytical projection
when the cited evidence supports all of its premises: the historical mechanism,
the current hazard, and the current environmental or social starting condition.
The exact projected sentence does not need to appear verbatim in a source. Judge
whether the inference reasonably follows from the cited premises without
inventing a magnitude, geography, timing, causal mechanism, or certainty. Reject
an Expected impacts statement that merely lists past/current facts without a
future consequence, or that copies a historical magnitude forward as a forecast.
Inline [@evidence_id] markers identify which evidence supports nearby text;
use them when judging entailment. Unknown or mismatched markers are a defect.
For plot-backed exposure claims, a latest-value statement is sufficient when
that value appears in the latest-value text or chart; do NOT require
year-by-year series narration. If the statement claims a multi-year trend or
change, that change must be visible in the chart or stated in the text.
Exposure language such as "highly exposed because of economic importance /
livelihood dependence / limited irrigation" is acceptable when the cited
scalars support those premises.
"""

REPAIR_SYSTEM = """\
You revise an impact-analysis statement so it is entailed by its cited evidence.
Tighten wording to the evidence; preserve material qualifiers and uncertainty.
Prefer keeping quantitative magnitudes from the evidence.
Prefer the FAO brief style: readable causal prose, latest levels for exposure,
no year-by-year plot recitation unless the change itself is the point.
Use the % symbol for percentages, not the word "percent".
Keep or restore inline [@evidence_id] markers immediately after each sentence
(or claim clause) that uses that source; repeat the same marker when multiple
sentences use the same source. Do not invent numeric [n] markers.
Restore named regions/zones/woredas and quantitative magnitudes from the
cited evidence. Replace any "affected locations/areas/regions/markets" or
"some regions/areas" with the actual place names from the evidence.
If the evidence is regional/global, restore that geographic scope and remove
unsupported attribution to the selected country.
Do not invent facts. Prefer a narrower true statement over remove=true.
When repairing an Expected impacts statement, preserve its forward-looking
function. Rebuild it as: evidenced current hazard + evidenced current agrifood
condition + historically evidenced mechanism -> cautious projected consequence.
Do not "repair" a projection by reducing it to a list of past and current facts.
Remove unsupported precision and qualify the inference instead.
When section=past_impacts and subsection_title=null, this is the mandatory
chapeau. NEVER set remove=true if the cited evidence can support any
narrower synthesis. Preserve: the overall evidence-bounded vulnerability
assessment; the latest agriculture GDP, agricultural employment, and irrigation
shares; the El Nino rainfall-deficit or flooding pattern where supported; and
the concise trajectory across the three impact directions. Remove unsupported
strength or detail instead of removing the paragraph.
For every required titled Past or Expected impacts statement, preserve its
assigned analytical role and opening pattern/trajectory conclusion. Narrow
unsupported claims and retain the strongest cited supporting data rather than
removing the required paragraph whenever an evidence-supported version remains.
Set remove=true only when nothing evidence-supported remains.
"""

COUNTRY_FILTER_SYSTEM = """\
You judge whether an evidence item should be kept for analyzing the selected
country.

Return discard=true when the evidence explicitly attributes its facts to a
different country (not the selected country). Mentions of other countries as
comparators are fine if the measured result is about the selected country.

Return discard=false when the evidence is about the selected country at
national or subnational level, OR about a broader region the country belongs
to (e.g. East Africa, Horn of Africa, Southern Africa, sub-Saharan Africa),
OR is a World Bank / FAOSTAT indicator for the selected country.

Do not discard merely because a regional document also names neighbours.
"""


class DraftStatement(BaseModel):
    section: ImpactSection
    subsection_title: str | None = None
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class DraftStatementList(BaseModel):
    statements: list[DraftStatement] = Field(default_factory=list)


class StatementVerification(BaseModel):
    statement_id: str
    verdict: Literal[
        "entailed",
        "partially_entailed",
        "contradicted",
        "insufficient",
    ]
    unsupported_parts: list[str] = Field(default_factory=list)
    reasoning: str
    suggested_revision: str | None = None


class StatementRepair(BaseModel):
    statement_id: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    remove: bool = False


class CountryFilterVerdict(BaseModel):
    evidence_id: str
    discard: bool
    reason: str


class CountryFilterList(BaseModel):
    verdicts: list[CountryFilterVerdict] = Field(default_factory=list)


class VerifiedStatement(BaseModel):
    statement_id: str
    section: ImpactSection
    subsection_title: str | None = None
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class ImpactAnalyzerOutput(BaseModel):
    country: str
    country_iso3: str
    markdown: str
    statements: list[VerifiedStatement] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    discarded_evidence_ids: list[str] = Field(default_factory=list)


def build_chat_model(
    *,
    llm_model: str | None = None,
    reasoning_effort: str | None = None,
    aws_bedrock_config: AwsBedrockConfig | None = None,
    config: ImpactAnalyzerConfig | None = None,
) -> BaseChatModel:
    cfg = config or get_config().impact_analyzer
    aws = aws_bedrock_config or get_config().aws_bedrock
    kwargs: dict[str, Any] = {
        "api_key": aws.api_key.get_secret_value(),
        "base_url": aws.base_url,
        "use_responses_api": True,
        "reasoning_effort": reasoning_effort or cfg.reasoning_effort,
    }
    chat_model = init_chat_model(llm_model or cfg.llm_model, **kwargs)
    if not isinstance(chat_model, BaseChatModel):
        raise TypeError(f"Expected BaseChatModel, got {type(chat_model)}")
    return chat_model


async def analyze_impact(
    *,
    country_iso3: str,
    reports: Sequence[ParsedMetricReport],
    config: ImpactAnalyzerConfig | None = None,
    model: BaseChatModel | None = None,
    verifier_model: BaseChatModel | None = None,
    filter_model: BaseChatModel | None = None,
) -> ImpactAnalyzerOutput:
    """Draft, verify, and render a cited impact analysis for one country."""
    cfg = config or get_config().impact_analyzer
    country_name = iso3_to_country_name(country_iso3)
    iso3 = country_iso3.upper()
    main_model = model or build_chat_model(config=cfg)
    verify_model = verifier_model or build_chat_model(
        llm_model=cfg.verifier_llm_model,
        reasoning_effort="low",
        config=cfg,
    )
    country_filter_model = filter_model or build_chat_model(
        llm_model=cfg.filter_llm_model,
        reasoning_effort="low",
        config=cfg,
    )

    started = time.perf_counter()
    evidence_items = all_evidence(list(reports))
    for item in evidence_items:
        if (
            item.source_type in {"worldbank", "faostat", "structured"}
            and item.plot_path
        ):
            # Fail fast if a linked plot is missing.
            load_plot_image_bytes(item)

    stage = time.perf_counter()
    kept, discarded_ids = await _filter_country_evidence(
        evidence_items,
        country_name=country_name,
        country_iso3=iso3,
        model=country_filter_model,
        config=cfg,
    )
    logger.info(
        "ImpactAnalyzer STAGE=filter country=%s evidence=%s kept=%s discarded=%s "
        "elapsed=%.1fs",
        iso3,
        len(evidence_items),
        len(kept),
        len(discarded_ids),
        time.perf_counter() - stage,
    )

    stage = time.perf_counter()
    drafted = await _draft_statements(
        reports=list(reports),
        evidence=kept,
        country_name=country_name,
        country_iso3=iso3,
        model=main_model,
        config=cfg,
    )
    logger.info(
        "ImpactAnalyzer STAGE=draft country=%s statements=%s elapsed=%.1fs",
        iso3,
        len(drafted),
        time.perf_counter() - stage,
    )

    stage = time.perf_counter()
    verified = await _verify_and_repair(
        drafted,
        evidence=kept,
        country_name=country_name,
        country_iso3=iso3,
        model=main_model,
        verifier_model=verify_model,
        config=cfg,
    )
    logger.info(
        "ImpactAnalyzer STAGE=verify country=%s verified=%s drafted=%s elapsed=%.1fs",
        iso3,
        len(verified),
        len(drafted),
        time.perf_counter() - stage,
    )

    markdown, references = _render_output(
        country_name=country_name,
        country_iso3=iso3,
        statements=verified,
        evidence=kept,
    )
    logger.info(
        "ImpactAnalyzer END country=%s total_elapsed=%.1fs refs=%s",
        iso3,
        time.perf_counter() - started,
        len(references),
    )
    return ImpactAnalyzerOutput(
        country=country_name,
        country_iso3=iso3,
        markdown=markdown,
        statements=verified,
        references=references,
        discarded_evidence_ids=discarded_ids,
    )


async def _structured_invoke(
    model: BaseChatModel,
    schema: type[BaseModel],
    *,
    system: str,
    user: Any,
    method: Literal["function_calling", "json_mode", "json_schema"] | None = None,
) -> BaseModel:
    structured = (
        model.with_structured_output(schema, method=method)
        if method is not None
        else model.with_structured_output(schema)
    )
    result = await structured.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def _evidence_by_id(evidence: Sequence[ParsedEvidence]) -> dict[str, ParsedEvidence]:
    return {item.evidence_id: item for item in evidence}


def _inline_citation_ids(text: str) -> list[str]:
    return _INLINE_CITE_RE.findall(text)


def _normalize_inline_citations(
    text: str,
    supporting_evidence_ids: Sequence[str],
    known: dict[str, ParsedEvidence],
) -> tuple[str, list[str]]:
    """Keep known [@evidence_id] markers; sync supporting ids from them."""
    inline_ids = _inline_citation_ids(text)
    if inline_ids:

        def _keep(match: re.Match[str]) -> str:
            evidence_id = match.group(1)
            return f"[@{evidence_id}]" if evidence_id in known else ""

        cleaned = _INLINE_CITE_RE.sub(_keep, text)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([.,;:])", r"\1", cleaned).strip()
        supported = list(
            dict.fromkeys(
                evidence_id for evidence_id in inline_ids if evidence_id in known
            )
        )
        return cleaned, supported
    supported = [
        evidence_id
        for evidence_id in dict.fromkeys(supporting_evidence_ids)
        if evidence_id in known
    ]
    return text.strip(), supported


def _render_statement_text(
    text: str,
    supporting_evidence_ids: Sequence[str],
    number_by_id: dict[str, int],
) -> str | None:
    """Convert [@evidence_id] markers to [n], or append end-of-paragraph cites."""
    if _INLINE_CITE_RE.search(text):

        def _replace(match: re.Match[str]) -> str:
            evidence_id = match.group(1)
            number = number_by_id.get(evidence_id)
            return f"[{number}]" if number is not None else ""

        rendered = _INLINE_CITE_RE.sub(_replace, text)
        rendered = re.sub(r"[ \t]{2,}", " ", rendered)
        rendered = re.sub(r" +([.,;:])", r"\1", rendered).strip()
        if not re.search(r"\[\d+\]", rendered):
            return None
        return rendered
    cites = " ".join(
        f"[{number_by_id[evidence_id]}]"
        for evidence_id in dict.fromkeys(supporting_evidence_ids)
        if evidence_id in number_by_id
    )
    if not cites:
        return None
    return f"{text} {cites}".rstrip()


def _is_risk_plot(evidence: ParsedEvidence) -> bool:
    haystack = " ".join(
        part
        for part in (evidence.metric_name, evidence.title, evidence.indicator or "")
        if part
    ).casefold()
    return any(term in haystack for term in _RISK_METRIC_TERMS)


def _plot_image_block(
    evidence: ParsedEvidence,
    *,
    detail: str = "low",
) -> dict[str, Any]:
    data = load_plot_image_bytes(evidence)
    encoded = base64.b64encode(data).decode("ascii")
    suffix = evidence.plot_path.suffix.lower() if evidence.plot_path else ".png"
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media};base64,{encoded}",
            "detail": detail,
        },
    }


def _format_evidence_for_prompt(
    item: ParsedEvidence,
    *,
    max_source_text_chars: int = 2_500,
) -> str:
    lines = [
        f"evidence_id: {item.evidence_id}",
        f"kind: {item.kind}",
        f"source_type: {item.source_type}",
        f"metric: {item.metric_name}",
        f"title: {item.title}",
    ]
    if item.indicator:
        lines.append(f"indicator: {item.indicator}")
    if item.physical_pages:
        pages = ", ".join(str(page) for page in item.physical_pages)
        lines.append(f"physical_pages: {pages}")
    if item.source_url:
        lines.append(f"source_url: {item.source_url}")
    if item.latest_value is not None:
        year = f" ({item.latest_year})" if item.latest_year is not None else ""
        lines.append(f"latest_value: {item.latest_value}{year}")
    if item.plot_path is not None:
        lines.append(f"plot: attached image for {item.plot_path.name}")
    if item.source_text:
        text = item.source_text
        if len(text) > max_source_text_chars:
            text = text[:max_source_text_chars].rstrip() + "\n...[truncated]"
        lines.append(f"source_text:\n{text}")
    if item.verified_visual_facts:
        facts = "\n".join(f"- {fact}" for fact in item.verified_visual_facts)
        lines.append(f"verified_visual_facts:\n{facts}")
    return "\n".join(lines)


def _country_name_variants(country_iso3: str, country_name: str) -> list[str]:
    variants = [country_name, country_iso3]
    country = pycountry.countries.get(alpha_3=country_iso3.upper())
    if country is not None:
        variants.append(str(country.name))
        official = getattr(country, "official_name", None)
        if isinstance(official, str) and official.strip():
            variants.append(official)
        common = getattr(country, "common_name", None)
        if isinstance(common, str) and common.strip():
            variants.append(common)
    # Longer names first so "Federal Democratic Republic of Ethiopia" wins.
    deduped: list[str] = []
    seen: set[str] = set()
    for name in sorted(
        {item.strip() for item in variants if item.strip()},
        key=len,
        reverse=True,
    ):
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _mentions_country(text: str, variants: Sequence[str]) -> bool:
    folded = text.casefold()
    return any(variant.casefold() in folded for variant in variants)


def _heuristic_country_decision(
    item: ParsedEvidence,
    *,
    selected_variants: Sequence[str],
) -> Literal["keep", "discard", "llm"] | None:
    """Fast path before LLM country filtering."""
    if item.source_type in {"worldbank", "faostat", "structured"}:
        return "keep"
    blob = " ".join(
        part
        for part in (item.title, item.source_text or "", item.source_url or "")
        if part
    )
    if not blob.strip():
        return "keep"
    if _mentions_country(blob, selected_variants):
        return "keep"
    # Title/source clearly about another country and silent on the selected one.
    other_iso3 = _explicit_other_country_iso3(blob, selected_variants)
    if other_iso3 is not None:
        return "discard"
    return "llm"


_COUNTRY_TOKEN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def _explicit_other_country_iso3(
    text: str,
    selected_variants: Sequence[str],
) -> str | None:
    """Return another country's ISO3 when the text looks country-specific to it."""
    selected = {variant.casefold() for variant in selected_variants}
    # Cheap scan: look up capitalized multi-word tokens via pycountry fuzzy match.
    candidates = _COUNTRY_TOKEN.findall(text)
    for candidate in candidates:
        if candidate.casefold() in selected:
            continue
        if len(candidate) < 4:
            continue
        try:
            matches = pycountry.countries.search_fuzzy(candidate)
        except LookupError:
            continue
        if not matches:
            continue
        match = matches[0]
        name = str(match.name).casefold()
        if name in selected:
            continue
        # Require the matched official/common name to appear, not a weak fuzzy hit.
        if name not in text.casefold() and str(match.name) not in text:
            common = getattr(match, "common_name", None)
            if not (isinstance(common, str) and common.casefold() in text.casefold()):
                continue
        return str(match.alpha_3)
    return None


def _answers_block(reports: Sequence[ParsedMetricReport]) -> str:
    parts: list[str] = []
    for report in reports:
        if not report.answer:
            continue
        parts.append(
            f"### Metric {report.meta.seq_number}: {report.meta.name}\n{report.answer}"
        )
    if not parts:
        return "(no Answer sections)"
    return "\n\n".join(parts)


async def _filter_country_evidence(
    evidence: Sequence[ParsedEvidence],
    *,
    country_name: str,
    country_iso3: str,
    model: BaseChatModel,
    config: ImpactAnalyzerConfig,
) -> tuple[list[ParsedEvidence], list[str]]:
    """Keep national/subnational/regional evidence; discard other-country items."""
    if not evidence:
        return [], []

    selected_variants = _country_name_variants(country_iso3, country_name)
    kept: list[ParsedEvidence] = []
    discarded: list[str] = []
    to_judge: list[ParsedEvidence] = []
    for item in evidence:
        decision = _heuristic_country_decision(
            item, selected_variants=selected_variants
        )
        if decision == "keep":
            kept.append(item)
        elif decision == "discard":
            discarded.append(item.evidence_id)
            logger.info(
                "Discarding evidence %s via heuristic (other-country attribution)",
                item.evidence_id,
            )
        else:
            to_judge.append(item)

    logger.info(
        "ImpactAnalyzer filter heuristic kept=%s discarded=%s llm_pending=%s",
        len(kept),
        len(discarded),
        len(to_judge),
    )
    if not to_judge:
        return kept, discarded

    batch_size = max(1, config.country_filter_batch_size)
    batches = [
        to_judge[index : index + batch_size]
        for index in range(0, len(to_judge), batch_size)
    ]

    async def _judge_batch(
        batch: list[ParsedEvidence],
    ) -> tuple[list[ParsedEvidence], list[str]]:
        block = "\n\n---\n\n".join(
            f"evidence_id: {item.evidence_id}\n"
            f"title: {item.title}\n"
            f"source_text:\n{(item.source_text or '')[: config.max_source_text_chars]}"
            for item in batch
        )
        user = (
            f"Selected country: {country_name} ({country_iso3})\n\n"
            f"Evidence items:\n{block}"
        )
        try:
            result = await _structured_invoke(
                model,
                CountryFilterList,
                system=COUNTRY_FILTER_SYSTEM,
                user=user,
            )
            assert isinstance(result, CountryFilterList)
            by_id = {verdict.evidence_id: verdict for verdict in result.verdicts}
        except Exception:
            logger.exception("Country filter failed; keeping batch conservatively")
            return list(batch), []
        batch_kept: list[ParsedEvidence] = []
        batch_discarded: list[str] = []
        for item in batch:
            verdict = by_id.get(item.evidence_id)
            if verdict is not None and verdict.discard:
                batch_discarded.append(item.evidence_id)
                logger.info(
                    "Discarding evidence %s: %s",
                    item.evidence_id,
                    verdict.reason,
                )
            else:
                batch_kept.append(item)
        return batch_kept, batch_discarded

    results = await asyncio.gather(*[_judge_batch(batch) for batch in batches])
    for batch_kept, batch_discarded in results:
        kept.extend(batch_kept)
        discarded.extend(batch_discarded)
    return kept, discarded


async def _draft_statements(
    *,
    reports: Sequence[ParsedMetricReport],
    evidence: Sequence[ParsedEvidence],
    country_name: str,
    country_iso3: str,
    model: BaseChatModel,
    config: ImpactAnalyzerConfig,
) -> list[VerifiedStatement]:
    # Prefer Direct + risk plots first so the model sees them even if truncated.
    ordered = sorted(
        evidence,
        key=lambda item: (
            0 if item.kind == "direct" else 1,
            0 if _is_risk_plot(item) else 1,
            item.metric_seq,
            item.evidence_id,
        ),
    )
    text_parts = [
        f"Country: {country_name} ({country_iso3})",
        "",
        (
            "Hard length budget: keep the entire analysis body at 650-800 "
            "words and use at most 12 unique evidence_ids; select only the "
            "strongest evidence needed to prove each conclusion. Never meet "
            "that budget by omitting a required paragraph. Start Past impacts "
            "with a compact chapeau assessing vulnerability through the latest "
            "agriculture GDP, employment and irrigation shares, then conclude "
            "the El Nino rainfall/flooding pattern and trajectory across the "
            "three impact directions. The first four Past impacts statements "
            "MUST have subsection_title values null, 'Agriculture and food "
            "production', 'Food security, nutrition and household wellbeing', "
            "and 'Markets and rural livelihoods' in exactly that order. Each "
            "titled paragraph must start with the historical impact-pattern "
            "conclusion and then give key supporting data. The three Expected "
            "impacts statements MUST be titled 'Projected losses in agriculture "
            "production', 'Existing and projected food insecurity, nutrition "
            "and humanitarian vulnerability', and 'Compound risks beyond rain "
            "deficits or "
            "flooding' in that order. Each must start with the projected "
            "trajectory and apply a historical mechanism to the current hazard "
            "and agrifood starting point. A list of past/current facts is "
            "invalid, and past magnitudes must not be reused as forecasts. Do "
            "not output preparedness or resilience programming. NEVER write "
            "'affected "
            "locations/areas/regions' or 'some regions/areas' when place "
            "names are available. Write percentages with the % symbol. Place "
            "[@evidence_id] immediately after each "
            "sentence that uses that source (repeat when reused)."
        ),
        "",
        "Metric Answer sections (direction ONLY; not citable):",
        _answers_block(reports),
        "",
        "Evidence items (cite by evidence_id):",
    ]
    image_blocks: list[dict[str, Any]] = []
    for item in ordered:
        text_parts.append(
            _format_evidence_for_prompt(
                item, max_source_text_chars=config.max_source_text_chars
            )
        )
        text_parts.append("---")
        if item.plot_path is not None:
            detail = (
                config.risk_plot_detail
                if _is_risk_plot(item)
                else config.default_plot_detail
            )
            text_parts.append(
                f"[IMAGE FOLLOWS for evidence_id={item.evidence_id} "
                f"metric={item.metric_name} indicator={item.indicator} "
                f"detail={detail}]"
            )
            image_blocks.append(_plot_image_block(item, detail=detail))

    logger.info(
        "ImpactAnalyzer draft payload evidence=%s plots=%s",
        len(ordered),
        len(image_blocks),
    )
    user_content: Any
    if image_blocks:
        user_content = [
            {"type": "text", "text": "\n".join(text_parts)},
            *image_blocks,
        ]
    else:
        user_content = "\n".join(text_parts)

    result = await _structured_invoke(
        model,
        DraftStatementList,
        system=DRAFT_SYSTEM,
        user=user_content,
    )
    assert isinstance(result, DraftStatementList)
    known = _evidence_by_id(evidence)
    statements: list[VerifiedStatement] = []
    for index, draft in enumerate(result.statements, start=1):
        text, supported = _normalize_inline_citations(
            draft.text,
            draft.supporting_evidence_ids,
            known,
        )
        if not supported or not text:
            continue
        statements.append(
            VerifiedStatement(
                statement_id=f"stmt_{index:03d}",
                section=draft.section,
                subsection_title=draft.subsection_title,
                text=text,
                supporting_evidence_ids=supported,
            )
        )
    return statements


async def _verify_and_repair(
    statements: Sequence[VerifiedStatement],
    *,
    evidence: Sequence[ParsedEvidence],
    country_name: str,
    country_iso3: str,
    model: BaseChatModel,
    verifier_model: BaseChatModel,
    config: ImpactAnalyzerConfig,
) -> list[VerifiedStatement]:
    known = _evidence_by_id(evidence)
    semaphore = asyncio.Semaphore(max(1, config.verify_concurrency))

    async def _verify_one(statement: VerifiedStatement) -> VerifiedStatement | None:
        async with semaphore:
            current = statement
            for attempt in range(config.max_answer_verification_retries + 1):
                try:
                    verification = await _verify_statement(
                        current,
                        evidence_by_id=known,
                        country_name=country_name,
                        country_iso3=country_iso3,
                        model=verifier_model,
                        config=config,
                    )
                except Exception:
                    logger.exception(
                        "Verification failed for %s; dropping statement",
                        current.statement_id,
                    )
                    return None
                if verification.verdict == "entailed":
                    return current
                if attempt >= config.max_answer_verification_retries:
                    logger.info(
                        "Dropping statement %s after retries verdict=%s",
                        current.statement_id,
                        verification.verdict,
                    )
                    return None
                try:
                    repaired = await _repair_statement(
                        current,
                        verification,
                        evidence_by_id=known,
                        model=model,
                        config=config,
                    )
                except Exception:
                    logger.exception(
                        "Repair failed for %s; dropping statement",
                        current.statement_id,
                    )
                    return None
                if repaired is None:
                    return None
                current = repaired
            return None

    results = await asyncio.gather(
        *[_verify_one(statement) for statement in statements]
    )
    return [item for item in results if item is not None]


async def _verify_statement(
    statement: VerifiedStatement,
    *,
    evidence_by_id: dict[str, ParsedEvidence],
    country_name: str,
    country_iso3: str,
    model: BaseChatModel,
    config: ImpactAnalyzerConfig,
) -> StatementVerification:
    cited = [
        evidence_by_id[evidence_id]
        for evidence_id in statement.supporting_evidence_ids
        if evidence_id in evidence_by_id
    ]
    text_block = (
        "\n\n".join(
            _format_evidence_for_prompt(
                item, max_source_text_chars=config.max_source_text_chars
            )
            for item in cited
        )
        or "(no evidence)"
    )
    user_text = (
        f"Country: {country_name} ({country_iso3})\n"
        f"Section: {statement.section}\n"
        f"Subsection title: {statement.subsection_title!r}\n"
        f"Statement ({statement.statement_id}): {statement.text}\n\n"
        f"Cited evidence:\n{text_block}"
    )
    image_blocks = [
        _plot_image_block(item, detail=config.default_plot_detail)
        for item in cited
        if item.plot_path is not None
    ]
    user: Any = (
        [{"type": "text", "text": user_text}, *image_blocks]
        if image_blocks
        else user_text
    )
    result = await _structured_invoke(
        model,
        StatementVerification,
        system=VERIFY_SYSTEM,
        user=user,
    )
    assert isinstance(result, StatementVerification)
    return StatementVerification(
        statement_id=statement.statement_id,
        verdict=result.verdict,
        unsupported_parts=list(result.unsupported_parts),
        reasoning=result.reasoning,
        suggested_revision=result.suggested_revision,
    )


async def _repair_statement(
    statement: VerifiedStatement,
    verification: StatementVerification,
    *,
    evidence_by_id: dict[str, ParsedEvidence],
    model: BaseChatModel,
    config: ImpactAnalyzerConfig,
) -> VerifiedStatement | None:
    cited = [
        evidence_by_id[evidence_id]
        for evidence_id in statement.supporting_evidence_ids
        if evidence_id in evidence_by_id
    ]
    claims_block = "\n\n".join(
        _format_evidence_for_prompt(
            item, max_source_text_chars=config.max_source_text_chars
        )
        for item in cited
    )
    user = (
        f"Section: {statement.section}\n"
        f"Subsection title: {statement.subsection_title!r}\n"
        f"Statement: {statement.text}\n"
        f"Verification verdict: {verification.verdict}\n"
        f"Unsupported parts: {verification.unsupported_parts}\n"
        f"Suggested revision: {verification.suggested_revision}\n"
        f"Reasoning: {verification.reasoning}\n\n"
        f"Cited evidence:\n{claims_block}"
    )
    result = await _structured_invoke(
        model,
        StatementRepair,
        system=REPAIR_SYSTEM,
        user=user,
    )
    assert isinstance(result, StatementRepair)
    if result.remove or not result.text.strip():
        return None
    text, supported = _normalize_inline_citations(
        result.text,
        result.supporting_evidence_ids or statement.supporting_evidence_ids,
        evidence_by_id,
    )
    if not supported or not text:
        return None
    return VerifiedStatement(
        statement_id=statement.statement_id,
        section=statement.section,
        subsection_title=statement.subsection_title,
        text=text,
        supporting_evidence_ids=supported,
    )


def _render_output(
    *,
    country_name: str,
    country_iso3: str,
    statements: Sequence[VerifiedStatement],
    evidence: Sequence[ParsedEvidence],
) -> tuple[str, list[str]]:
    known = _evidence_by_id(evidence)
    used_ids: list[str] = []
    for statement in statements:
        inline_ids = _inline_citation_ids(statement.text)
        ordered_ids = inline_ids or list(statement.supporting_evidence_ids)
        for evidence_id in ordered_ids:
            if evidence_id in known and evidence_id not in used_ids:
                used_ids.append(evidence_id)

    number_by_id = {evidence_id: index for index, evidence_id in enumerate(used_ids, 1)}
    references = [
        construct_reference_line(known[evidence_id], country_iso3=country_iso3)
        for evidence_id in used_ids
    ]

    section_map: dict[str, list[str]] = {
        "Past impacts": [],
        "Expected impacts": [],
    }
    section_keys = {
        "past_impacts": "Past impacts",
        "expected_impacts": "Expected impacts",
    }
    last_subsection: dict[str, str | None] = {key: None for key in section_map}
    for statement in statements:
        heading = section_keys[statement.section]
        text = _render_statement_text(
            statement.text,
            statement.supporting_evidence_ids,
            number_by_id,
        )
        if text is None:
            continue
        if statement.subsection_title:
            title = statement.subsection_title.strip()
            if last_subsection[heading] != title:
                section_map[heading].append(f"**{title}**: {text}")
                last_subsection[heading] = title
            else:
                section_map[heading].append(text)
        else:
            section_map[heading].append(text)
            last_subsection[heading] = None

    sections = {
        heading: "\n\n".join(paragraphs) for heading, paragraphs in section_map.items()
    }
    markdown = render_impact_markdown(
        country_name=country_name,
        sections=sections,
        references=references,
    )
    markdown = ensure_citations_present(markdown)
    return markdown, references
