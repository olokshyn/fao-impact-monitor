"""LangGraph agent that extracts crawl candidate URLs from an HTML page."""

import logging
from typing import Any, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from fao_impact_monitor.config import AwsBedrockConfig, PdfCrawlConfig, get_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a link-extraction agent for the FAO Impact Monitor data lake.

Your job is to read an HTML page and select the most important hyperlinks to
crawl next in search of documentary PDF evidence (reports, publications,
annexes, statistical briefs, knowledge-repository items).

Prioritize links that:
- Point directly at PDFs (href contains .pdf, or captions like "Download PDF",
  "Full report", "Read the publication")
- Likely lead to PDFs or documentary evidence (publications, resources,
  documents, reports, libraries, repositories, annexes, data downloads)
- Are substantive content hubs worth browsing (topic pages, document listings)

Deprioritize or ignore page chrome: login, language switchers, share widgets,
cookie banners, identical site-wide footer/header noise, pure social media.

Critical rules:
1. Copy each URL string EXACTLY as it appears in the page body/source (href or
   visible URL text). Do not invent, normalize, or absolutize URLs.
2. Return at most the requested maximum number of URLs.
3. Prefer evidence-bearing links; if more candidates exist than the cap, keep
   the most important ones.
4. For each URL provide a short reason citing the link text or nearby context.
"""


class LinkCandidate(BaseModel):
    reason: str = Field(
        default="",
        description="Short rationale based on anchor text or nearby context",
    )
    url: str = Field(description="URL copied exactly as it appears in the page body")


class PdfLinkCandidateList(BaseModel):
    links: list[LinkCandidate] = Field(default_factory=list)


class AgentState(TypedDict):
    page_url: str
    page_body: str
    max_urls: int
    missing_urls: list[str]
    retries_left: int
    urls: list[str]


def build_chat_model(
    pdf_crawl: PdfCrawlConfig | None = None,
    aws_bedrock: AwsBedrockConfig | None = None,
) -> BaseChatModel:
    config = get_config()
    pdf_crawl = pdf_crawl or config.pdf_crawl
    aws_bedrock = aws_bedrock or config.aws_bedrock
    return init_chat_model(
        pdf_crawl.llm_model,
        api_key=aws_bedrock.api_key.get_secret_value(),
        base_url=aws_bedrock.base_url,
        use_responses_api=True,
    )


def _user_prompt(
    *,
    page_url: str,
    page_body: str,
    max_urls: int,
    missing_urls: list[str],
) -> str:
    parts = [
        f"Page URL: {page_url}",
        f"Return at most {max_urls} important links.",
        "Page body follows:",
        page_body,
    ]
    if missing_urls:
        missing = "\n".join(f"- {url}" for url in missing_urls)
        parts.insert(
            0,
            "Your previous URL list contained values that do NOT appear as "
            "substrings in the page body. Produce a new list. These were not "
            f"found:\n{missing}\n"
            "Only include URLs that appear verbatim in the page body.",
        )
    return "\n\n".join(parts)


def _validate_urls_in_body(urls: list[str], page_body: str) -> list[str]:
    return [url for url in urls if url not in page_body]


def build_link_agent(model: BaseChatModel) -> Any:
    """Compile a LangGraph that extracts and validates page links."""

    structured = model.with_structured_output(PdfLinkCandidateList)

    async def extract(state: AgentState) -> dict[str, Any]:
        prompt = _user_prompt(
            page_url=state["page_url"],
            page_body=state["page_body"],
            max_urls=state["max_urls"],
            missing_urls=state["missing_urls"],
        )
        result = await structured.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        if not isinstance(result, PdfLinkCandidateList):
            result = PdfLinkCandidateList.model_validate(result)
        urls = [link.url for link in result.links[: state["max_urls"]]]
        logger.info(
            "Link agent extracted %s URL(s) from %s",
            len(urls),
            state["page_url"],
        )
        return {"urls": urls, "missing_urls": []}

    async def validate(state: AgentState) -> dict[str, Any]:
        missing = _validate_urls_in_body(state["urls"], state["page_body"])
        if missing:
            logger.warning(
                "Link agent produced %s URL(s) missing from body for %s",
                len(missing),
                state["page_url"],
            )
            return {
                "missing_urls": missing,
                "retries_left": state["retries_left"] - 1,
            }
        valid = [url for url in state["urls"] if url in state["page_body"]]
        return {"urls": valid[: state["max_urls"]], "missing_urls": []}

    def route_after_validate(
        state: AgentState,
    ) -> Literal["extract", "__end__"]:
        if state["missing_urls"] and state["retries_left"] > 0:
            return "extract"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("extract", extract)
    graph.add_node("validate", validate)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"extract": "extract", "__end__": END},
    )
    return graph.compile()


async def extract_page_urls(
    *,
    page_url: str,
    page_body: str,
    max_urls: int,
    max_retries: int,
    model: BaseChatModel | None = None,
) -> list[str]:
    """Run the link agent and return validated URLs (as they appear in the body)."""
    chat_model = model or build_chat_model()
    agent = build_link_agent(chat_model)
    final_state: AgentState = await agent.ainvoke(
        {
            "page_url": page_url,
            "page_body": page_body,
            "max_urls": max_urls,
            "missing_urls": [],
            "retries_left": max_retries,
            "urls": [],
        }
    )
    urls = final_state.get("urls", [])
    # Drop any still-invalid URLs after retries are exhausted.
    return [url for url in urls if url in page_body][:max_urls]
