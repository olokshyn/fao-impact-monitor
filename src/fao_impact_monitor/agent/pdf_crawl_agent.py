"""LangGraph agent that extracts crawl candidate URLs and document titles."""

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

Your job is to read an HTML page and select only the hyperlinks that advance
a crawl toward documentary PDF evidence. On PDF abstract / document-detail
pages, also extract the document title.

Select ONLY:
1. URLs that download a PDF directly (href contains .pdf, or clear download
   actions such as "Download PDF", "Full report", "Download publication").
2. URLs that lead to a PDF abstract / detail page — a page that shows the
   document title and abstract (or summary) and a PDF download button/link.
3. URLs that lead to publication / document listing hubs that are clearly on
   the path to PDFs (archives, repositories, annex libraries, "publications",
   "resources", "documents", "reports"). Prefer these over generic site pages.
4. URLs that lead to the next page of search results (pagination: "Next",
   page numbers, "Load more" for results). Search-result pages must be
   paginated through so later result pages can be crawled.

Do NOT select:
- Keyword / tag / topic facet links
- Links to parts of a document (chapters, sections, table of contents anchors)
- Author, contributor, or profile pages
- Login, language switchers, share widgets, cookie banners, social media,
  footer/header chrome, unrelated site navigation

Document title (document_title):
- Output a document_title ONLY on a PDF abstract / document-details page —
  the page that shows the document title, abstract/summary, metadata, and a
  PDF download control (e.g. "Download PDF").
- Copy the title EXACTLY as it appears in the page body (the main document
  heading). Do not invent, normalize, translate, or paraphrase.
- On listing hubs, search results, pagination pages, or any page that is not
  a single-document details page, set document_title to null.
- Never use the browser tab title, site name, or breadcrumb alone as the
  document title unless that exact string is the document's main heading.

Quality over quantity: prefer fewer, highly relevant URLs over a long noisy
list. If unsure whether a link advances toward a PDF (download, abstract page,
or listing hub), omit it.

Ordering:
- Return URLs in the same order they appear on the page. Listing/search pages
  usually already rank results by relevance; preserve that order.
- Put pagination ("next page") links after the on-page result/document links.

Critical rules:
1. Copy each URL string EXACTLY as it appears in the page body/source (href or
   visible URL text). Do not invent, normalize, or absolutize URLs.
2. Return at most the requested maximum number of URLs.
3. For each URL provide a short reason citing the link text or nearby context.
4. If you return document_title, it must appear verbatim as a substring of the
   page body.
"""


class LinkCandidate(BaseModel):
    reason: str = Field(
        default="",
        description="Short rationale based on anchor text or nearby context",
    )
    url: str = Field(description="URL copied exactly as it appears in the page body")


class PdfLinkCandidateList(BaseModel):
    document_title: str | None = Field(
        default=None,
        description=(
            "Exact document title from a PDF abstract/detail page only; "
            "null on all other pages"
        ),
    )
    links: list[LinkCandidate] = Field(default_factory=list)


class PdfPageExtract(BaseModel):
    """Validated URLs and optional document title from a crawled HTML page."""

    urls: list[str] = Field(default_factory=list)
    document_title: str | None = None


class AgentState(TypedDict):
    page_url: str
    page_body: str
    max_urls: int
    missing_urls: list[str]
    missing_title: str | None
    retries_left: int
    urls: list[str]
    document_title: str | None


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
    missing_title: str | None,
) -> str:
    parts = [
        f"Page URL: {page_url}",
        (
            f"Return at most {max_urls} highly relevant links "
            "(PDF downloads, PDF abstract pages, publication/document listing "
            "hubs, and search pagination only), in page order."
        ),
        (
            "If this is a single-document details/abstract page, also return "
            "document_title copied verbatim from the page body; otherwise null."
        ),
        "Page body follows:",
        page_body,
    ]
    corrections: list[str] = []
    if missing_urls:
        missing = "\n".join(f"- {url}" for url in missing_urls)
        corrections.append(
            "Your previous URL list contained values that do NOT appear as "
            "substrings in the page body. Produce a new list. These were not "
            f"found:\n{missing}\n"
            "Only include URLs that appear verbatim in the page body."
        )
    if missing_title is not None:
        corrections.append(
            "Your previous document_title does NOT appear as a substring in "
            f"the page body: {missing_title!r}\n"
            "Either copy the title exactly as it appears in the page body, "
            "or set document_title to null if this is not a document details "
            "page."
        )
    if corrections:
        parts.insert(0, "\n\n".join(corrections))
    return "\n\n".join(parts)


def _validate_urls_in_body(urls: list[str], page_body: str) -> list[str]:
    return [url for url in urls if url not in page_body]


def _normalize_title(title: str | None) -> str | None:
    if title is None:
        return None
    cleaned = " ".join(title.split())
    return cleaned or None


def _title_missing_from_body(title: str | None, page_body: str) -> str | None:
    if title is None:
        return None
    if title not in page_body:
        return title
    return None


def _log_links_detected(page_url: str, count: int) -> None:
    message = f"pdf_crawl links detected: {count} for {page_url}"
    print(message, flush=True)
    logger.info(message)


def _log_title_detected(page_url: str, title: str | None) -> None:
    if title is None:
        return
    message = f"pdf_crawl document title detected for {page_url}: {title}"
    print(message, flush=True)
    logger.info(message)


def _log_missing_url(page_url: str, missing_url: str) -> None:
    message = f"pdf_crawl URL not found in page body for {page_url}: {missing_url}"
    print(message, flush=True)
    logger.warning(message)


def _log_missing_title(page_url: str, missing_title: str) -> None:
    message = (
        f"pdf_crawl document title not found in page body for {page_url}: "
        f"{missing_title}"
    )
    print(message, flush=True)
    logger.warning(message)


def build_link_agent(model: BaseChatModel) -> Any:
    """Compile a LangGraph that extracts and validates page links and titles."""

    structured = model.with_structured_output(PdfLinkCandidateList)

    async def extract(state: AgentState) -> dict[str, Any]:
        prompt = _user_prompt(
            page_url=state["page_url"],
            page_body=state["page_body"],
            max_urls=state["max_urls"],
            missing_urls=state["missing_urls"],
            missing_title=state["missing_title"],
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
        document_title = _normalize_title(result.document_title)
        _log_links_detected(state["page_url"], len(urls))
        _log_title_detected(state["page_url"], document_title)
        return {
            "urls": urls,
            "document_title": document_title,
            "missing_urls": [],
            "missing_title": None,
        }

    async def validate(state: AgentState) -> dict[str, Any]:
        missing_urls = _validate_urls_in_body(state["urls"], state["page_body"])
        missing_title = _title_missing_from_body(
            state["document_title"], state["page_body"]
        )
        if missing_urls or missing_title is not None:
            for missing_url in missing_urls:
                _log_missing_url(state["page_url"], missing_url)
            if missing_title is not None:
                _log_missing_title(state["page_url"], missing_title)
            return {
                "missing_urls": missing_urls,
                "missing_title": missing_title,
                "retries_left": state["retries_left"] - 1,
            }
        valid = [url for url in state["urls"] if url in state["page_body"]]
        return {
            "urls": valid[: state["max_urls"]],
            "document_title": state["document_title"],
            "missing_urls": [],
            "missing_title": None,
        }

    def route_after_validate(
        state: AgentState,
    ) -> Literal["extract", "__end__"]:
        if (state["missing_urls"] or state["missing_title"] is not None) and state[
            "retries_left"
        ] > 0:
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
) -> PdfPageExtract:
    """Run the link agent; return validated URLs and optional document title."""
    chat_model = model or build_chat_model()
    agent = build_link_agent(chat_model)
    final_state: AgentState = await agent.ainvoke(
        {
            "page_url": page_url,
            "page_body": page_body,
            "max_urls": max_urls,
            "missing_urls": [],
            "missing_title": None,
            "retries_left": max_retries,
            "urls": [],
            "document_title": None,
        }
    )
    urls = final_state.get("urls", [])
    document_title = _normalize_title(final_state.get("document_title"))
    # Drop any still-invalid values after retries are exhausted.
    still_missing = _validate_urls_in_body(urls, page_body)
    for missing_url in still_missing:
        _log_missing_url(page_url, missing_url)
    valid_urls = [url for url in urls if url in page_body][:max_urls]
    if _title_missing_from_body(document_title, page_body) is not None:
        assert document_title is not None
        _log_missing_title(page_url, document_title)
        document_title = None
    return PdfPageExtract(urls=valid_urls, document_title=document_title)
