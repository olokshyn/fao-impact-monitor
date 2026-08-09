"""DSPy agent that extracts topic-relevant crawl links from HTML pages."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import dspy

from fao_impact_monitor.config import AwsBedrockConfig, LinkExtractConfig, get_config

logger = logging.getLogger(__name__)

EXTRACT_INSTRUCTIONS = """\
You are a link-extraction agent.

Your job is to read an HTML page and select hyperlinks that advance a crawl
toward documentary evidence for the given research topic, subject to the
data_filter constraints.

Always use topic and data_filter when deciding which links are relevant.
Only keep links that help gather evidence matching both.

Select:
1. URLs that download a PDF or other document directly (href contains .pdf,
   or clear download actions such as "Download PDF", "Full report").
2. URLs that lead to a document abstract / detail page — a page that shows
   the document title, abstract/summary, and a download control.
3. URLs that lead to publication / document listing hubs clearly on the path
   to relevant documents (archives, repositories, "publications",
   "resources", "documents", "reports").
4. URLs to HTML pages that themselves provide evidence for the topic (and
   satisfy data_filter) — reports, assessments, situation updates, articles,
   or other content pages whose substance is on-topic, not generic site chrome.
5. Relevant HTML pages that continue the research trail toward such evidence
   for topic/data_filter.
6. URLs that lead to the next page of search results (pagination: "Next",
   page numbers, "Load more"). If the page is a paginated result list, you
   MUST include the link to the next page.

Do NOT select:
- Keyword / tag / topic facet links
- Links to parts of a document (chapters, sections, TOC anchors)
- Author, contributor, or profile pages
- Login, language switchers, share widgets, cookie banners, social media,
  footer/header chrome, unrelated site navigation
- Links unrelated to topic or that fail data_filter

Ordering:
- Return URLs in the same order they appear on the page.
- Put pagination ("next page") links after the on-page result/document links.

Critical rules:
1. Copy each URL string EXACTLY as it appears in the page body/source (href
   or visible URL text). Do not invent, normalize, or absolutize URLs.
2. Return at most max_urls URLs.
3. If correction is non-empty, previous URLs were invalid; produce a new list
   using only URLs that appear verbatim in the page body.
"""


class ExtractLinks(dspy.Signature):  # type: ignore[misc]
    __doc__ = EXTRACT_INSTRUCTIONS

    page_url: str = dspy.InputField(desc="URL of the HTML page being analyzed")
    page_body: str = dspy.InputField(desc="Raw HTML / text body of the page")
    topic: str = dspy.InputField(desc="Research topic that selected links must serve")
    data_filter: str = dspy.InputField(
        desc="Exact requirements a data source must satisfy to be eligible"
    )
    max_urls: int = dspy.InputField(desc="Maximum number of URLs to return")
    correction: str = dspy.InputField(
        desc=(
            "Empty on the first attempt. Otherwise lists URLs from the previous "
            "attempt that were not found as substrings in page_body."
        )
    )
    urls: list[str] = dspy.OutputField(
        desc=(
            "Relevant URLs copied exactly from the page body, in page order, "
            "including next-page pagination when present"
        )
    )


def _langchain_model_to_dspy(model: str) -> str:
    """Convert ``openai:model.id`` (LangChain) to ``openai/model.id`` (LiteLLM)."""
    if model.startswith("openai:"):
        return "openai/" + model.removeprefix("openai:")
    if "/" in model:
        return model
    return f"openai/{model}"


def build_lm(
    link_extract: LinkExtractConfig | None = None,
    aws_bedrock: AwsBedrockConfig | None = None,
) -> dspy.LM:
    """Build a DSPy LM pointed at the Bedrock OpenAI-compatible endpoint."""
    config = get_config()
    link_extract = link_extract or config.link_extract
    aws_bedrock = aws_bedrock or config.aws_bedrock
    return dspy.LM(
        model=_langchain_model_to_dspy(link_extract.llm_model),
        api_key=aws_bedrock.api_key.get_secret_value(),
        api_base=aws_bedrock.base_url,
        model_type="responses",
    )


def validate_urls_in_body(urls: list[str], page_body: str) -> list[str]:
    """Return URLs that do not appear as substrings of ``page_body``."""
    return [url for url in urls if url not in page_body]


def _correction_message(missing_urls: list[str]) -> str:
    missing = "\n".join(f"- {url}" for url in missing_urls)
    return (
        "Your previous URL list contained values that do NOT appear as "
        "substrings in the page body. Produce a new list. These were not "
        f"found:\n{missing}\n"
        "Only include URLs that appear verbatim in the page body."
    )


def _log_links_detected(page_url: str, count: int) -> None:
    message = f"link_extract links detected: {count} for {page_url}"
    print(message, flush=True)
    logger.info(message)


def _log_missing_url(page_url: str, missing_url: str) -> None:
    message = f"link_extract URL not found in page body for {page_url}: {missing_url}"
    print(message, flush=True)
    logger.warning(message)


class LinkExtractAgent(dspy.Module):  # type: ignore[misc]
    """Extract and validate topic-relevant links from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractLinks)

    def forward(
        self,
        *,
        page_url: str,
        page_body: str,
        topic: str,
        data_filter: str,
        max_urls: int,
        max_retries: int = 3,
    ) -> dspy.Prediction:
        # Unescape entities so href values with ``&amp;`` match returned ``&``.
        body = html.unescape(page_body)
        correction = ""
        urls: list[str] = []
        for _ in range(max_retries + 1):
            prediction = self.extract(
                page_url=page_url,
                page_body=body,
                topic=topic,
                data_filter=data_filter,
                max_urls=max_urls,
                correction=correction,
            )
            raw_urls = list(prediction.urls or [])
            urls = [url for url in raw_urls if isinstance(url, str)][:max_urls]
            _log_links_detected(page_url, len(urls))
            missing = validate_urls_in_body(urls, body)
            if not missing:
                return dspy.Prediction(urls=urls)
            for missing_url in missing:
                _log_missing_url(page_url, missing_url)
            correction = _correction_message(missing)

        still_missing = validate_urls_in_body(urls, body)
        for missing_url in still_missing:
            _log_missing_url(page_url, missing_url)
        valid = [url for url in urls if url in body][:max_urls]
        return dspy.Prediction(urls=valid)


def configure_dspy_lm(
    link_extract: LinkExtractConfig | None = None,
    aws_bedrock: AwsBedrockConfig | None = None,
) -> dspy.LM:
    """Configure the global DSPy LM and return it."""
    lm = build_lm(link_extract=link_extract, aws_bedrock=aws_bedrock)
    dspy.configure(lm=lm)
    return lm


def load_link_extract_agent(
    state_path: Path | str | None = None,
    *,
    require_state: bool = True,
) -> LinkExtractAgent:
    """Create a ``LinkExtractAgent`` and load trained DSPy state from disk."""
    config = get_config().link_extract
    path = Path(state_path) if state_path is not None else Path(config.dspy_state_path)
    agent = LinkExtractAgent()
    if path.is_file():
        agent.load(str(path))
        logger.info("Loaded LinkExtractAgent state from %s", path)
    elif require_state:
        raise FileNotFoundError(
            f"Trained LinkExtractAgent state not found at {path}. "
            "Run train-link-extract-agent first."
        )
    return agent


def save_link_extract_agent(agent: LinkExtractAgent, state_path: Path | str) -> None:
    """Persist compiled agent state to ``state_path``."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    logger.info("Saved LinkExtractAgent state to %s", path)


async def extract_page_urls(
    *,
    page_url: str,
    page_body: str,
    topic: str,
    data_filter: str,
    max_urls: int | None = None,
    max_retries: int | None = None,
    agent: LinkExtractAgent | None = None,
    state_path: Path | str | None = None,
) -> list[str]:
    """Run the link agent; return validated URLs in page order."""
    config = get_config().link_extract
    max_urls = config.max_urls_per_page if max_urls is None else max_urls
    max_retries = config.max_agent_retries if max_retries is None else max_retries
    if agent is None:
        configure_dspy_lm()
        agent = load_link_extract_agent(state_path)
    result = agent(
        page_url=page_url,
        page_body=page_body,
        topic=topic,
        data_filter=data_filter,
        max_urls=max_urls,
        max_retries=max_retries,
    )
    if not isinstance(result, dspy.Prediction):
        raise TypeError(f"Expected dspy.Prediction, got {type(result)}")
    urls = getattr(result, "urls", [])
    if not urls or not isinstance(urls, list):
        raise ValueError(f"No URLs found. Expected list of URLs, got {type(urls)}")
    return urls
