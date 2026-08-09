"""LinkExtractStage: extract topic-relevant crawl links from fetched HTML."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from fao_impact_monitor.hydra.document.document import Document
from fao_impact_monitor.hydra.stage.fetch_stage import ContentType, FetchStageResult
from fao_impact_monitor.hydra.stage.stage import Stage, StageResult
from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task, TaskState
from fao_impact_monitor.utils.fs import read_file

if TYPE_CHECKING:
    from fao_impact_monitor.agent.link_extract_agent import LinkExtractAgent
    from fao_impact_monitor.config import LinkExtractConfig

logger = logging.getLogger(__name__)

ExtractFn = Callable[..., Awaitable[list[str]]]


class LinkExtractStageResult(StageResult):
    name: str = "link_extract"
    urls: list[str] = Field(default_factory=list)


def _latest_fetch_result(
    document: Document,
    *,
    workflow_name: str,
    node_name: str,
) -> FetchStageResult | None:
    results = document.stage_results.get(workflow_name, {}).get(node_name) or []
    for result in reversed(results):
        if isinstance(result, FetchStageResult):
            return result
        if getattr(result, "name", None) == "fetch":
            return FetchStageResult.model_validate(
                result.model_dump() if isinstance(result, StageResult) else result
            )
    return None


class LinkExtractStage(Stage):
    """Read fetched HTML and extract crawl candidate URLs via LinkExtractAgent."""

    name = "link_extract"
    context_required: ClassVar[dict[str, str] | None] = {
        "topic": "research topic used to decide which links are relevant",
        "data_filter": (
            "exact requirements a data source must satisfy to be eligible "
            "for collection"
        ),
        "depth": "number of hops from the root URL; root has depth 0",
    }

    def __init__(
        self,
        *,
        extract_fn: ExtractFn | None = None,
        agent: LinkExtractAgent | None = None,
        config: LinkExtractConfig | None = None,
    ) -> None:
        self._extract_fn = extract_fn
        self._agent = agent
        self._config = config

    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> tuple[StageResult, TaskState | None]:
        context = dict(task.context or {})
        try:
            topic = str(context["topic"])
            data_filter = str(context["data_filter"])
            depth = int(context["depth"])
        except KeyError as exc:
            raise RuntimeError(f"Missing required context key: {exc.args[0]}")

        if task.document_id is None:
            raise RuntimeError("Task.document_id is required")

        document = await Document.get(task.document_id)
        if document is None:
            raise RuntimeError(f"Document not found: {task.document_id}")

        fetch_node_name = params.get("fetch_node_name")
        if not isinstance(fetch_node_name, str) or not fetch_node_name:
            raise RuntimeError("stage_params['fetch_node_name'] is required")

        fetch_workflow_name = params.get("fetch_workflow_name") or workflow_name
        if not isinstance(fetch_workflow_name, str) or not fetch_workflow_name:
            fetch_workflow_name = workflow_name

        fetch_result = _latest_fetch_result(
            document,
            workflow_name=fetch_workflow_name,
            node_name=fetch_node_name,
        )
        if fetch_result is None:
            raise RuntimeError(
                f"No FetchStageResult at "
                f"stage_results[{fetch_workflow_name!r}][{fetch_node_name!r}]"
            )

        if fetch_result.status != Status.COMPLETED:
            raise RuntimeError(f"Prior fetch not completed: {fetch_result.error}")

        if fetch_result.content_type != ContentType.HTML:
            raise RuntimeError(
                f"Expected HTML content_type, got {fetch_result.content_type!r}"
            )

        if not fetch_result.body_path:
            raise RuntimeError("FetchStageResult.body_path is missing")

        if not task.url:
            raise RuntimeError("Task.url is required")

        try:
            page_body = read_file(fetch_result.body_path, "r")
        except Exception as exc:
            logger.exception("Failed to read body for %s", task.url, exc_info=exc)
            raise RuntimeError(f"Failed to read body_path: {exc}")

        # Lazy imports: avoid circular import via hydra.__init__ → this module
        # → config while config is still loading HydraConfig.
        from fao_impact_monitor.agent.link_extract_agent import (
            configure_dspy_lm,
            extract_page_urls,
            load_link_extract_agent,
        )
        from fao_impact_monitor.config import get_config

        config = self._config or get_config().link_extract
        try:
            if self._extract_fn is not None:
                urls = await self._extract_fn(
                    page_url=task.url,
                    page_body=page_body,
                    topic=topic,
                    data_filter=data_filter,
                    max_urls=config.max_urls_per_page,
                    max_retries=config.max_agent_retries,
                )
            else:
                agent = self._agent
                if agent is None:
                    configure_dspy_lm(link_extract=config)
                    agent = load_link_extract_agent(config.dspy_state_path)
                urls = await extract_page_urls(
                    page_url=task.url,
                    page_body=page_body,
                    topic=topic,
                    data_filter=data_filter,
                    max_urls=config.max_urls_per_page,
                    max_retries=config.max_agent_retries,
                    agent=agent,
                )
        except Exception as exc:
            logger.exception(
                "Link extraction failed for %s: %s", task.url, exc_info=exc
            )
            result = LinkExtractStageResult(
                name=self.name,
                status=Status.FAILED,
                error=str(exc),
            )
            await document.push_stage_result(workflow_name, workflow_node_name, result)
            return result, None

        result = LinkExtractStageResult(
            name=self.name,
            status=Status.COMPLETED,
            urls=list(urls),
        )
        await document.push_stage_result(workflow_name, workflow_node_name, result)

        child_context = {
            **context,
            "depth": depth + 1,
        }
        return result, TaskState(
            context=child_context,
            priority=task.priority + 1,
        )
