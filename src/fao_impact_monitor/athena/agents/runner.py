"""OpenAI Agents SDK runner. Decisions are logged before the caller continues."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx
from pydantic import BaseModel

from fao_impact_monitor.athena.observe import DecisionLog, DecisionRecord, input_hash
from fao_impact_monitor.athena.storage.rdf import utc_stamp
from fao_impact_monitor.config import get_config

logger = logging.getLogger(__name__)

# The SDK connect timeout is 5s. A slow handshake to Bedrock then aborts ingest.
_TIMEOUT = httpx.Timeout(600.0, connect=60.0)
_ATTEMPTS = 6
_RETRY_DELAY_SECONDS = 2.0


def openai_model_name(name: str) -> str:
    if name.startswith("openai:"):
        return name.split(":", 1)[1]
    return name


def _should_retry(exc: Exception) -> bool:
    from openai import APIConnectionError, APIStatusError

    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


async def _run_with_retry[T](
    operation: Callable[[], Awaitable[T]], *, attempts: int = _ATTEMPTS
) -> T:
    from openai import APIConnectionError, APIStatusError

    delay = _RETRY_DELAY_SECONDS
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except (APIConnectionError, APIStatusError) as exc:
            if not _should_retry(exc) or attempt == attempts:
                raise
            logger.warning(
                "model request failed (%s); retrying %s/%s",
                type(exc).__name__,
                attempt + 1,
                attempts,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("model retry loop ended without a result")


async def run_structured[T: BaseModel](
    *,
    agent: str,
    instructions: str,
    output_type: type[T],
    model_name: str,
    prompt: str,
    decisions: DecisionLog,
    claim_ids: list[str] | None = None,
) -> T:
    from agents import (
        Agent,
        AgentOutputSchema,
        OpenAIChatCompletionsModel,
        Runner,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    bedrock = get_config().aws_bedrock
    client = AsyncOpenAI(
        base_url=bedrock.base_url,
        api_key=bedrock.api_key.get_secret_value() or "missing",
        timeout=_TIMEOUT,
        max_retries=0,
    )
    model = OpenAIChatCompletionsModel(
        model=openai_model_name(model_name),
        openai_client=client,
    )
    built = Agent(
        name=agent,
        instructions=instructions,
        model=model,
        output_type=AgentOutputSchema(output_type, strict_json_schema=False),
    )
    result = await _run_with_retry(lambda: Runner.run(built, prompt))
    parsed = result.final_output
    if not isinstance(parsed, output_type):
        parsed = output_type.model_validate(parsed)
    decisions.write(
        DecisionRecord(
            agent=agent,
            model=openai_model_name(model_name),
            stage=agent,
            input_hash=input_hash(prompt),
            output=parsed.model_dump(mode="json"),
            claim_ids=claim_ids or [],
            timestamp=utc_stamp(),
        )
    )
    return parsed
