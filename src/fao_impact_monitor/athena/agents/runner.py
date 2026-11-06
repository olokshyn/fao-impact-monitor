"""OpenAI Agents SDK runner. Decisions are logged before the caller continues."""

from __future__ import annotations

from pydantic import BaseModel

from fao_impact_monitor.athena.observe import DecisionLog, DecisionRecord, input_hash
from fao_impact_monitor.athena.storage.rdf import utc_stamp
from fao_impact_monitor.config import get_config


def openai_model_name(name: str) -> str:
    if name.startswith("openai:"):
        return name.split(":", 1)[1]
    return name


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
    from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    bedrock = get_config().aws_bedrock
    client = AsyncOpenAI(
        base_url=bedrock.base_url,
        api_key=bedrock.api_key.get_secret_value() or "missing",
    )
    model = OpenAIChatCompletionsModel(
        model=openai_model_name(model_name),
        openai_client=client,
    )
    built = Agent(
        name=agent,
        instructions=instructions,
        model=model,
        output_type=output_type,
    )
    result = await Runner.run(built, prompt)
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
