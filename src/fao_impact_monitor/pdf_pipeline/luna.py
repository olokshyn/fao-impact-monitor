"""Luna summaries of already validated section scope evidence only."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from fao_impact_monitor.config import AwsBedrockConfig, PdfPipelineConfig, get_config

SYSTEM = """Write a concise retrieval description from the supplied validated source evidence.
Do not add facts, values, countries, El Nino attribution, or dates absent from that source.
This is indexing material only, not evidence. Mention forecast/observed/scenario labels when supplied."""


class LunaSectionSummarizer:
    def __init__(
        self,
        config: PdfPipelineConfig | None = None,
        aws_config: AwsBedrockConfig | None = None,
    ) -> None:
        settings = get_config()
        self.config = config or settings.pdf_pipeline
        aws = aws_config or settings.aws_bedrock
        self.model = init_chat_model(
            self.config.luna_model,
            base_url=aws.base_url,
            api_key=aws.api_key.get_secret_value(),
            use_responses_api=True,
        )

    async def summarize(self, *, title: str, scope_source_text: str) -> str:
        response = await self.model.ainvoke(
            [
                SystemMessage(SYSTEM),
                HumanMessage(
                    f"Section title: {title}\nValidated source evidence:\n{scope_source_text}"
                ),
            ]
        )
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ]
            if text_parts:
                return "\n".join(text_parts)
        raise ValueError("Luna returned no text summary")
