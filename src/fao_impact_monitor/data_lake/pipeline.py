from datetime import UTC, datetime
from typing import Annotated, Any

from beanie import Document as BeanieDocument
from beanie import Indexed
from pydantic import BaseModel, Field

from .document import Document
from .stage import StageResult, StageStatus, get_stage


class PipelineStep(BaseModel):
    stage_name: str
    params: dict[str, Any]


class Pipeline(BeanieDocument):
    name: Annotated[str, Indexed(unique=True)]
    steps: list[PipelineStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def is_completed(self, document: Document) -> bool:
        for step in self.steps:
            if self._get_stage_result(document, step) is None:
                return False
        return True

    async def run(self, document: Document) -> None:
        # TODO: re-run stage if the upstream or the current stage version has changed
        for step in self.steps:
            if self._get_stage_result(document, step) is not None:
                continue
            stage = get_stage(step.stage_name)
            prev_results = self._get_prev_results(document, step)
            result = await stage.run(document, step.params, prev_results)
            document.stage_results.setdefault(step.stage_name, []).append(result)
            await document.save()

    def _get_stage_result(
        self, document: Document, step: PipelineStep
    ) -> StageResult | None:
        stage_results = document.stage_results.get(step.stage_name)
        if not stage_results or stage_results[-1].status != StageStatus.COMPLETED:
            return None
        return stage_results[-1]

    def _get_prev_results(
        self, document: Document, current_step: PipelineStep
    ) -> list[StageResult]:
        prev_results: list[StageResult] = []
        for step in self.steps:
            if step.stage_name == current_step.stage_name:
                break
            result = self._get_stage_result(document, step)
            if result is None:
                break
            prev_results.append(result)
        return prev_results
