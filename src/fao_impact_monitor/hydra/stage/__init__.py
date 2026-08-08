from typing import TYPE_CHECKING, Any

from fao_impact_monitor.hydra.stage.stage import (
    Stage,
    StageResult,
    get_stage,
    get_stage_result_class,
)

if TYPE_CHECKING:
    from fao_impact_monitor.hydra.stage.fetch_stage import (
        ContentType,
        Fetch,
        FetchRequest,
        FetchResponse,
        FetchStage,
        FetchStageResult,
    )

__all__ = [
    "ContentType",
    "Fetch",
    "FetchRequest",
    "FetchResponse",
    "FetchStage",
    "FetchStageResult",
    "Stage",
    "StageResult",
    "get_stage",
    "get_stage_result_class",
]

_FETCH_EXPORTS = {
    "ContentType",
    "Fetch",
    "FetchRequest",
    "FetchResponse",
    "FetchStage",
    "FetchStageResult",
}


def __getattr__(name: str) -> Any:
    if name in _FETCH_EXPORTS:
        from fao_impact_monitor.hydra.stage import fetch_stage as _fetch_stage

        return getattr(_fetch_stage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
