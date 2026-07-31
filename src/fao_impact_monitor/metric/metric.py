from pydantic import BaseModel

from fao_impact_monitor.data_source.data_source_config import DataSourceConfig


class Metric(BaseModel):
    name: str
    description: str
    example: str
    unit: str
    data_sources: list[DataSourceConfig]
