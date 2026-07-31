from pydantic import BaseModel, ConfigDict


class DataSourceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    unit: str | None = None
