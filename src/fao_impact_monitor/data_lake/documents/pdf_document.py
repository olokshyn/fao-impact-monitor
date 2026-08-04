from pydantic import Field, computed_field

from fao_impact_monitor.data_lake.document import Document, DocumentType


class PdfDocument(Document):
    page_paths: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        return f"{self.title} ({self.url})"

    class Settings:
        class_id_value = DocumentType.PDF
