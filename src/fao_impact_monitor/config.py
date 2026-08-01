from pathlib import Path

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_DIR = Path(__file__).parent.parent.parent / "fetched_data"


class AwsBedrockConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AWS_BEDROCK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = SecretStr("")
    region: str = "us-east-1"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_url(self) -> str:
        return f"https://bedrock-mantle.{self.region}.api.aws/openai/v1"


class PdfCrawlConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PDF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_url_depth: int = 10
    max_pdfs: int = 50
    max_pdf_size: int = 10 * 1024 * 1024  # 10MB
    save_dir: Path = DATA_DIR / "pdf"
    web_page_save_dir: Path = DATA_DIR / "web_page"
    max_urls_per_page: int = 10
    llm_model: str = "openai:openai.gpt-5.6-luna"
    max_agent_retries: int = 3


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aws_bedrock: AwsBedrockConfig = Field(default_factory=AwsBedrockConfig)
    pdf_crawl: PdfCrawlConfig = Field(default_factory=PdfCrawlConfig)


def get_config() -> Config:
    return Config()
