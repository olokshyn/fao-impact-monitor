from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_DIR = Path(__file__).parent.parent.parent / "fetched_data"

_COMMON_SETTINGS = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


class AwsBedrockConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AWS_BEDROCK_",
        **_COMMON_SETTINGS,
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
        **_COMMON_SETTINGS,
    )

    max_url_depth: int = 10
    max_pdfs: int = 50
    max_urls: int | None = None  # None = unlimited (includes the seed URL)
    max_pdf_size: int = 10 * 1024 * 1024  # 10MB
    save_dir: Path = DATA_DIR / "pdf"
    web_page_save_dir: Path = DATA_DIR / "web_page"
    max_urls_per_page: int = 10
    llm_model: str = "openai:openai.gpt-5.6-luna"
    max_agent_retries: int = 3
    # How many descendant URL hops a details-page title remains valid for.
    detected_title_validity_depth: int = 3


class PdfExtractConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PDF_EXTRACT_",
        **_COMMON_SETTINGS,
    )

    save_dir: Path = DATA_DIR / "pdf_markdown"


class CountryDetectConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COUNTRY_DETECT_",
        **_COMMON_SETTINGS,
    )

    llm_model: str = "openai:openai.gpt-5.6-luna"
    max_agent_retries: int = 3


class QueryGeneratorConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUERY_GENERATOR_",
        **_COMMON_SETTINGS,
    )

    llm_model: str = "openai:openai.gpt-5.6-luna"
    max_agent_retries: int = 3
    min_queries: int = 3
    max_queries: int = 5


class TellusConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TELLUS_",
        **_COMMON_SETTINGS,
    )

    bearer_token: SecretStr = SecretStr("")
    api_base: str = "https://tellus.dev.fao.org"
    save_dir: Path = DATA_DIR / "tellus"
    min_year: int = 1997
    max_results: int = 50


class VectorStoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VECTOR_STORE_",
        **_COMMON_SETTINGS,
    )

    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int | None = 1024
    collection_name: str = "embeddings"
    vector_index_name: str = "embeddings_vector_index"
    text_index_name: str = "embeddings_text_index"
    vector_num_candidates: int = 100
    limit: int = 10
    vector_weight: float = 1.0
    text_weight: float = 1.0
    embed_batch_size: int = 32


class MongoConfig(BaseSettings):
    """MongoDB connection settings (local Atlas Compose defaults)."""

    model_config = SettingsConfigDict(
        env_prefix="MONGO_",
        **_COMMON_SETTINGS,
    )

    host: str = "127.0.0.1"
    port: int = 27018
    db_name: str = "fao_impact_monitor"
    username: SecretStr = SecretStr("")
    password: SecretStr = SecretStr("")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uri(self) -> str:
        user = self.username.get_secret_value()
        password = self.password.get_secret_value()
        if not user or not password:
            raise ValueError("Set MONGO_USERNAME and MONGO_PASSWORD in .env")
        return (
            f"mongodb://{quote_plus(user)}:{quote_plus(password)}"
            f"@{self.host}:{self.port}/?directConnection=true&authSource=admin"
        )


class Config(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON_SETTINGS)

    aws_bedrock: AwsBedrockConfig = Field(default_factory=AwsBedrockConfig)
    pdf_crawl: PdfCrawlConfig = Field(default_factory=PdfCrawlConfig)
    pdf_extract: PdfExtractConfig = Field(default_factory=PdfExtractConfig)
    country_detect: CountryDetectConfig = Field(default_factory=CountryDetectConfig)
    query_generator: QueryGeneratorConfig = Field(default_factory=QueryGeneratorConfig)
    tellus: TellusConfig = Field(default_factory=TellusConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    mongo: MongoConfig = Field(default_factory=MongoConfig)


def get_config() -> Config:
    return Config()
