from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "VPP LangGraph Agent"
    environment: str = "development"
    allowed_origins: str = "*"
    enable_screen_actions: bool = True
    log_level: str = "INFO"
    llm_enabled: bool = False
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 1
    llm_structured_mode: Literal["json_prompt", "native"] = "json_prompt"
    llm_history_messages: int = 8
    db_host: str = "db.example.invalid"
    db_port: int = 3306
    db_user: str = "readonly_user"
    db_password: SecretStr = SecretStr("")
    db_name: str = "example_db"
    db_charset: str = "utf8mb4"
    # P6 知识检索配置
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    knowledge_collection: str = "vpp_knowledge"
    knowledge_top_k: int = 3
    knowledge_model_path: str = "BAAI/bge-small-zh-v1.5"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VPP_", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
