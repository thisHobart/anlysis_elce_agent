from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "VPP LangGraph Agent"
    environment: str = "development"
    allowed_origins: str = "*"
    enable_screen_actions: bool = True
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VPP_", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
