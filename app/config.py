
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # Application
    app_name: str = "Research Agent with Cited Reports"
    app_env: str = "development"

    # Gemini configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"

    # Tavily configuration
    tavily_api_key: str = ""

    # Database
    database_url: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Research limits
    max_research_steps: int = 8
    max_research_time_seconds: int = 300
    max_report_cost_usd: float = 1.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
