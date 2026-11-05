from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Research Agent with Cited Reports"
    app_env: str = "development"
    gemini_api_key: str = ""
    tavily_api_key: str = ""
    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"
    max_research_steps: int = 8
    max_research_time_seconds: int = 300
    max_report_cost_usd: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
