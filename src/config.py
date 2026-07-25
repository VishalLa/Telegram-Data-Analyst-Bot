import os
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    DEBUG: bool = False

    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    MODEL_NAME: str = "openai/gpt-oss-120b"
    TEMPERATURE: float = 0.0

    TELEGRAM_TOKEN: str
    TELEGRAM_API: Optional[str] = None

    MAX_HISTORY_PER_CHAT: int = 10

    LOG_DIR: str = "logs"
    LOG_FILE: str = "run.jsonl"
    LOG_URL: str = "https://telegram-data-analyst-bot-kf8a.onrender.com/run.jsonl"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _finalize(self) -> "Settings":
        self.TELEGRAM_API = f"https://api.telegram.org/bot{self.TELEGRAM_TOKEN}"
        os.makedirs(self.LOG_DIR, exist_ok=True)
        return self

    @property
    def LOG_PATH(self) -> str:
        return os.path.join(self.LOG_DIR, self.LOG_FILE)


settings = Settings()
