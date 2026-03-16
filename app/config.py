import secrets
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./lottery.db"
    SECRET_KEY: str = secrets.token_urlsafe(64)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    CA_LOTTERY_API_BASE: str = "https://www.calottery.com/api/DrawGameApi"
    DAILY3_GAME_ID: int = 9
    DAILY4_GAME_ID: int = 14
    API_PAGE_SIZE: int = 50
    FETCH_RETRY_ATTEMPTS: int = 3

    APP_MODE: str = "local"  # "local" or "cloud"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()
