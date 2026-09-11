from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "WaferInspect API"
    APP_VERSION: str = "0.1.0"
    DATABASE_URL: str = "sqlite:///./waferinspect.db"
    MAX_UPLOAD_SIZE: int = 20 * 1024 * 1024  # 20MB
    MODEL_PATH: str = "models/waferinspect_model_v1.pt"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
