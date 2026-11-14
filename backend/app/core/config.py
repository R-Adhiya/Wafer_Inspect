"""Application settings configuration using pydantic-settings."""

import json
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    DATABASE_URL: str = "sqlite:///./waferinspect.db"
    MAX_UPLOAD_SIZE: int = 20971520
    MODEL_PATH: str = "models/waferinspect_model.pt"
    LOG_LEVEL: str = "INFO"
    API_BASE_URL: str = "http://localhost:8000"
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS_ORIGINS from list or formatted string."""
        if isinstance(v, str):
            trimmed = v.strip()
            if trimmed.startswith("[") and trimmed.endswith("]"):
                try:
                    parsed = json.loads(trimmed)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return [origin.strip() for origin in trimmed.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
