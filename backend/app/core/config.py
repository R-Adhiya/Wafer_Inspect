"""Application settings configuration using pydantic-settings."""

import json
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR: Path = Path(__file__).resolve().parents[3]

# Single-source-of-truth model metadata constants reported in inspection responses
MODEL_NAME: str = "WaferInspect-CV"
MODEL_VERSION: str = "1.0.0"
DETECTION_METHOD_LABEL: str = "Classical CV Baseline"



class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    DATABASE_URL: str = "sqlite:///./waferinspect.db"
    MAX_UPLOAD_SIZE: int = 20971520  # 20 MB default
    MODEL_PATH: str = "models/waferinspect_model.pt"
    LOG_LEVEL: str = "INFO"
    API_BASE_URL: str = "http://localhost:8000"
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]

    # Image upload and validation settings
    UPLOAD_DIR: str = "data/uploads"
    RESULTS_DIR: str = "data/results"
    MIN_IMAGE_DIMENSION: int = 64
    MAX_IMAGE_DIMENSION: int = 8192
    MAX_IMAGE_PIXELS: int = 67108864  # 8192 * 8192

    @property
    def upload_path(self) -> Path:
        """Resolve upload directory path relative to project root if not absolute."""
        p = Path(self.UPLOAD_DIR)
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()

    @property
    def results_path(self) -> Path:
        """Resolve results directory path relative to project root if not absolute."""
        p = Path(self.RESULTS_DIR)
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()

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
