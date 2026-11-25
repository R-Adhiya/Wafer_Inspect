"""Application settings configuration using pydantic-settings."""

import json
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR: Path = Path(__file__).resolve().parents[3]

# Single-source-of-truth model metadata constants reported in inspection responses
MODEL_NAME: str = "WaferInspect-CV"
MODEL_VERSION: str = "1.0.0"
DETECTION_METHOD_LABEL: str = "Classical CV Baseline"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    DATA_DIR: Optional[str] = None
    DATABASE_URL: Optional[str] = None
    MAX_UPLOAD_SIZE: int = 20971520  # 20 MB default
    MODEL_PATH: str = "models/waferinspect_model.pt"
    LOG_LEVEL: str = "INFO"
    API_BASE_URL: str = "http://localhost:8000"
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Image upload and validation settings
    UPLOAD_DIR: str = "data/uploads"
    RESULTS_DIR: str = "data/results"
    MIN_IMAGE_DIMENSION: int = 64
    MAX_IMAGE_DIMENSION: int = 4096
    MAX_IMAGE_PIXELS: int = 16777216  # 4096 * 4096 (lowered from 8192 to guarantee peak process memory < 2 GB)

    @property
    def data_path(self) -> Path:
        """Resolve base data directory (defaults to <repo root>/data)."""
        if self.DATA_DIR:
            p = Path(self.DATA_DIR)
            return p if p.is_absolute() else (ROOT_DIR / p).resolve()
        return (ROOT_DIR / "data").resolve()

    @property
    def upload_path(self) -> Path:
        """Resolve upload directory path."""
        p = Path(self.UPLOAD_DIR)
        if p.is_absolute():
            return p
        if self.DATA_DIR:
            return (self.data_path / "uploads").resolve()
        return (ROOT_DIR / p).resolve()

    @property
    def results_path(self) -> Path:
        """Resolve results directory path."""
        p = Path(self.RESULTS_DIR)
        if p.is_absolute():
            return p
        if self.DATA_DIR:
            return (self.data_path / "results").resolve()
        return (ROOT_DIR / p).resolve()

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_database_url(cls, v: Optional[str], info: Any) -> str:
        """Resolve DATABASE_URL, defaulting to SQLite under DATA_DIR if unset."""
        if v:
            return v
        data_dir = info.data.get("DATA_DIR") if info.data else None
        if data_dir:
            p = Path(data_dir)
            base = p if p.is_absolute() else (ROOT_DIR / p).resolve()
            return f"sqlite:///{(base / 'waferinspect.db').as_posix()}"
        return f"sqlite:///{(ROOT_DIR / 'data' / 'waferinspect.db').as_posix()}"

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
