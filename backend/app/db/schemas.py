"""
Pydantic Schemas Architecture (Phase 1 Foundation).
Defines Health API schema and future inspection response contracts.
"""
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
