"""Health endpoint schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Stable liveness payload."""

    status: Literal["ok"]
    service: str
    version: str
    timestamp: datetime
