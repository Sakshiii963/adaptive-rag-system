"""Liveness endpoint used by orchestration and deployment tooling."""

from datetime import UTC, datetime

from fastapi import APIRouter, status

from backend.app.api.schemas.health import HealthResponse
from backend.app.core.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """Return process liveness without probing future RAG dependencies."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(UTC),
    )
