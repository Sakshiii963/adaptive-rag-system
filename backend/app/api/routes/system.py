"""Non-sensitive system metadata routes."""

from fastapi import APIRouter

from backend.app.api.schemas.system import SystemInfoResponse
from backend.app.core.config import get_settings

router = APIRouter()


@router.get("/info", response_model=SystemInfoResponse)
async def system_info() -> SystemInfoResponse:
    """Expose safe deployment metadata for the frontend and operators."""
    settings = get_settings()
    return SystemInfoResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
