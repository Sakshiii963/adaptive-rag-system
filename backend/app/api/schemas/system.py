"""System metadata schemas."""

from pydantic import BaseModel


class SystemInfoResponse(BaseModel):
    """Public, non-secret deployment information."""

    app_name: str
    version: str
    environment: str
