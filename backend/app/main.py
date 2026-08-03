"""FastAPI application factory and ASGI entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router import api_router
from backend.app.core.config import get_settings
from backend.app.core.errors import register_exception_handlers
from backend.app.core.logging import configure_logging, get_logger
from backend.app.core.middleware import RequestContextMiddleware
from backend.app.dependencies import ApplicationContainer

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and release application-owned infrastructure resources."""
    logger.info("application_starting", extra={"environment": settings.environment})
    app.state.container = ApplicationContainer.create(settings)
    app.state.container.initialize()
    yield
    app.state.container.close()
    logger.info("application_stopping")


def create_app() -> FastAPI:
    """Create a configured FastAPI application suitable for ASGI servers and tests."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Local-first adaptive agentic RAG API.",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
