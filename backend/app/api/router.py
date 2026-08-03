"""Versioned API router composition."""

from fastapi import APIRouter

from backend.app.api.routes.agent import router as agent_router
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.generation import router as generation_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.jobs import router as jobs_router
from backend.app.api.routes.retrieval import router as retrieval_router
from backend.app.api.routes.system import router as system_router
from backend.app.api.routes.verification import router as verification_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(agent_router, prefix="/agent", tags=["adaptive-agent"])
api_router.include_router(generation_router, prefix="/generation", tags=["generation"])
api_router.include_router(jobs_router, prefix="/jobs", tags=["indexing-jobs"])
api_router.include_router(retrieval_router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(verification_router, prefix="/verification", tags=["verification"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
