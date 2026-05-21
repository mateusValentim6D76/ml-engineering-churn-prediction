from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import ModelContainer, get_model
from src.api.schemas import HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health(model: ModelContainer = Depends(get_model)):
    return HealthResponse(
        status="healthy" if model.loaded else "degraded",
        model_loaded=model.loaded,
        version=model.version,
        model_metrics=model.metrics,
    )
