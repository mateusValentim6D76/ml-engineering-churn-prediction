"""
Router de Health Check.


    Aqui criamos manualmente, mas a ideia é a mesma do actuator/health:
    ALB, Docker e Kubernetes usam esse endpoint para saber se a API esta de pe.

Uso do APIRouter:
    APIRouter é como definir um @RestController separado.
    Depois ele é "registrado" no app principal (como @ComponentScan encontra controllers).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import ModelContainer, get_model
from src.api.schemas import HealthResponse

# Cria o router (equivalente a RestController)
router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health(model: ModelContainer = Depends(get_model)):
    """Health check usado pelo ALB e Docker.

    O parametro `model: ModelContainer = Depends(get_model)` é o DI:
    FastAPI chama get_model() e injeta o resultado aqui automaticamente.

    """
    return HealthResponse(
        status="healthy" if model.loaded else "degraded",
        model_loaded=model.loaded,
        version=model.version,
        model_metrics=model.metrics,
    )
