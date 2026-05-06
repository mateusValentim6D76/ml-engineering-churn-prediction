"""
Router de Predição — endpoints /predict e /predict-batch.

    
    Recebe o DTO (CustomerFeatures), chama o Service (ModelContainer),
    e retorna (PredictionResponse).

Fluxo de uma predição:
    1. FastAPI valida o JSON com Pydantic 
    2. Depends(get_model) injeta o modelo 
    3. to_raw_dict() converte DTO  formato do dataset
    4. raw_to_ohe() aplica One-Hot Encoding
    5. predictor.predict() normaliza + roda o modelo
    6. Retorna a resposta
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import ModelContainer, get_model
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionResponse,
)
from src.data.preprocessing import raw_to_ohe
from src.utils.logging import get_logger

logger = get_logger(__name__)

# router eh tipo um RestController pra quem eh Javeiro
router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    features: CustomerFeatures,
    model: ModelContainer = Depends(get_model),
):
    """Predição individual recebe 1 cliente, retorna probabilidade de churn.

    """
    if not model.loaded or model.predictor is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Modelo não carregado. Tente novamente mais tarde."},
        )

    # 1. Converter schema Pydantic -> dict formato original do dataset
    raw_dict = features.to_raw_dict()

    # 2. Aplicar OHE (One-Hot Encoding)
    ohe_dict = raw_to_ohe(raw_dict)

    # 3. Predição (normalização + modelo)
    probability, prediction = model.predictor.predict(ohe_dict)

    logger.info(
        "prediction_completed",
        tenure=features.tenure,
        monthly_charges=features.monthly_charges,
        contract=features.contract,
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
    )

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        model_version=model.version,
    )


@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model: ModelContainer = Depends(get_model),
):
    """Predição em lote recebe N clientes, retorna N resultados.

    util quando precisa classificar vários clientes de uma vez
    (ex: campanha de retenção para base toda).
        O processo é o mesmo da predição individual, mas iterando
    """
    if not model.loaded or model.predictor is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Modelo não carregado. Tente novamente mais tarde."},
        )

    results: list[PredictionResponse] = []

    for customer in request.customers:
        raw_dict = customer.to_raw_dict()
        ohe_dict = raw_to_ohe(raw_dict)
        probability, prediction = model.predictor.predict(ohe_dict)

        results.append(
            PredictionResponse(
                churn_probability=round(probability, 4),
                churn_prediction=prediction,
                model_version=model.version,
            )
        )

    logger.info(
        "batch_prediction_completed",
        batch_size=len(results),
        churn_count=sum(1 for r in results if r.churn_prediction),
    )

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
    )
