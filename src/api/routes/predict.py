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
from src.data.preprocessing import ChurnPreprocessor
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])

# ChurnPreprocessor não tem estado mutável, então uma única instância
# compartilhada entre requests é suficiente.
_preprocessor = ChurnPreprocessor()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    features: CustomerFeatures,
    model: ModelContainer = Depends(get_model),
):
    if not model.loaded or model.predictor is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Modelo não carregado. Tente novamente mais tarde."},
        )

    ohe_dict = _preprocessor.transform(features.to_raw_dict())
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
    if not model.loaded or model.predictor is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Modelo não carregado. Tente novamente mais tarde."},
        )

    results: list[PredictionResponse] = []

    for customer in request.customers:
        ohe_dict = _preprocessor.transform(customer.to_raw_dict())
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
