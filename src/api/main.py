from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import health, predict
from src.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

APP_VERSION = "0.1.0"

app = FastAPI(
    title="Churn Prediction API",
    description="API de predição de churn para operadora de telecomunicações",
    version=APP_VERSION,
)

app.include_router(health.router)
app.include_router(predict.router)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    response.headers["X-Process-Time-Ms"] = str(round(duration_ms, 2))
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor. Tente novamente."},
    )


def start_server():
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
