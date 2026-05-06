"""
Aplicação FastAPI — ponto de entrada (Application class).

Analogia Spring:
    Este arquivo é o equivalente ao Application.java com @SpringBootApplication.
    Ele NÃO contém lógica de negócio — apenas:
    1. Cria a instância do app (como SpringApplication.run())
    2. Registra os routers (como @ComponentScan encontra os @Controllers)
    3. Configura middleware (como FilterChain do Spring Security)
    4. Configura exception handlers

    Toda a lógica real está nos routers (controllers) e dependencies (beans).

Estrutura modular:
    main.py          → Application.java (wiring)
    dependencies.py  → @Configuration + @Bean (DI)
    routes/health.py → HealthController.java
    routes/predict.py→ PredictController.java
    schemas.py       → DTOs (Request/Response objects)
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import health, predict
from src.utils.logging import get_logger, setup_logging

# Configura logging estruturado (como logback.xml)
setup_logging()
logger = get_logger(__name__)

APP_VERSION = "0.1.0"

# ── Instancia do App ─────────────────────────────────────────────────────────
# equivale ao SpringApplication.run(Application.class)
app = FastAPI(
    title="Churn Prediction API",
    description="API de predição de churn para operadora de telecomunicações",
    version=APP_VERSION,
)


# ── Registrar Routers ────────────────────────────────────────────────────────
# Equivalente ao ComponentScan q registra cada controller no app
app.include_router(health.router)
app.include_router(predict.router)


# ── Middleware de latência ────────────────────────────────────────────────────
# eqvalnet a um Filter/Interceptor do springao  que mede tempo de resposta
@app.middleware("http")
async def log_latency(request: Request, call_next):
    """Registra latência de cada request no log e no header de resposta."""
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


# ── Exception Handler Global ─────────────────────────────────────────────────
# Equivalente ao ControllerAdvice
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura exceções não tratadas e retorna 500"""
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


# ── Entry Point ──────────────────────────────────────────────────────────────
def start_server():
    """Roda via `churn-prediction serve` (definido no pyproject.toml)."""
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
