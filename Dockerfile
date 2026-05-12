# ============================================================================
# Dockerfile — Multi-stage build para a API de Churn Prediction
# ============================================================================
# Stage 1 (builder): instala dependencias e empacota o projeto
# Stage 2 (runtime): imagem minima so com o necessario pra rodar
# ============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Copia so o pyproject.toml primeiro pra aproveitar o cache de camadas do Docker.
# Se as dependencias nao mudaram, o Docker reutiliza essa camada sem reinstalar tudo.
COPY pyproject.toml ./

# Instala dependencias do projeto (sem as de dev)
RUN pip install --no-cache-dir .

# Copia o codigo fonte e artefatos necessarios
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
# Imagem limpa, so com o necessario pra rodar a aplicacao
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="Mateus Valentim <https://www.linkedin.com/in/mateus-valentim-076a33177/>"
LABEL description="Docker image para rodar o projeto de Churn Prediction"
LABEL version="1.0"

# Cria um usuario nao-root por seguranca (nunca rode containers como root em prod)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copia as dependencias instaladas do builder (so os pacotes Python)
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copia o codigo da aplicacao
COPY --from=builder /build/src ./src
COPY --from=builder /build/models ./models
COPY --from=builder /build/data ./data

# Muda para o usuario nao-root
USER appuser

# Porta que o FastAPI vai usar
EXPOSE 8000

# Healthcheck — Docker/ECS usa pra saber se o container ta saudavel
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Inicia a API com uvicorn
# --host 0.0.0.0 permite conexoes externas (necessario dentro do container)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
