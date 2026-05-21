FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./

RUN pip install --no-cache-dir .

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

FROM python:3.11-slim AS runtime

LABEL maintainer="Mateus Valentim <https://www.linkedin.com/in/mateus-valentim-076a33177/>"
LABEL description="Docker image para rodar o projeto de Churn Prediction"
LABEL version="1.0"

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

COPY --from=builder /build/src ./src
COPY --from=builder /build/models ./models
COPY --from=builder /build/data ./data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
