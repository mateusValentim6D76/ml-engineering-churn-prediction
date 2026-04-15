
#  1. Builder - instala dependências e gera o pacote

FROM python:3.11-slim AS builder

WORKDIR /build

# Copia apenas os arquivos de dependência primeiro (cache de camadas Docker).
# Se as dependências não mudaram, o Docker reutiliza o cache dessa camada.
COPY pyproject.toml ./

# instala dependências do projeto (sem as de dev)
RUN pip install --no-cache-dir --

# copia o coigo fonte
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

#  2. Runtime - imagem final para rodar a aplcacao

#Metadata 
LABEL maintainer="Mateus Valentim <https://www.linkedin.com/in/mateus-valentim-076a33177/>"
LABEL description="Docker image para rodar o projeto de Churn Prediction"
LABEL version="1.0"

# Cria um usuário não root (NAO AOS containers como root)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copia as dependências instaladas do builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia o código da aplicação
COPY --from=builder /build/src ./src
COPY --from=builder /build/models ./models
COPY --from=builder /build/data ./data

# muda o usuário para o appuser
USER appuser

# porta que o Fast API vai rodar
EXPOSE 8080

# Healthcheck — o Docker/ECS usa isso para saber se o container esta saudavel
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando para iniciar a API
# --host 0.0.0.0 permite conexões externas (necessário dentro do container)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
