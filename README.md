# Churn Prediction — ML Engineering Pipeline

End-to-end Machine Learning Engineering project for telecom customer churn prediction, featuring exploratory data analysis, baseline models, a PyTorch MLP, experiment tracking with MLflow, inference via FastAPI, and production-oriented practices such as testing, containerization, and cloud deployment.

## Tech Stack

- **Model**: PyTorch MLP + Scikit-Learn baselines
- **API**: FastAPI + Pydantic + Uvicorn
- **Experiment Tracking**: MLflow
- **Testing**: pytest + pandera
- **Linting**: ruff
- **Containerization**: Docker + Docker Compose
- **Infrastructure**: Terraform (AWS ECS Fargate)
- **Dataset**: Telco Customer Churn (IBM)

## Project Structure

```
.
├── src/                    # Código-fonte principal
│   ├── api/                # FastAPI (endpoints, schemas, middleware)
│   ├── data/               # Carregamento e pré-processamento de dados
│   ├── models/             # Modelos (MLP PyTorch, baselines)
│   ├── evaluation/         # Métricas e avaliação
│   └── utils/              # Logging estruturado, helpers
├── data/
│   ├── raw/                # Dataset original (não versionado)
│   └── processed/          # Dados pré-processados
├── models/                 # Artefatos de modelos treinados
├── tests/                  # Testes automatizados (pytest)
├── notebooks/              # Jupyter notebooks (EDA, experimentos)
├── docs/                   # Documentação (ML Canvas, Model Card)
├── terraform/              # Infraestrutura como código (AWS)
├── Dockerfile              # Container da aplicação
├── docker-compose.yml      # Orquestração local (API + MLflow)
├── Makefile                # Atalhos de comandos
└── pyproject.toml          # Dependências e configuração
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) AWS CLI + Terraform for cloud deploy

### Local Development

```bash
# 1. Clone o repositório
git clone https://github.com/mateusValentim6D76/ml-engineering-churn-prediction

cd ml-engineering-churn-prediction

# 2. Instale as dependências (recomendado: use um virtualenv)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
make install

# 3. Rode os testes
make test

# 4. Rode o linter
make lint

# 5. Inicie a API localmente
make serve
# Acesse: http://localhost:8000/docs (Swagger UI)
```

### Docker

```bash
# Sobe API + MLflow
make docker-up

# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000

# Para os containers
make docker-down
```

### Cloud Deploy (AWS)

```bash
# 1. Configure o Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edite terraform.tfvars com seus valores

# 2. Inicialize e aplique
make tf-init
make tf-plan
make tf-apply

# 3. Faça deploy da imagem
make deploy ECR_REPO=<url-do-ecr-do-output>
```

## API Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | `/health`  | Health check                       |
| POST   | `/predict` | Predição de churn para um cliente  |
| GET    | `/docs`    | Swagger UI (documentação interativa)|

## Development Commands

Run `make help` to see all available commands.

## License

MIT
