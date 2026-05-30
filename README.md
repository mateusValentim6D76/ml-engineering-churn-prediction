# Churn Prediction - ML Engineering Pipeline

Pipeline end-to-end de Machine Learning para predição de churn (cancelamento) de clientes de uma operadora de telecomunicações. O projeto cobre desde a análise exploratória até o deploy de uma API de inferência com FastAPI.

## Sobre o Projeto

**Dataset:** [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7.043 clientes com 19 features (demográficas, serviços contratados, financeiras) e o target binário `Churn` (Yes/No).

**Modelo:** Rede Neural MLP (Multi-Layer Perceptron) treinada com PyTorch, comparada contra baselines (DummyClassifier e Logistic Regression) rastreados via MLflow.

**Objetivo acadêmico:** Tech Challenge da Pós-Graduação em ML Engineering (FIAP).

## Tech Stack

- **Modelo**: PyTorch MLP + Scikit-Learn baselines
- **API**: FastAPI + Pydantic + Uvicorn
- **Experiment Tracking**: MLflow
- **Testing**: pytest + httpx
- **Linting**: ruff
- **Containerization**: Docker + Docker Compose
- **Infrastructure**: Terraform (AWS ECS Fargate + API Gateway)
- **Logging**: structlog (JSON estruturado)

## Estrutura do Projeto

```
.
├── src/
│   ├── api/
│   │   ├── main.py            # App FastAPI (wiring de routers + middleware)
│   │   ├── dependencies.py    # DI - carregamento do modelo com @lru_cache
│   │   ├── schemas.py         # DTOs Pydantic (entrada/saída da API)
│   │   └── routes/
│   │       ├── health.py      # GET /health
│   │       └── predict.py     # POST /predict + POST /predict-batch
│   ├── data/
│   │   └── preprocessing.py   # Conversão JSON -> One-Hot Encoding
│   ├── models/
│   │   └── mlp.py             # ChurnMLP (nn.Module) + ChurnPredictor
│   └── utils/
│       └── logging.py         # Configuração structlog
├── models/
│   ├── churn_mlp.pt           # Checkpoint do modelo treinado
│   └── scaler.joblib          # StandardScaler ajustado no treino
├── notebooks/
│   ├── 01_eda_baselines.ipynb # Etapa 1: EDA + Baselines + MLflow
│   └── 02_mlp_pytorch.ipynb   # Etapa 2: MLP PyTorch + Cross-Validation
├── tests/
│   ├── test_api.py            # Testes dos endpoints + preprocessing
│   ├── test_schema.py         # Testes de validação Pydantic
│   └── test_smoke.py          # Smoke tests básicos
├── data/raw/                   # Dataset original (.csv)
├── terraform/                  # Infraestrutura como código (AWS)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml              # Dependências e config (ruff, pytest)
└── README.md
```

## Pré-requisitos

- Python 3.11+
- pip (ou pipx)
- Git
- (Opcional) Docker + Docker Compose para containerização
- (Opcional) AWS CLI + Terraform para deploy na nuvem

## 1. Setup do Ambiente

```powershell
# Clonar o repositório
git clone https://github.com/mateusValentim6D76/ml-engineering-churn-prediction
cd ml-engineering-churn-prediction

# Criar virtual environment
python -m venv .venv

# Ativar (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependências (produção + dev)
pip install -e ".[dev]"
```

## 2. Treinar o Modelo

O treinamento é feito nos Jupyter Notebooks, na seguinte ordem:

### Etapa 1 - EDA + Baselines

```powershell
jupyter notebook notebooks/01_eda_baselines.ipynb
```

O que esse notebook faz:
- Carrega o dataset Telco Customer Churn
- Análise exploratória (distribuições, correlações, outliers)
- Pré-processamento (One-Hot Encoding, train/test split 80/20)
- Treina baselines: DummyClassifier (random) e Logistic Regression
- Registra experimentos no MLflow

### Etapa 2 - MLP PyTorch

```powershell
jupyter notebook notebooks/02_mlp_pytorch.ipynb
```

O que esse notebook faz:
- Carrega dados já processados
- Define a arquitetura MLP (128->64->32 neurônios, BatchNorm, Dropout)
- Treina com Early Stopping (BCEWithLogitsLoss + pos_weight para desbalanceamento)
- Validação cruzada estratificada (5-fold)
- Compara MLP vs Baselines
- Salva artefatos em `models/`:
  - `churn_mlp.pt` - checkpoint com state_dict, arquitetura, métricas, feature_names
  - `scaler.joblib` - StandardScaler ajustado nos dados de treino

### Visualizar experimentos no MLflow

```powershell
mlflow ui --port 5000
# Abrir http://localhost:5000 no navegador
```

## 3. Subir a API de Inferência

### Opção A - Localmente com Uvicorn

```powershell
# Certifique-se que os artefatos existem:
#   models/churn_mlp.pt
#   models/scaler.joblib

# Subir a API (hot-reload habilitado)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em:
- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Predição:** POST http://localhost:8000/predict
- **Predição em lote:** POST http://localhost:8000/predict-batch

### Opção B - Com Docker

```powershell
# Build + run
docker-compose up --build

# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000
```

## 4. Testar a API

### Rodar testes automatizados

```powershell
# Todos os testes
pytest tests/ -v

# Apenas testes da API
pytest tests/test_api.py -v

# Com cobertura
pytest tests/ --cov=src --cov-report=term-missing
```

### Testar manualmente (curl ou Swagger)

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predição individual:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 1,
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "phone_service": "No",
    "multiple_lines": "No phone service",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "monthly_charges": 29.85,
    "total_charges": 29.85
  }'
```

**Resposta esperada:**
```json
{
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "model_version": "0.1.0"
}
```

**Predição em lote (batch):**
```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 48,
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)",
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "monthly_charges": 100.50,
        "total_charges": 4824.00
      }
    ]
  }'
```

## 5. Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Health check (status, modelo carregado, métricas) |
| POST | `/predict` | Predição de churn para 1 cliente |
| POST | `/predict-batch` | Predição de churn para N clientes (máx 100) |
| GET | `/docs` | Swagger UI (documentação interativa) |
| GET | `/redoc` | ReDoc (documentação alternativa) |

## 6. Arquitetura da API (Modular)

A API segue o padrão de Dependency Injection do FastAPI (similar ao Spring Boot):

```
Request HTTP
    |
    v
main.py (middleware mede latência)
    |
    v
routes/predict.py (valida entrada com Pydantic)
    |
    +-- Depends(get_model) -> dependencies.py (injeta ModelContainer singleton)
    |
    v
preprocessing.py (OHE) -> models/mlp.py (normaliza + infere)
    |
    v
Response JSON (PredictionResponse)
```

| Arquivo | Responsabilidade |
|---------|-----------------|
| `main.py` | Instancia o app, registra routers e middleware |
| `dependencies.py` | Carrega modelo 1x com `@lru_cache`, injeta via `Depends()` |
| `routes/health.py` | Endpoint `/health` |
| `routes/predict.py` | Endpoints `/predict` e `/predict-batch` |
| `schemas.py` | Validação de entrada/saída (DTOs Pydantic) |
| `models/mlp.py` | Classe do modelo + wrapper de inferência |
| `data/preprocessing.py` | Conversão JSON bruto -> One-Hot Encoding |

## 7. Linting e Formatação

```powershell
# Verificar problemas
ruff check src/ tests/

# Corrigir automaticamente
ruff check src/ tests/ --fix

# Formatar código
ruff format src/ tests/
```

## 8. Deploy na AWS (Terraform)

A infraestrutura usa API Gateway HTTP API + ECS Fargate + Cloud Map para service discovery, substituindo o ALB para redução de custos.

```powershell
cd terraform
terraform init
terraform plan
terraform apply
```

Os outputs exibem a URL do API Gateway e demais recursos criados.

### Fazer o build e push da imagem para o ECR

```powershell
# Login no ECR
docker login --username AWS --password (aws ecr get-login-password --region us-east-1) <ECR_URL>

# Build e push
docker build -t <ECR_URL>:latest .
docker push <ECR_URL>:latest
```

### Destruir a infra

```powershell
terraform destroy -auto-approve
```

## Métricas do Modelo (referência)

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| DummyClassifier | ~0.50 | ~0.36 | 0.50 |
| Logistic Regression | ~0.80 | ~0.57 | ~0.84 |
| **MLP PyTorch** | **~0.80** | **~0.59** | **~0.85** |

*Métricas exatas variam conforme execução. Consulte o MLflow para valores reais.*

## License

MIT
