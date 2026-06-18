# Churn Prediction - ML Engineering Pipeline

Pipeline end-to-end de Machine Learning para predição de churn (cancelamento) de clientes de uma operadora de telecomunicações. O projeto cobre desde a análise exploratória até o deploy de uma API de inferência com FastAPI, rodando em ECS Fargate na AWS.

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
- **Containerização**: Docker + Docker Compose
- **Infraestrutura**: Terraform AWS ECS Fargate + API Gateway HTTP API + Cloud Map

## Estrutura do Projeto

```
.
├── src/
│   ├── api/
│   │   ├── main.py            # App FastAPI (routers + middleware)
│   │   ├── dependencies.py    # DI - carregamento do modelo com @lru_cache
│   │   ├── schemas.py         # DTOs Pydantic (entrada/saída da API)
│   │   └── routes/
│   │       ├── health.py      # GET /health
│   │       └── predict.py     # POST /predict + POST /predict-batch
│   ├── data/
│   │   └── preprocessing.py   # ChurnPreprocessor (JSON -> One-Hot Encoding)
│   ├── models/
│   │   ├── mlp.py             # ChurnMLP + ChurnPredictor + ModelLoader
│   │   └── train.py           # Pipeline de treinamento (CLI)
│   └── utils/
│       └── logging.py         # structlog (JSON estruturado)
├── models/
│   ├── churn_mlp.pt           # Checkpoint do modelo treinado
│   └── scaler.joblib          # StandardScaler ajustado no treino
├── notebooks/
│   ├── 01_eda_baselines.ipynb # EDA + Baselines + MLflow
│   └── 02_mlp_pytorch.ipynb   # MLP PyTorch + Cross-Validation
├── tests/
│   ├── test_api.py            # Testes dos endpoints + preprocessing
│   ├── test_schema.py         # Testes de validação Pydantic
│   └── test_smoke.py          # Smoke tests básicos
├── data/raw/                   # Dataset original (.csv)
├── terraform/                  # Infraestrutura como código (AWS)
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── README.md
```

## Pré-requisitos

- Python 3.11+
- Docker + Docker Compose
- (Para deploy na AWS) AWS CLI configurado + Terraform instalado

## Executando Localmente

### 1. Clonar e instalar dependências

```powershell
git clone https://github.com/mateusValentim6D76/ml-engineering-churn-prediction
cd ml-engineering-churn-prediction

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

### 2. Treinar o modelo

O comando abaixo executa o pipeline completo de treinamento e salva os artefatos em `models/`:

```powershell
python -m src.models.train
```

Os artefatos gerados são:
- `models/churn_mlp.pt` - checkpoint com state_dict, arquitetura, métricas e feature_names
- `models/scaler.joblib` - StandardScaler ajustado nos dados de treino

Opções disponíveis:

```powershell
python -m src.models.train --epochs 200 --no-mlflow
python -m src.models.train --model-dir models/
```

Alternativamente, os notebooks cobrem o treinamento com EDA completo:

```powershell
jupyter notebook notebooks/01_eda_baselines.ipynb  # EDA + Baselines
jupyter notebook notebooks/02_mlp_pytorch.ipynb    # MLP PyTorch
```

### 3. Subir a API

**Opção A - Uvicorn direto** (mais rápido para desenvolvimento):

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Opção B - Docker Compose** (API + MLflow):

```powershell
docker compose up --build -d
```

A API estará disponível em:

| Rota | Descrição |
|------|-----------|
| http://localhost:8000/health | Health check |
| http://localhost:8000/predict | Predição individual |
| http://localhost:8000/predict-batch | Predição em lote |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:5000 | MLflow UI (apenas Docker Compose) |

### 4. Testar a API

**Health check:**

```powershell
curl http://localhost:8000/health
```

**Predição individual:**

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 2,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.70,
    "total_charges": 151.65
  }'
```

**Predição em lote:**

```powershell
curl -X POST http://localhost:8000/predict-batch `
  -H "Content-Type: application/json" `
  -d '{
    "customers": [
      {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 2,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.70,
        "total_charges": 151.65
      },
      {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 34,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "One year",
        "paperless_billing": "No",
        "payment_method": "Mailed check",
        "monthly_charges": 56.95,
        "total_charges": 1889.50
      },
      {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "Yes",
        "tenure": 22,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 89.10,
        "total_charges": 1949.40
      }
    ]
  }'
```

### 5. Rodar os testes

```powershell
pytest tests/ -v --tb=short

# Com cobertura
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

### 6. Derrubar os containers

```powershell
docker compose down
```

## Deploy na AWS com Terraform

### Pré-requisitos

- AWS CLI instalado e configurado (`aws configure`) com região `us-east-1`
- Terraform instalado
- Docker instalado e rodando

### 1. Provisionar a infraestrutura

```powershell
cd terraform
terraform init
terraform apply
```

Ao final do `apply`, os outputs mostram os recursos criados:

```
api_url               = "https://<id>.execute-api.us-east-1.amazonaws.com"
ecr_repository_url    = "<account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction"
ecs_cluster_name      = "churn-prediction-cluster"
ecs_service_name      = "churn-prediction-service"
cloudwatch_log_group  = "/ecs/churn-prediction"
```

Guarde o valor de `ecr_repository_url` ele e necessario nos próximos passos.

### 2. Fazer o build e push da imagem para o ECR

> O token do ECR expira em 12 horas. Se receber erro 403 no push, refaça o login.

```powershell
# Login no ECR (use a sintaxe abaixo no PowerShell - o pipe causa erro de encoding)
docker login --username AWS --password (aws ecr get-login-password --region us-east-1) <ECR_URL>

# Build da imagem
docker build -t <ECR_URL>:latest .

# Push para o ECR
docker push <ECR_URL>:latest
```

Substitua `<ECR_URL>` pelo valor de `ecr_repository_url` do output do Terraform.

### 3. Forçar novo deploy no ECS

Após o push da imagem, force o ECS a subir uma nova task:

```powershell
aws ecs update-service `
  --cluster churn-prediction-cluster `
  --service churn-prediction-service `
  --force-new-deployment `
  --region us-east-1
```

### 4. Aguardar o container subir

Aguarde uns2 minutos e verifique se a task está em execução:

```powershell
aws ecs describe-services `
  --cluster churn-prediction-cluster `
  --services churn-prediction-service `
  --region us-east-1 `
  --query "services[0].{running:runningCount,desired:desiredCount,status:status}"
```

O retorno esperado é `"running": 1`.

### 5. Testar na AWS

Use a `api_url` do output do Terraform:

```powershell
curl https://<id>.execute-api.us-east-1.amazonaws.com/health

curl -X POST https://<id>.execute-api.us-east-1.amazonaws.com/predict `
  -H "Content-Type: application/json" `
  -d '{ ... }'
```

### 6. Ver logs em caso de erro

**Via AWS Console:**
1. ECS -> cluster `churn-prediction-cluster`
2. Aba **Tasks** -> filtre por **Stopped** para ver tasks com erro
3. Clique na task -> **View logs in CloudWatch**

**Via CLI:**
```powershell
aws logs tail /ecs/churn-prediction --since 30m --region us-east-1
```

### 7. Destruir a infraestrutura

```powershell
cd terraform
terraform destroy
```

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status da API e do modelo carregado |
| POST | `/predict` | Predição de churn para 1 cliente |
| POST | `/predict-batch` | Predição de churn para N clientes (max 100) |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |

## Metricas do Modelo

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| DummyClassifier | ~0.50 | ~0.36 | 0.50 |
| Logistic Regression | ~0.80 | ~0.57 | ~0.84 |
| **MLP PyTorch** | **~0.80** | **~0.59** | **~0.85** |

*Metricas exatas variam conforme execucao. Consulte o MLflow para valores reais.*

## License

MIT
