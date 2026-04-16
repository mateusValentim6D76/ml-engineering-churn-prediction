# ============================================================================
# Makefile — Atalhos para comandos do projeto
# ============================================================================
# Uso: make <comando>
#   make install     → instala dependências
#   make lint        → verifica estilo do código
#   make test        → roda testes
#   make docker-up   → sobe containers (API + MLflow)
#   make deploy      → faz build e push da imagem para ECR
# ============================================================================

.PHONY: help install lint format test docker-build docker-up docker-down \
        serve train deploy tf-init tf-plan tf-apply clean

# Comando padrão quando roda só "make" sem argumentos
help:
	@echo ""
	@echo "  Churn Prediction — Comandos disponíveis"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install       Instala dependências (dev)"
	@echo "  make lint          Verifica estilo do código (ruff)"
	@echo "  make format        Formata o código automaticamente"
	@echo "  make test          Roda testes (pytest)"
	@echo "  make serve         Inicia a API localmente"
	@echo "  make train         Treina o modelo"
	@echo "  make docker-build  Build da imagem Docker"
	@echo "  make docker-up     Sobe containers (API + MLflow)"
	@echo "  make docker-down   Para os containers"
	@echo "  make tf-init       Inicializa o Terraform"
	@echo "  make tf-plan       Mostra plano de mudanças na infra"
	@echo "  make tf-apply      Aplica as mudanças na AWS"
	@echo "  make deploy        Build + push imagem para ECR"
	@echo "  make clean         Remove artefatos temporários"
	@echo ""

# ── Python ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

train:
	python -m src.models.train

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t churn-prediction:latest .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ── Terraform ─────────────────────────────────────────────────────────────────

tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan

tf-apply:
	cd terraform && terraform apply

tf-destroy:
	cd terraform && terraform destroy

# ── Deploy (build + push para ECR) ────────────────────────────────────────────

# Uso: make deploy ECR_REPO=<url-do-ecr>
# A URL do ECR é exibida no output do terraform apply
deploy:
	@echo "Fazendo login no ECR..."
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(ECR_REPO)
	@echo "Building imagem..."
	docker build -t $(ECR_REPO):latest .
	@echo "Pushing para ECR..."
	docker push $(ECR_REPO):latest
	@echo "Deploy completo!"

# ── Limpeza ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info
