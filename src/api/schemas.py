"""
Schemas Pydantic  validação de entrada e saída da API.

Pydantic eh um validaro de dados que garante que as requisições e respostas da API sigam um formato definido
Define a estrutura dos dados que a API aceita e retorna,
com validação automática e documentação no Swagger.

Cada campo tem:
    - tipo (str, int, float)
    - validações (ge=0, le=1, Literal[...])
    - descrição para o Swagger
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Schema de entrada  features de um cliente para predição de churn.

    Contém todas as 20 features do dataset Telco Customer Churn (IBM).
    Os nomes usam snake_case na API, mas são convertidos para o formato
    original do dataset internamente.

    """

    # Demográficas
    gender: Literal["Male", "Female"] = Field(
        ..., description="Gênero do cliente"
    )
    senior_citizen: int = Field(
        ..., ge=0, le=1, description="É idoso? (0=Não, 1=Sim)"
    )
    partner: Literal["Yes", "No"] = Field(
        ..., description="Tem parceiro(a)?"
    )
    dependents: Literal["Yes", "No"] = Field(
        ..., description="Tem dependentes?"
    )

    # Conta
    tenure: int = Field(
        ..., ge=0, le=72, description="Meses como cliente (0-72)"
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato"
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ..., description="Fatura digital?"
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Método de pagamento")

    # Serviços de telefonia
    phone_service: Literal["Yes", "No"] = Field(
        ..., description="Tem serviço de telefone?"
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Tem múltiplas linhas?"
    )

    # Serviços de internet
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de serviço de internet"
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tem segurança online?"
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tem backup online?"
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tem proteção de dispositivo?"
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tem suporte técnico?"
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tem streaming de TV?"
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tem streaming de filmes?"
    )

    # Financeiras
    monthly_charges: float = Field(
        ..., ge=0, description="Cobrança mensal ($)"
    )
    total_charges: float = Field(
        ..., ge=0, description="Cobrança total acumulada ($)"
    )

    def to_raw_dict(self) -> dict:
        """Converte o schema Pydantic para o formato original do dataset.

        Analogia Java:
            É como um método toEntity() que converte DTO → Entity.
            Mapeia snake_case → PascalCase/formato original.
        """
        return {
            "gender": self.gender,
            "SeniorCitizen": self.senior_citizen,
            "Partner": self.partner,
            "Dependents": self.dependents,
            "tenure": self.tenure,
            "Contract": self.contract,
            "PaperlessBilling": self.paperless_billing,
            "PaymentMethod": self.payment_method,
            "PhoneService": self.phone_service,
            "MultipleLines": self.multiple_lines,
            "InternetService": self.internet_service,
            "OnlineSecurity": self.online_security,
            "OnlineBackup": self.online_backup,
            "DeviceProtection": self.device_protection,
            "TechSupport": self.tech_support,
            "StreamingTV": self.streaming_tv,
            "StreamingMovies": self.streaming_movies,
            "MonthlyCharges": self.monthly_charges,
            "TotalCharges": self.total_charges,
        }

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                    "total_charges": 29.85,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema de saída  resultado da predição."""

    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probabilidade de churn (0.0 a 1.0)"
    )
    churn_prediction: bool = Field(
        ..., description="Predição: True = vai cancelar"
    )
    model_version: str = Field(
        ..., description="Versão do modelo usado"
    )


class HealthResponse(BaseModel):
    """Schema do endpoint /health."""

    status: str = Field(..., description="Status da API")
    model_loaded: bool = Field(..., description="Modelo carregado com sucesso?")
    version: str = Field(..., description="Versão da aplicação")
    model_metrics: dict | None = Field(
        None, description="Métricas do modelo carregado (se disponível)"
    )


# ── Schemas de Batch ─────────────────────────────────────────────────────────


class BatchPredictionRequest(BaseModel):
    """Schema de entrada para predição em lote.

    Analogia Spring:
        É como receber List<CustomerFeaturesDTO> no @RequestBody.
    """

    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=100,
        description="Lista de clientes para predição (máx 100 por request)"
    )


class BatchPredictionResponse(BaseModel):
    """Schema de saída para predição em lote."""

    predictions: list[PredictionResponse] = Field(
        ..., description="Lista de predições (mesma ordem dos clientes)"
    )
    total: int = Field(..., description="Total de predições realizadas")
