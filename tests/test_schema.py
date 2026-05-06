"""
Testes de schema — validação dos modelos Pydantic.

Verifica que os schemas aceitam dados válidos e rejeitam dados inválidos.

Analogia Java:
    Similar a testar DTOs com Bean Validation (@Valid, @NotNull, @Min, @Max).
    Se o DTO aceita dados inválidos, o bug vai parar no banco de dados.
    Melhor falhar rápido na validação.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import CustomerFeatures, PredictionResponse

# Payload de exemplo válido (reutilizado em vários testes)
VALID_PAYLOAD = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
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
    "total_charges": 358.20,
}


class TestCustomerFeaturesSchema:
    """Testes do schema de entrada."""

    def test_valid_input(self):
        """Dados válidos devem ser aceitos."""
        features = CustomerFeatures(**VALID_PAYLOAD)
        assert features.gender == "Female"
        assert features.tenure == 12
        assert features.monthly_charges == 29.85

    def test_all_fields_present(self):
        """Deve ter todas as 19 features do dataset."""
        features = CustomerFeatures(**VALID_PAYLOAD)
        raw = features.to_raw_dict()
        assert len(raw) == 19

    def test_negative_tenure_rejected(self):
        """Tenure negativo deve ser rejeitado (ge=0)."""
        payload = {**VALID_PAYLOAD, "tenure": -1}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_invalid_senior_citizen_rejected(self):
        """senior_citizen fora de 0/1 deve ser rejeitado."""
        payload = {**VALID_PAYLOAD, "senior_citizen": 5}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_invalid_gender_rejected(self):
        """Gênero fora de Male/Female deve ser rejeitado."""
        payload = {**VALID_PAYLOAD, "gender": "Other"}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_invalid_contract_rejected(self):
        """Tipo de contrato inválido deve ser rejeitado."""
        payload = {**VALID_PAYLOAD, "contract": "Weekly"}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_missing_required_field_rejected(self):
        """Campos obrigatórios ausentes devem gerar erro."""
        with pytest.raises(ValidationError):
            CustomerFeatures(gender="Male")

    def test_to_raw_dict_conversion(self):
        """to_raw_dict() deve converter snake_case para formato original."""
        features = CustomerFeatures(**VALID_PAYLOAD)
        raw = features.to_raw_dict()
        assert "SeniorCitizen" in raw
        assert "MonthlyCharges" in raw
        assert "PaperlessBilling" in raw
        assert raw["SeniorCitizen"] == 0
        assert raw["MonthlyCharges"] == 29.85


class TestPredictionResponseSchema:
    """Testes do schema de saída."""

    def test_valid_prediction(self):
        """Predição válida deve ser aceita."""
        response = PredictionResponse(
            churn_probability=0.85,
            churn_prediction=True,
            model_version="v0.1.0",
        )
        assert response.churn_probability == 0.85
        assert response.churn_prediction is True

    def test_probability_out_of_range_rejected(self):
        """Probabilidade fora de [0, 1] deve ser rejeitada."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=1.5,
                churn_prediction=True,
                model_version="v0.1.0",
            )
