import pytest
from pydantic import ValidationError

from src.api.schemas import CustomerFeatures, PredictionResponse

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

    def test_valid_input(self):
        features = CustomerFeatures(**VALID_PAYLOAD)
        assert features.gender == "Female"
        assert features.tenure == 12
        assert features.monthly_charges == 29.85

    def test_all_fields_present(self):
        features = CustomerFeatures(**VALID_PAYLOAD)
        raw = features.to_raw_dict()
        assert len(raw) == 19

    def test_negative_tenure_rejected(self):
        payload = {**VALID_PAYLOAD, "tenure": -1}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_invalid_senior_citizen_rejected(self):
        payload = {**VALID_PAYLOAD, "senior_citizen": 5}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_invalid_gender_rejected(self):
        payload = {**VALID_PAYLOAD, "gender": "Other"}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_invalid_contract_rejected(self):
        payload = {**VALID_PAYLOAD, "contract": "Weekly"}
        with pytest.raises(ValidationError):
            CustomerFeatures(**payload)

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(gender="Male")

    def test_to_raw_dict_conversion(self):
        features = CustomerFeatures(**VALID_PAYLOAD)
        raw = features.to_raw_dict()
        assert "SeniorCitizen" in raw
        assert "MonthlyCharges" in raw
        assert "PaperlessBilling" in raw
        assert raw["SeniorCitizen"] == 0
        assert raw["MonthlyCharges"] == 29.85


class TestPredictionResponseSchema:

    def test_valid_prediction(self):
        response = PredictionResponse(
            churn_probability=0.85,
            churn_prediction=True,
            model_version="v0.1.0",
        )
        assert response.churn_probability == 0.85
        assert response.churn_prediction is True

    def test_probability_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=1.5,
                churn_prediction=True,
                model_version="v0.1.0",
            )
