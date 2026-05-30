from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
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


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_has_latency_header(self):
        response = client.get("/health")
        assert "X-Process-Time-Ms" in response.headers


class TestPredictEndpoint:

    def test_predict_returns_200_or_503(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code in (200, 503)

    def test_predict_response_structure_when_model_loaded(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "model_version" in data
            assert 0 <= data["churn_probability"] <= 1
            assert isinstance(data["churn_prediction"], bool)

    def test_predict_invalid_payload_returns_422(self):
        payload = {"gender": "Female"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_gender_returns_422(self):
        payload = {**VALID_PAYLOAD, "gender": "Invalid"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_tenure_returns_422(self):
        payload = {**VALID_PAYLOAD, "tenure": -5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestPredictBatchEndpoint:

    def test_batch_returns_200_or_503(self):
        response = client.post("/predict-batch", json={"customers": [VALID_PAYLOAD]})
        assert response.status_code in (200, 503)

    def test_batch_response_structure_when_model_loaded(self):
        payload = {"customers": [VALID_PAYLOAD, VALID_PAYLOAD]}
        response = client.post("/predict-batch", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total" in data
            assert data["total"] == 2
            assert len(data["predictions"]) == 2

    def test_batch_empty_list_returns_422(self):
        response = client.post("/predict-batch", json={"customers": []})
        assert response.status_code == 422


class TestPreprocessing:

    RAW_INPUT = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    def test_preprocessor_transform_produces_correct_keys(self):
        from src.data.preprocessing import ChurnPreprocessor

        preprocessor = ChurnPreprocessor()
        ohe = preprocessor.transform(self.RAW_INPUT)

        assert "SeniorCitizen" in ohe
        assert "tenure" in ohe
        assert "MonthlyCharges" in ohe
        assert "TotalCharges" in ohe
        assert "gender_Male" in ohe
        assert ohe["gender_Male"] == 1.0
        assert "Partner_Yes" in ohe
        assert ohe["Partner_Yes"] == 1.0
        assert "Contract_One year" in ohe
        assert ohe["Contract_One year"] == 0.0

    def test_preprocessor_transform_female_gender(self):
        from src.data.preprocessing import ChurnPreprocessor

        preprocessor = ChurnPreprocessor()
        ohe = preprocessor.transform({"gender": "Female", "tenure": 5, "MonthlyCharges": 30, "TotalCharges": 150})
        assert ohe["gender_Male"] == 0.0

    def test_preprocessor_transform_batch(self):
        from src.data.preprocessing import ChurnPreprocessor

        preprocessor = ChurnPreprocessor()
        results = preprocessor.transform_batch([self.RAW_INPUT, self.RAW_INPUT])
        assert len(results) == 2
        assert all("gender_Male" in r for r in results)

    def test_raw_to_ohe_wrapper_compatibility(self):
        """Garante que o wrapper de compatibilidade continua funcionando."""
        from src.data.preprocessing import raw_to_ohe

        ohe = raw_to_ohe(self.RAW_INPUT)
        assert "gender_Male" in ohe
        assert ohe["gender_Male"] == 1.0
