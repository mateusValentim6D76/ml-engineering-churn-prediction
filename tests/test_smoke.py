class TestImports:

    def test_import_api(self):
        from src.api.main import app  # noqa: F401

    def test_import_schemas(self):
        from src.api.schemas import CustomerFeatures, PredictionResponse  # noqa: F401

    def test_import_logging(self):
        from src.utils.logging import get_logger, setup_logging  # noqa: F401

    def test_import_model(self):
        from src.models.mlp import BasePredictor, ChurnMLP, ChurnPredictor, ModelLoader  # noqa: F401

    def test_import_preprocessing(self):
        from src.data.preprocessing import ChurnPreprocessor, raw_to_ohe  # noqa: F401


class TestAppConfiguration:

    def test_app_has_title(self):
        from src.api.main import app

        assert app.title == "Churn Prediction API"

    def test_app_has_routes(self):
        from src.api.main import app

        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/predict" in routes
