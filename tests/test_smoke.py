"""
Smoke tests — verificações básicas de que o sistema "liga".

Smoke tests são os testes mais simples: verificam que os imports funcionam,
que módulos carregam, que a aplicação inicia. Se o smoke test falha,
algo fundamental está quebrado.

Analogia Java:
    É como verificar que o mvn compile funciona antes de rodar os testes.
"""


class TestImports:
    """Verifica que os módulos principais importam sem erro."""

    def test_import_api(self):
        """A API deve importar sem erros."""
        from src.api.main import app  # noqa: F401

    def test_import_schemas(self):
        """Os schemas devem importar sem erros."""
        from src.api.schemas import CustomerFeatures, PredictionResponse  # noqa: F401

    def test_import_logging(self):
        """O módulo de logging deve importar sem erros."""
        from src.utils.logging import get_logger, setup_logging  # noqa: F401

    def test_import_model(self):
        """O módulo do modelo deve importar sem erros."""
        from src.models.mlp import ChurnMLP, ChurnPredictor  # noqa: F401

    def test_import_preprocessing(self):
        """O módulo de pré-processamento deve importar sem erros."""
        from src.data.preprocessing import raw_to_ohe  # noqa: F401


class TestAppConfiguration:
    """Verifica configurações básicas da aplicação."""

    def test_app_has_title(self):
        """A app FastAPI deve ter título configurado."""
        from src.api.main import app

        assert app.title == "Churn Prediction API"

    def test_app_has_routes(self):
        """A app deve ter os endpoints /health e /predict registrados."""
        from src.api.main import app

        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/predict" in routes
