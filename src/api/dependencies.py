from __future__ import annotations

from functools import lru_cache

from src.models.mlp import BasePredictor, ModelLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelContainer:
    """Mantém o estado do modelo carregado e suas métricas."""

    def __init__(self, loader: ModelLoader) -> None:
        self.predictor: BasePredictor | None = None
        self.loaded: bool = False
        self.metrics: dict | None = None
        self.version: str = "0.1.0"
        self._load(loader)

    def _load(self, loader: ModelLoader) -> None:
        try:
            # load() retorna o checkpoint junto com o predictor, então não
            # precisamos abrir o arquivo duas vezes só para ler as métricas.
            predictor, checkpoint = loader.load()

            self.predictor = predictor
            self.metrics = checkpoint.get("metrics")
            self.loaded = True

            logger.info(
                "modelo_carregado",
                model_loaded=True,
                n_features=len(predictor.feature_names),
            )
        except Exception as e:
            logger.error("erro_carregando_modelo", error=str(e))
            self.loaded = False


@lru_cache(maxsize=1)
def get_model() -> ModelContainer:
    """Retorna o singleton do ModelContainer — instanciado apenas uma vez."""
    logger.info("inicializando_model_container")
    return ModelContainer(loader=ModelLoader())
