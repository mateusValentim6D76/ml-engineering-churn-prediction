from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelContainer:

    def __init__(self):
        self.predictor = None
        self.loaded: bool = False
        self.metrics: dict | None = None
        self.version: str = "0.1.0"
        self._load()

    def _load(self):
        try:
            from src.models.mlp import load_model

            self.predictor = load_model()
            self.loaded = True

            import torch

            checkpoint_path = Path(__file__).parent.parent.parent / "models" / "churn_mlp.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                self.metrics = checkpoint.get("metrics")

            logger.info(
                "modelo_carregado",
                model_loaded=True,
                n_features=len(self.predictor.feature_names),
            )
        except Exception as e:
            logger.error("erro_carregando_modelo", error=str(e))
            self.loaded = False


@lru_cache(maxsize=1)
def get_model() -> ModelContainer:
    logger.info("inicializando_model_container")
    return ModelContainer()
