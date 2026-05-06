"""
Dependency Injection — carregamento e injeção do modelo.

Analogia Spring:
    Este arquivo é como uma classe @Configuration com métodos @Bean.
    O @lru_cache garante que o modelo é instanciado UMA vez (Singleton).
    O Depends() do FastAPI injeta essa instância nos endpoints,
    igual ao @Autowired no Spring.

curso (Capitulo 04):
    @lru_cache -> garante instancia unica (Singleton pattern)
    Depends()  -> injeta no endpoint automaticamente (DI container)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelContainer:
    """Container que mantém o modelo + metadados carregados.

    obs:
        e como um @Component que encapsula o serviço de ML.
        Contem o predictor (modelo), se está carregado, e métricas.
    """

    def __init__(self):
        self.predictor = None
        self.loaded: bool = False
        self.metrics: dict | None = None
        self.version: str = "0.1.0"
        self._load()

    def _load(self):
        """Carrega o modelo e scaler do disco.

        obs:
            eh como se fosse um @PostConstruct - roda automaticamente na criação do bean.
        """
        try:
            from src.models.mlp import load_model

            self.predictor = load_model()
            self.loaded = True

            # Carregar metricas do checkpoint (se disponível)
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
    """Factory function que retorna o ModelContainer (Singleton).

    obs  analogia spring:
        @Bean
        public ModelContainer modelContainer() {
            return new ModelContainer();  // criado 1 vez cacheado
        }

    O @lru_cache garante que, não importa quantas vezes essa função
    seja chamada, o modelo é carregado apenas UMA vez na memória.

    Nos endpoints usamos assim:
        def predict(model: ModelContainer = Depends(get_model)):
            ...

    equivalente ao:
        @Autowired
        private ModelContainer model;
    """
    logger.info("inicializando_model_container")
    return ModelContainer()
