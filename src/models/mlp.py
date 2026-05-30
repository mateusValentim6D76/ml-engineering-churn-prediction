from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# BasePredictor define o contrato que qualquer modelo de predição deve cumprir.
# Usamos Protocol em vez de ABC para não forçar herança: qualquer classe que
# tenha predict() e predict_proba() com as assinaturas corretas já satisfaz
# o contrato. Isso facilita trocar o modelo (XGBoost, LightGBM, etc.) no futuro
# sem mexer no código da API.
@runtime_checkable
class BasePredictor(Protocol):

    def predict_proba(self, features_dict: dict[str, Any]) -> float:
        """Retorna a probabilidade de churn (0.0 a 1.0)."""
        ...

    def predict(self, features_dict: dict[str, Any], threshold: float = 0.5) -> tuple[float, bool]:
        """Retorna (probabilidade, classificação binária)."""
        ...


class ChurnMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ChurnPredictor:
    """Wrapper de inferência — satisfaz o contrato de BasePredictor."""

    def __init__(
        self,
        model: ChurnMLP,
        scaler: StandardScaler,
        feature_names: list[str],
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, features_dict: dict[str, Any]) -> float:
        values = [float(features_dict.get(col, 0)) for col in self.feature_names]

        arr = np.array([values], dtype=np.float32)
        arr_scaled = self.scaler.transform(arr)

        tensor = torch.FloatTensor(arr_scaled).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits).cpu().item()

        return probability

    def predict(self, features_dict: dict[str, Any], threshold: float = 0.5) -> tuple[float, bool]:
        proba = self.predict_proba(features_dict)
        return proba, proba >= threshold


# ModelLoader tem uma única responsabilidade: carregar artefatos do disco.
# Ele também retorna o checkpoint junto com o predictor para que quem chamar
# load() não precise reabrir o arquivo só para ler métricas.
class ModelLoader:

    _DEFAULT_MODEL_FILENAME = "churn_mlp.pt"
    _DEFAULT_SCALER_FILENAME = "scaler.joblib"

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
    ) -> None:
        self._model_path, self._scaler_path = self._resolve_paths(
            model_path, scaler_path
        )

    def _resolve_paths(
        self,
        model_path: str | Path | None,
        scaler_path: str | Path | None,
    ) -> tuple[Path, Path]:
        """Resolve caminhos a partir dos argumentos, env vars ou defaults."""
        project_root = Path(__file__).parent.parent.parent

        resolved_model = Path(model_path) if model_path else (
            project_root / "models" / self._DEFAULT_MODEL_FILENAME
        )
        resolved_scaler = Path(scaler_path) if scaler_path else (
            project_root / "models" / self._DEFAULT_SCALER_FILENAME
        )

        env_model = os.getenv("MODEL_PATH")
        if env_model:
            env_path = Path(env_model)
            resolved_model = (
                env_path / self._DEFAULT_MODEL_FILENAME
                if env_path.is_dir()
                else env_path
            )
            resolved_scaler = env_path / self._DEFAULT_SCALER_FILENAME if env_path.is_dir() else resolved_scaler

        env_scaler = os.getenv("SCALER_PATH")
        if env_scaler:
            resolved_scaler = Path(env_scaler)

        return resolved_model, resolved_scaler

    def load(self) -> tuple[ChurnPredictor, dict]:
        """Carrega o modelo e retorna (predictor, checkpoint)."""
        checkpoint = torch.load(
            self._model_path, map_location="cpu", weights_only=False
        )

        mlp = ChurnMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"],
        )
        mlp.load_state_dict(checkpoint["model_state_dict"])

        scaler = joblib.load(self._scaler_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        predictor = ChurnPredictor(
            model=mlp,
            scaler=scaler,
            feature_names=checkpoint["feature_names"],
            device=device,
        )

        return predictor, checkpoint


# Mantida para compatibilidade com código que já importa load_model diretamente.
def load_model(
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> ChurnPredictor:
    predictor, _ = ModelLoader(model_path, scaler_path).load()
    return predictor
