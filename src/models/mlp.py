"""
Módulo do modelo MLP (Multi-Layer Perceptron) para predição de churn.

Contém a definição da arquitetura da rede neural e funções de carregamento.

Analogia Java:
    - ChurnMLP é como uma classe que implementa a interface Predictor
    - load_model() é como um Factory Method que instancia o modelo a partir de um arquivo
    - O state_dict é como serialização/desserialização com ObjectInputStream
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class ChurnMLP(nn.Module):
    """MLP para classificação binária de churn.

    Arquitetura:
        Input → [Linear → BatchNorm → ReLU → Dropout] × N -> Linear -> Output

    obs:
        É como uma classe que herda de AbstractPredictor e implementa forward()
        (equivalente ao método predict()). Cada camada é um "Filter" no pipeline.
    """

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

        # Camada de saída: 1 neurônio (classificação binária)
        # Sem sigmoid  BCEWithLogitsLoss aplica internamente
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — dados entram e saem transformados."""
        return self.network(x).squeeze(-1)


class ChurnPredictor:
    """Wrapper que encapsula modelo + scaler para inferência.

    no java seria tipo:
        um Service que orquestra o fluxo completo:
        dados brutos -> normalização -> modelo -> probabilidade.
        Equivalente a um @Service com @Autowired do Repository (scaler)
        e do Model (MLP).
    """

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
        """Retorna a probabilidade de churn (0.0 a 1.0) para um cliente.

        Args:
            features_dict: dicionário com as features OHE do cliente.

        Returns:
            Probabilidade de churn.
        """
        # Montar array na ordem correta das features
        values = []
        for col in self.feature_names:
            values.append(float(features_dict.get(col, 0)))

        arr = np.array([values], dtype=np.float32)
        arr_scaled = self.scaler.transform(arr)

        tensor = torch.FloatTensor(arr_scaled).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits).cpu().item()

        return probability

    def predict(self, features_dict: dict[str, Any], threshold: float = 0.5) -> tuple[float, bool]:
        """Retorna probabilidade e predição binária.

        Returns:
            (probabilidade, predição_bool)
        """
        proba = self.predict_proba(features_dict)
        return proba, proba >= threshold


def load_model(
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> ChurnPredictor:
    """Carrega o modelo treinado e o scaler do disco.

    no Java:
        Factory Method q cria uma instancia de ChurnPredictor
        a partir dos artefatos serializados.

    Args:
        model_path: caminho para o arquivo .pt (default: models/churn_mlp.pt)
        scaler_path: caminho para o scaler .joblib (default: models/scaler.joblib)

    Returns:
        ChurnPredictor pronto para inferência.
    """
    project_root = Path(__file__).parent.parent.parent

    if model_path is None:
        model_path = project_root / "models" / "churn_mlp.pt"
    if scaler_path is None:
        scaler_path = project_root / "models" / "scaler.joblib"

    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    # Permitir override via variavel de ambiente
    # MODEL_PATH pode ser um diretorio (ex: /app/models) ou arquivo completo
    env_model = os.getenv("MODEL_PATH")
    if env_model:
        env_path = Path(env_model)
        if env_path.is_dir():
            model_path = env_path / "churn_mlp.pt"
            scaler_path = env_path / "scaler.joblib"
        else:
            model_path = env_path

    env_scaler = os.getenv("SCALER_PATH")
    if env_scaler:
        scaler_path = Path(env_scaler)

    # Carregar checkpoint do modelo
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Reconstruir a arquitetura
    mlp = ChurnMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout_rate=checkpoint["dropout_rate"],
    )
    mlp.load_state_dict(checkpoint["model_state_dict"])

    feature_names = checkpoint["feature_names"]

    # Carregar scaler
    scaler = joblib.load(scaler_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ChurnPredictor(
        model=mlp,
        scaler=scaler,
        feature_names=feature_names,
        device=device,
    )
