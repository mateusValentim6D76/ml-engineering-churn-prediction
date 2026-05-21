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
        proba = self.predict_proba(features_dict)
        return proba, proba >= threshold


def load_model(
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> ChurnPredictor:
    project_root = Path(__file__).parent.parent.parent

    if model_path is None:
        model_path = project_root / "models" / "churn_mlp.pt"
    if scaler_path is None:
        scaler_path = project_root / "models" / "scaler.joblib"

    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

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

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    mlp = ChurnMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout_rate=checkpoint["dropout_rate"],
    )
    mlp.load_state_dict(checkpoint["model_state_dict"])

    feature_names = checkpoint["feature_names"]

    scaler = joblib.load(scaler_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ChurnPredictor(
        model=mlp,
        scaler=scaler,
        feature_names=feature_names,
        device=device,
    )
