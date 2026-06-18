"""
Treino do modelo de churn (MLP PyTorch).

Este mdulo extrai a logica de treino que antes vivia apenas no notebook
(notebooks/02_mlp_pytorch.ipynb) para um script versionavel e reproduzivel,
executavel via ``python -m src.models.train`` ou ``make train``.

Cada etapa do pipeline tem uma responsabilidade nica (carregar dados, dividir,
normalizar, treinar, avaliar, persistir), seguindo a separação de
responsabilidades e os contratos de entrada ousaida discutidos nas aulas de
padronização de pipelines de ML. A arquitetura ChurnMLP é reutilizada de
src.models.mlp não ha duplicação da definição do modelo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import ChurnMLP
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

#Constantes de reprodutibilidade e caminhos
RANDOM_STATE = 42
TARGET_COLUMN = "Churn"

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv"
DEFAULT_RAW_PATH = _PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models"
DEFAULT_MLRUNS_DIR = _PROJECT_ROOT / "mlruns"

# ── Hiperparâmetros padrão (espelham o modelo atualmente em produção) ─────────
DEFAULT_HIDDEN_DIMS = [256, 128, 64]
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 15
DATASET_VERSION = "v1.0-telco-ibm"


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fixa as seeds para tornar o treino reproduzível."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Interrompe o treino quando a loss de validação para de melhorar."""

    def __init__(self, patience: int = DEFAULT_PATIENCE, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.best_model_state: dict | None = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        """Restaura os pesos da melhor época observada."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def load_dataset(processed_path: Path, raw_path: Path) -> pd.DataFrame:
    """
    Carrega o dataset já com One-Hot Encoding.

    Se o CSV processado não existir, reconstrói a partir do bruto aplicando o
    mesmo pré-processamento da etapa exploratória (mesma lógica do notebook),
    garantindo consistência entre treino e o que a API espera.
    """
    if processed_path.exists():
        logger.info("dataset_carregado", source=str(processed_path))
        return pd.read_csv(processed_path)

    logger.info("processado_ausente_reconstruindo", raw=str(raw_path))
    df = pd.read_csv(raw_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)

    categorical = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(processed_path, index=False)
    logger.info("processado_salvo", path=str(processed_path))
    return df_encoded


def split_data(df: pd.DataFrame):
    """
    Divide em treino/validação/teste de forma estratificada.

    O conjunto de teste é idêntico ao usado na etapa de baselines (mesma seed +
    stratify); a validação é extraída do treino para alimentar o early stopping.
    """
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    x_train_full, x_val, y_train_full, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    return {
        "feature_names": features.columns.tolist(),
        "x_train": x_train_full,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train_full,
        "y_val": y_val,
        "y_test": y_test,
    }


def scale_features(x_train, x_val, x_test):
    """Normaliza com StandardScaler ajustado apenas no treino (evita leakage)."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return scaler, x_train_scaled, x_val_scaled, x_test_scaled


def build_loaders(x_train_scaled, y_train, x_val_scaled, y_val, batch_size: int):
    """Empacota treino e validação em DataLoaders do PyTorch."""
    train_ds = TensorDataset(
        torch.FloatTensor(x_train_scaled), torch.FloatTensor(y_train.values)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(x_val_scaled), torch.FloatTensor(y_val.values)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_pos_weight(y_train, device: torch.device) -> torch.Tensor:
    """Calcula o peso da classe minoritária para compensar o desbalanceamento."""
    n_positive = float(y_train.sum())
    n_negative = len(y_train) - n_positive
    return torch.tensor([n_negative / n_positive], dtype=torch.float32, device=device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Treina o modelo com early stopping e retorna o histórico de loss."""
    early_stopping = EarlyStopping(patience=patience)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(model(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_losses.append(criterion(model(x_batch), y_batch).item())

        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            logger.info(
                "epoch_completed",
                epoch=epoch + 1,
                max_epochs=n_epochs,
                train_loss=round(avg_train_loss, 4),
                val_loss=round(avg_val_loss, 4),
                patience=f"{early_stopping.counter}/{patience}",
            )

        early_stopping(avg_val_loss, model)
        if early_stopping.should_stop:
            logger.info(
                "early_stopping",
                epoch=epoch + 1,
                best_val_loss=round(early_stopping.best_loss, 4),
            )
            break

    early_stopping.restore_best(model)
    return history


def evaluate(model: nn.Module, x_test_scaled, y_test, device: torch.device) -> dict[str, float]:
    """Avalia o modelo no conjunto de teste e retorna as métricas."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(x_test_scaled).to(device))
        proba = torch.sigmoid(logits).cpu().numpy()
    preds = (proba >= 0.5).astype(int)
    y_true = y_test.values

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def save_artifacts(
    model: nn.Module,
    scaler: StandardScaler,
    feature_names: list[str],
    hidden_dims: list[int],
    dropout_rate: float,
    metrics: dict[str, float],
    model_dir: Path,
) -> tuple[Path, Path]:
    """
    Persiste o checkpoint do modelo e o scaler.

    O formato do checkpoint é exatamente o esperado por
    src.models.mlp.ModelLoader (input_dim, hidden_dims, dropout_rate,
    model_state_dict, feature_names, metrics), garantindo compatibilidade
    direta com a API de inferência.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "churn_mlp.pt"
    scaler_path = model_dir / "scaler.joblib"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "feature_names": feature_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "metrics": metrics,
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def _log_to_mlflow(params: dict, metrics: dict, history: dict, artifacts: list[Path]) -> None:
    """Registra a execução no MLflow (best-effort: nunca quebra o treino)."""
    try:
        import mlflow

        DEFAULT_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(DEFAULT_MLRUNS_DIR.as_uri())
        mlflow.set_experiment("churn_prediction")
        with mlflow.start_run(run_name="mlp_pytorch_train"):
            mlflow.log_params(params)
            paired = zip(history["train_loss"], history["val_loss"], strict=False)
            for epoch, (tl, vl) in enumerate(paired):
                mlflow.log_metric("train_loss", tl, step=epoch)
                mlflow.log_metric("val_loss", vl, step=epoch)
            mlflow.log_metric("n_epochs_trained", len(history["train_loss"]))
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
        logger.info("mlflow_run_registrado")
    except Exception as exc:  # noqa: BLE001 - tracking é opcional, não deve abortar o treino
        logger.warning("mlflow_indisponivel", error=str(exc))


def run_training(
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    hidden_dims: list[int] | None = None,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    track_mlflow: bool = True,
) -> dict[str, float]:
    """Orquestra o pipeline completo de treino e retorna as métricas de teste."""
    hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("treino_iniciado", device=str(device), hidden_dims=hidden_dims)

    df = load_dataset(processed_path, raw_path)
    data = split_data(df)
    logger.info(
        "dados_divididos",
        treino=len(data["y_train"]),
        validacao=len(data["y_val"]),
        teste=len(data["y_test"]),
        features=len(data["feature_names"]),
    )

    scaler, x_train_scaled, x_val_scaled, x_test_scaled = scale_features(
        data["x_train"], data["x_val"], data["x_test"]
    )
    train_loader, val_loader = build_loaders(
        x_train_scaled, data["y_train"], x_val_scaled, data["y_val"], batch_size
    )

    set_seed(RANDOM_STATE)
    model = ChurnMLP(
        input_dim=len(data["feature_names"]),
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    ).to(device)

    pos_weight = compute_pos_weight(data["y_train"], device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("treinando", pos_weight=round(pos_weight.item(), 2))
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        n_epochs=max_epochs, patience=patience, device=device,
    )

    metrics = evaluate(model, x_test_scaled, data["y_test"], device)
    logger.info("avaliacao_concluida", **{k: round(v, 4) for k, v in metrics.items()})

    model_path, scaler_path = save_artifacts(
        model, scaler, data["feature_names"], hidden_dims, dropout_rate, metrics, model_dir,
    )
    logger.info("artefatos_salvos", model=str(model_path), scaler=str(scaler_path))

    if track_mlflow:
        params = {
            "model_type": "MLP_PyTorch",
            "hidden_dims": str(hidden_dims),
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            "pos_weight": round(pos_weight.item(), 2),
            "n_features": len(data["feature_names"]),
            "random_state": RANDOM_STATE,
            "dataset_version": DATASET_VERSION,
        }
        _log_to_mlflow(params, metrics, history, [model_path, scaler_path])

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de churn (MLP PyTorch).")
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=",".join(str(d) for d in DEFAULT_HIDDEN_DIMS),
        help="Camadas ocultas separadas por vírgula (ex.: 256,128,64).",
    )
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Desativa o tracking no MLflow."
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    hidden_dims = [int(d) for d in args.hidden_dims.split(",") if d.strip()]

    metrics = run_training(
        processed_path=args.processed_path,
        raw_path=args.raw_path,
        model_dir=args.model_dir,
        hidden_dims=hidden_dims,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        track_mlflow=not args.no_mlflow,
    )

    print("\n" + "=" * 52)
    print("  MLP PyTorch — métricas no conjunto de teste")
    print("=" * 52)
    for name, value in metrics.items():
        print(f"  {name:>12s}: {value:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
