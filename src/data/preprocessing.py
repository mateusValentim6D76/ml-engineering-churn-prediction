"""
Módulo de pré-processamento para o dataset Telco Customer Churn.

Converte os dados brutos da API (JSON com campos descritivos) para o formato
OHE (One-Hot Encoded) que o modelo espera.

Analogia Java:
    É um Adapter Pattern — converte o DTO da API (CustomerFeatures)
    para o formato interno que o "serviço" (modelo) consome.
    Em Spring, seria como um @Converter ou um MapStruct mapper.
"""

from __future__ import annotations

import pandas as pd


# Colunas categóricas e seus valores possíveis (na ordem do get_dummies com drop_first=True)
# Estas são as colunas OHE que o modelo espera após o pré-processamento da Etapa 1
CATEGORICAL_MAPPINGS: dict[str, list[str]] = {
    "gender": ["Male"],  # drop_first remove "Female"
    "Partner": ["Yes"],
    "Dependents": ["Yes"],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No phone service", "Yes"],
    "InternetService": ["Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "Yes"],
    "OnlineBackup": ["No internet service", "Yes"],
    "DeviceProtection": ["No internet service", "Yes"],
    "TechSupport": ["No internet service", "Yes"],
    "StreamingTV": ["No internet service", "Yes"],
    "StreamingMovies": ["No internet service", "Yes"],
    "Contract": ["One year", "Two year"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Credit card (automatic)", "Electronic check", "Mailed check"],
}


def raw_to_ohe(raw: dict) -> dict[str, float]:
    """Converte um registro JSON bruto para o formato OHE do modelo.

    O modelo foi treinado com pd.get_dummies(drop_first=True) aplicado
    sobre as colunas categóricas. Esta função replica esse processo
    para um único registro.

    Args:
        raw: dicionário com os campos do cliente no formato original.
             Ex: {"gender": "Male", "SeniorCitizen": 0, "tenure": 12, ...}

    Returns:
        Dicionário com as features OHE prontas para o modelo.
        Ex: {"SeniorCitizen": 0, "tenure": 12, "gender_Male": 1, ...}

    """
    result: dict[str, float] = {}

    # Features numricas  passam diretas
    result["SeniorCitizen"] = float(raw.get("SeniorCitizen", raw.get("senior_citizen", 0)))
    result["tenure"] = float(raw.get("tenure", 0))
    result["MonthlyCharges"] = float(raw.get("MonthlyCharges", raw.get("monthly_charges", 0)))
    result["TotalCharges"] = float(raw.get("TotalCharges", raw.get("total_charges", 0)))

    # Features categoricas  aplica OHE manualmente
    for col, dummy_values in CATEGORICAL_MAPPINGS.items():
        # Buscar valor do campo (aceita snake_case e PascalCase)
        snake_key = _to_snake_case(col)
        value = raw.get(col, raw.get(snake_key, ""))

        for dummy_val in dummy_values:
            ohe_col = f"{col}_{dummy_val}"
            result[ohe_col] = 1.0 if str(value) == dummy_val else 0.0

    return result


def _to_snake_case(name: str) -> str:
    """Converte PascalCase/camelCase para snake_case."""
    import re

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def batch_raw_to_ohe(records: list[dict]) -> list[dict[str, float]]:
    """Converte uma lista de registros brutos para OHE.

    Args:
        records: lista de dicionários com dados brutos.

    Returns:
        Lista de dicionários com features OHE.
    """
    return [raw_to_ohe(r) for r in records]


def validate_raw_input(raw: dict) -> list[str]:
    """Valida se o input bruto contém os campos minumos necessarios.

    Returns:
        Lista de erros encontrados (vazia se tudo OK).
    """
    errors: list[str] = []

    # Campos numericos obrigatorios
    required_numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    for field in required_numeric:
        snake = _to_snake_case(field)
        if field not in raw and snake not in raw:
            errors.append(f"Campo obrigatório ausente: {field}")

    return errors
