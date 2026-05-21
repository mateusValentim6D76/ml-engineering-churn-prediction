from __future__ import annotations

import pandas as pd


CATEGORICAL_MAPPINGS: dict[str, list[str]] = {
    "gender": ["Male"],
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
    result: dict[str, float] = {}

    result["SeniorCitizen"] = float(raw.get("SeniorCitizen", raw.get("senior_citizen", 0)))
    result["tenure"] = float(raw.get("tenure", 0))
    result["MonthlyCharges"] = float(raw.get("MonthlyCharges", raw.get("monthly_charges", 0)))
    result["TotalCharges"] = float(raw.get("TotalCharges", raw.get("total_charges", 0)))

    for col, dummy_values in CATEGORICAL_MAPPINGS.items():
        snake_key = _to_snake_case(col)
        value = raw.get(col, raw.get(snake_key, ""))

        for dummy_val in dummy_values:
            ohe_col = f"{col}_{dummy_val}"
            result[ohe_col] = 1.0 if str(value) == dummy_val else 0.0

    return result


def _to_snake_case(name: str) -> str:
    import re

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def batch_raw_to_ohe(records: list[dict]) -> list[dict[str, float]]:
    return [raw_to_ohe(r) for r in records]


def validate_raw_input(raw: dict) -> list[str]:
    errors: list[str] = []

    required_numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    for field in required_numeric:
        snake = _to_snake_case(field)
        if field not in raw and snake not in raw:
            errors.append(f"Campo obrigatório ausente: {field}")

    return errors
