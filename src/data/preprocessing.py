from __future__ import annotations

import re


class ChurnPreprocessor:
    """Converte dicionários brutos de cliente em features One-Hot Encoded."""

    # Mapeamento de cada coluna categórica para os valores que viram colunas dummy.
    # Manter aqui como atributo de classe em vez de constante global deixa
    # o estado encapsulado e facilita sobrescrever em subclasses se necessário.
    _CATEGORICAL_MAPPINGS: dict[str, list[str]] = {
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

    def transform(self, raw: dict) -> dict[str, float]:
        """Converte um registro bruto para One-Hot Encoding."""
        result: dict[str, float] = {}

        result["SeniorCitizen"] = float(raw.get("SeniorCitizen", raw.get("senior_citizen", 0)))
        result["tenure"] = float(raw.get("tenure", 0))
        result["MonthlyCharges"] = float(raw.get("MonthlyCharges", raw.get("monthly_charges", 0)))
        result["TotalCharges"] = float(raw.get("TotalCharges", raw.get("total_charges", 0)))

        for col, dummy_values in self._CATEGORICAL_MAPPINGS.items():
            snake_key = self._to_snake_case(col)
            value = raw.get(col, raw.get(snake_key, ""))

            for dummy_val in dummy_values:
                ohe_col = f"{col}_{dummy_val}"
                result[ohe_col] = 1.0 if str(value) == dummy_val else 0.0

        return result

    def transform_batch(self, records: list[dict]) -> list[dict[str, float]]:
        """Converte uma lista de registros brutos para OHE."""
        return [self.transform(r) for r in records]

    def validate(self, raw: dict) -> list[str]:
        """Retorna lista de erros de validação para um registro bruto."""
        errors: list[str] = []

        required_numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
        for field in required_numeric:
            snake = self._to_snake_case(field)
            if field not in raw and snake not in raw:
                errors.append(f"Campo obrigatório ausente: {field}")

        return errors

    @staticmethod
    def _to_snake_case(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# Instância padrão reutilizada pelas funções de compatibilidade abaixo.
_default_preprocessor = ChurnPreprocessor()


# Funções mantidas para não quebrar imports existentes — delegam para a classe.
def raw_to_ohe(raw: dict) -> dict[str, float]:
    return _default_preprocessor.transform(raw)


def batch_raw_to_ohe(records: list[dict]) -> list[dict[str, float]]:
    return _default_preprocessor.transform_batch(records)


def validate_raw_input(raw: dict) -> list[str]:
    return _default_preprocessor.validate(raw)
