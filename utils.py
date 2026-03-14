from __future__ import annotations

import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd


DATASET_CANDIDATES = [
    Path("data/credit_risk_dataset.csv"),
    Path("data/External_Cibil_Dataset.xlsx"),
    Path("credit_risk_dataset.csv"),
    Path("External_Cibil_Dataset.xlsx"),
]
ARTIFACTS_DIR = Path("artifacts")
FEATURE_ALIASES = {
    "income_inr": "income",
    "loan_amount_inr": "loan_amount",
    "existing_debt_inr": "debt_to_income_ratio",
    "credit_history_years": "credit_history_length",
    "credit_history_length": "credit_history_length",
    "interest_rate_pct": "interest_rate",
    "num_loans": "number_of_loans",
    "loan_count": "number_of_loans",
    "late_payment_count": "late_payments",
}


def find_dataset_path() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No dataset file was found. Add `credit_risk_dataset.csv` or `External_Cibil_Dataset.xlsx` "
        "to the project root or the `data/` folder."
    )


def load_credit_dataset(path: Path) -> pd.DataFrame:
    return _read_dataset(path.suffix.lower(), path)


def load_credit_dataset_from_upload(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    buffer = BytesIO(file_bytes)
    return _read_dataset(suffix, buffer)


def _read_dataset(suffix: str, source) -> pd.DataFrame:
    if suffix == ".csv":
        df = pd.read_csv(source)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(source)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    df.columns = [column.strip().lower() for column in df.columns]
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values}),
        "missing_values": pd.DataFrame(
            {
                "column": df.columns,
                "missing_values": df.isna().sum().values,
                "missing_percentage": (df.isna().mean() * 100).round(2).values,
            }
        ).sort_values("missing_values", ascending=False),
        "statistics": df.describe(include="all").transpose().fillna(""),
    }


def build_prediction_frame(
    user_inputs: dict,
    feature_columns: list[str],
    numeric_inputs: list[str],
) -> pd.DataFrame:
    row = {}
    for column in feature_columns:
        if column in user_inputs:
            row[column] = user_inputs[column]
        else:
            mapped_input = _resolve_feature_value(column, user_inputs)
            if mapped_input is not None:
                row[column] = mapped_input
            elif column in numeric_inputs or _looks_numeric(column):
                row[column] = _default_numeric_value(column)
            else:
                row[column] = "Unknown"
    return pd.DataFrame([row])


def make_prediction(model_pipeline, input_frame: pd.DataFrame, positive_label: str) -> tuple[str, float]:
    prediction_code = model_pipeline.predict(input_frame)[0]
    probability = model_pipeline.predict_proba(input_frame)[0][1]
    prediction = positive_label if prediction_code == 1 else "Low Risk"
    return prediction, float(probability)


def save_model_artifact(model_name: str, dataset_name: str, model_bundle: dict) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = _slugify(model_name)
    safe_dataset = _slugify(dataset_name)
    path = ARTIFACTS_DIR / f"{safe_model}_{safe_dataset}_{timestamp}.joblib"

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "model_bundle": model_bundle,
    }
    joblib.dump(payload, path)
    return path


def list_model_artifacts() -> list[Path]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(ARTIFACTS_DIR.glob("*.joblib"), reverse=True)


def load_model_artifact(path: str | Path) -> dict:
    return joblib.load(path)


def build_prediction_report(
    input_frame: pd.DataFrame,
    prediction_label: str,
    probability: float,
    model_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    report_df = input_frame.copy()
    report_df.insert(0, "generated_at", datetime.now().isoformat(timespec="seconds"))
    report_df.insert(1, "dataset_name", dataset_name)
    report_df.insert(2, "model_name", model_name)
    report_df["prediction"] = prediction_label
    report_df["probability_score"] = round(probability, 6)
    return report_df


def prediction_report_downloads(
    report_df: pd.DataFrame,
    explanation_df: pd.DataFrame,
    method: str,
) -> tuple[bytes, str]:
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    payload = {
        "report": report_df.to_dict(orient="records"),
        "explanation_method": method,
        "top_explanations": explanation_df.to_dict(orient="records"),
    }
    return csv_bytes, json.dumps(payload, indent=2)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _resolve_feature_value(column: str, user_inputs: dict):
    normalized_column = column.strip().lower()

    if normalized_column in FEATURE_ALIASES and FEATURE_ALIASES[normalized_column] in user_inputs:
        base_value = user_inputs[FEATURE_ALIASES[normalized_column]]
        if normalized_column == "existing_debt_inr":
            income = float(user_inputs.get("income", 0) or 0)
            ratio = float(user_inputs.get("debt_to_income_ratio", 0) or 0)
            return income * ratio
        return base_value

    tokens = set(normalized_column.replace("-", "_").split("_"))
    token_aliases = {
        "income": "income",
        "amount": "loan_amount",
        "loan": "loan_amount",
        "rate": "interest_rate",
        "interest": "interest_rate",
        "age": "age",
        "score": None,
        "debt": "debt_to_income_ratio",
        "ratio": "debt_to_income_ratio",
        "history": "credit_history_length",
        "length": "credit_history_length",
        "loans": "number_of_loans",
        "payments": "late_payments",
        "employment": "employment_length",
    }

    for token, input_name in token_aliases.items():
        if token in tokens and input_name in user_inputs:
            return user_inputs[input_name]

    return None


def _looks_numeric(column: str) -> bool:
    numeric_tokens = {
        "age",
        "income",
        "amount",
        "score",
        "debt",
        "ratio",
        "length",
        "history",
        "loans",
        "payments",
        "id",
        "rate",
        "count",
        "number",
        "inr",
        "pct",
    }
    tokens = set(column.strip().lower().replace("-", "_").split("_"))
    return bool(tokens & numeric_tokens)


def _default_numeric_value(column: str):
    normalized_column = column.strip().lower()
    if "score" in normalized_column:
        return 700.0
    if "age" in normalized_column:
        return 35.0
    if "rate" in normalized_column or "ratio" in normalized_column:
        return 0.0
    return 0.0
