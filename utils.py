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
        elif column in numeric_inputs:
            row[column] = 0
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
