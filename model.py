from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _resolve_target_column(columns: list[str], requested_target: str) -> str:
    normalized_columns = [column.strip().lower() for column in columns]
    requested_target = requested_target.strip().lower()

    if requested_target in normalized_columns:
        return requested_target

    target_aliases = {
        "loan status",
        "loan_status",
        "status",
        "risk",
        "risk_level",
        "credit_risk",
        "default",
        "default_status",
        "default_risk",
        "default_risk_label",
        "risk_label",
        "class",
        "label",
        "target",
    }
    normalized_aliases = {alias.replace(" ", "_") for alias in target_aliases}

    for column in normalized_columns:
        if column in normalized_aliases:
            return column

    keyword_groups = [
        {"loan", "status"},
        {"default", "risk"},
        {"default", "label"},
        {"risk", "label"},
        {"target"},
        {"class"},
    ]
    for column in normalized_columns:
        tokens = set(column.replace("-", "_").split("_"))
        if any(group.issubset(tokens) for group in keyword_groups):
            return column

    raise ValueError(
        f"Target column '{requested_target}' was not found in the dataset. "
        f"Available columns: {', '.join(normalized_columns)}"
    )


def _normalize_target(series: pd.Series) -> tuple[pd.Series, str]:
    normalized = series.astype(str).str.strip().str.lower()
    risk_keywords = {"high risk", "high_risk", "default", "bad", "1", "yes", "risky"}
    positive_mask = normalized.isin(risk_keywords)
    if positive_mask.any():
        return positive_mask.astype(int), "High Risk"

    unique_values = list(normalized.dropna().unique())
    if len(unique_values) == 2:
        positive_value = sorted(unique_values)[-1]
        positive_label = series.astype(str).loc[normalized == positive_value].mode().iloc[0]
        return (normalized == positive_value).astype(int), str(positive_label)

    return (normalized != normalized.mode().iloc[0]).astype(int), str(series.astype(str).mode().iloc[0])


def _build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    numeric_features = X.select_dtypes(include="number").columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def _flatten_matrix_row(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)[0]


def _evaluate_pipeline(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    positive_label: str,
    feature_columns: list[str],
) -> dict[str, Any]:
    scale_numeric = model_name == "Logistic Regression"
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(X_train, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    feature_importances = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        feature_importances = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(10)
        )

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "feature_importances": feature_importances,
        "positive_label": positive_label,
        "feature_columns": feature_columns,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "target_column": "loan_status",
        "sample_probabilities": y_proba[:5].tolist(),
    }


def train_credit_risk_models(df: pd.DataFrame, target_column: str = "loan_status") -> dict[str, Any]:
    cleaned = df.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]
    target_column = _resolve_target_column(cleaned.columns.tolist(), target_column)

    y, positive_label = _normalize_target(cleaned[target_column])
    X = cleaned.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model_registry = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    model_results: dict[str, Any] = {}
    for model_name, estimator in model_registry.items():
        model_results[model_name] = _evaluate_pipeline(
            model_name,
            estimator,
            X_train,
            X_test,
            y_train,
            y_test,
            positive_label,
            X.columns.tolist(),
        )
        model_results[model_name]["target_column"] = target_column

    comparison_df = pd.DataFrame(
        [
            {
                "model": result["model_name"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
            }
            for result in model_results.values()
        ]
    ).sort_values("accuracy", ascending=False)

    best_model_name = comparison_df.iloc[0]["model"]

    return {
        "models": model_results,
        "comparison": comparison_df.reset_index(drop=True),
        "best_model_name": best_model_name,
    }


def explain_prediction(model_pipeline: Pipeline, input_frame: pd.DataFrame, top_n: int = 12) -> dict[str, Any]:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]

    transformed = preprocessor.transform(input_frame)
    row = _flatten_matrix_row(transformed)
    feature_names = preprocessor.get_feature_names_out()

    method = "Transformed feature magnitude"
    contributions = row.astype(float)

    if hasattr(estimator, "feature_importances_"):
        contributions = estimator.feature_importances_ * row
        method = "SHAP-style proxy using feature importance weighted contributions"
    elif hasattr(estimator, "coef_"):
        coefficients = np.ravel(estimator.coef_)
        contributions = coefficients * row[: len(coefficients)]
        method = "SHAP-style proxy using coefficient weighted contributions"

    explanation_df = pd.DataFrame(
        {
            "feature": [name.replace("num__", "").replace("cat__", "") for name in feature_names],
            "contribution": contributions,
        }
    )
    explanation_df["abs_contribution"] = explanation_df["contribution"].abs()
    explanation_df = explanation_df.sort_values("abs_contribution", ascending=False).head(top_n)

    return {
        "method": method,
        "explanation": explanation_df.reset_index(drop=True),
    }
