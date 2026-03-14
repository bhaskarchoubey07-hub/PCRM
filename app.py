from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import explain_prediction, train_credit_risk_models
from utils import (
    build_prediction_frame,
    build_prediction_report,
    dataset_overview,
    find_dataset_path,
    list_model_artifacts,
    load_credit_dataset,
    load_credit_dataset_from_upload,
    load_model_artifact,
    make_prediction,
    prediction_report_downloads,
    save_model_artifact,
)


st.set_page_config(
    page_title="Predictive Credit Risk Modeling",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


NUMERIC_INPUTS = [
    "age",
    "income",
    "employment_length",
    "loan_amount",
    "interest_rate",
    "credit_history_length",
    "number_of_loans",
    "late_payments",
    "debt_to_income_ratio",
]


def apply_custom_styling() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.16), transparent 24%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-shell {
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.2rem;
        }
        .hero-kicker {
            color: #99f6e4;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        .hero-title {
            color: #f8fafc;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.45rem;
        }
        .hero-copy {
            color: #cbd5e1;
            font-size: 1rem;
            max-width: 900px;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(241,245,249,0.95));
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            min-height: 112px;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #475569;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0f172a;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        div[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        div[data-testid="stTabs"] button {
            border-radius: 999px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(dataset_name: str, selected_model_name: str, model_source: str) -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Predictive Analytics Workspace</div>
            <div class="hero-title">Predictive Credit Risk Modeling</div>
            <div class="hero-copy">
                Explore borrower behavior, compare classifiers, save production-ready models, and score
                applicants with a polished Kaggle-style dashboard. Active dataset: <b>{dataset_name}</b>.
                Active model: <b>{selected_model_name}</b>. Source: <b>{model_source}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_local_data():
    path = find_dataset_path()
    df = load_credit_dataset(path)
    return df, path.name


@st.cache_data(show_spinner=False)
def get_uploaded_data(file_name: str, file_bytes: bytes):
    df = load_credit_dataset_from_upload(file_name, file_bytes)
    return df, file_name


@st.cache_resource(show_spinner=True)
def get_model_bundle(df: pd.DataFrame):
    return train_credit_risk_models(df)


@st.cache_data(show_spinner=False)
def get_saved_artifact_options():
    return [str(path) for path in list_model_artifacts()]


def build_distribution_chart(df: pd.DataFrame, column: str, title: str, marginal: str | None = None):
    if column not in df.columns:
        return None
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=title,
        marginal=marginal,
        color_discrete_sequence=["#0f766e"],
    )
    fig.update_layout(plot_bgcolor="rgba(255,255,255,0.9)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_overview_tab(df: pd.DataFrame, overview: dict, selected_model: dict, comparison_df: pd.DataFrame) -> None:
    st.subheader("Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Rows", f"{overview['shape'][0]:,}")
    with col2:
        render_metric_card("Columns", str(overview["shape"][1]))
    with col3:
        render_metric_card("Missing Values", str(int(overview["missing_values"]["missing_values"].sum())))
    with col4:
        render_metric_card("Best Accuracy", f"{comparison_df['accuracy'].max():.2%}")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Target Balance")
        target_counts = (
            df[selected_model["target_column"]]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("loan_status")
            .reset_index(name="count")
        )
        fig = px.pie(
            target_counts,
            names="loan_status",
            values="count",
            hole=0.52,
            color_discrete_sequence=["#0f766e", "#f97316", "#334155", "#14b8a6"],
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Workflow Highlights")
        with st.expander("Reports", expanded=True):
            st.write("Export prediction outcomes as CSV or JSON for downstream review or audit trails.")
        with st.expander("Persistence", expanded=True):
            st.write("Save trained models to disk and reload them from the sidebar when needed.")
        with st.expander("Explainability", expanded=True):
            st.write("Inspect SHAP-style feature contribution proxies for each prediction.")


def render_data_tab(df: pd.DataFrame, overview: dict, dataset_name: str) -> None:
    st.subheader("Dataset Exploration")
    st.caption(f"Current dataset: `{dataset_name}`")
    preview_col, info_col = st.columns([1.4, 1])
    with preview_col:
        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with info_col:
        st.markdown("#### Dataset Shape")
        st.write(f"Rows: `{overview['shape'][0]}`")
        st.write(f"Columns: `{overview['shape'][1]}`")
        st.markdown("#### Column Types")
        st.dataframe(overview["dtypes"], use_container_width=True, height=320)

    stats_col, missing_col = st.columns(2)
    with stats_col:
        st.markdown("#### Basic Statistics")
        st.dataframe(overview["statistics"], use_container_width=True)
    with missing_col:
        st.markdown("#### Missing Values")
        st.dataframe(overview["missing_values"], use_container_width=True)


def render_visualization_tab(df: pd.DataFrame, target_column: str) -> None:
    st.subheader("Interactive Visualization Dashboard")
    numeric_df = df.select_dtypes(include="number")

    chart1, chart2 = st.columns(2)
    with chart1:
        fig = build_distribution_chart(df, "age", "Age Distribution", marginal="box")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with chart2:
        fig = build_distribution_chart(df, "income", "Income Distribution", marginal="violin")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    chart3, chart4 = st.columns(2)
    with chart3:
        fig = build_distribution_chart(df, "loan_amount", "Loan Amount Distribution")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with chart4:
        fig = build_distribution_chart(df, "debt_to_income_ratio", "Debt-to-Income Ratio Histogram")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        target_counts = (
            df[target_column]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("loan_status")
            .reset_index(name="count")
        )
        fig = px.pie(
            target_counts,
            names="loan_status",
            values="count",
            title="Loan Status Distribution",
            color_discrete_sequence=["#0f766e", "#f97316", "#334155", "#14b8a6"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with bottom_right:
        if not numeric_df.empty:
            corr = numeric_df.corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)


def render_model_tab(
    model_results: dict,
    selected_model_name: str,
    selected_model: dict,
    dataset_name: str,
    using_saved_artifact: bool,
) -> None:
    st.subheader("Model Performance")
    comparison_df = model_results["comparison"]

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Selected Accuracy", f"{selected_model['accuracy']:.2%}")
    with top2:
        st.metric("Selected F1", f"{selected_model['f1_score']:.2%}")
    with top3:
        st.metric("Train Rows", selected_model["train_shape"][0])
    with top4:
        st.metric("Best Model", model_results["best_model_name"])

    save_col, info_col = st.columns([0.8, 1.2])
    with save_col:
        if st.button("Save Selected Model", use_container_width=True, disabled=using_saved_artifact):
            saved_path = save_model_artifact(selected_model_name, dataset_name, selected_model)
            st.success(f"Saved model artifact to `{saved_path}`")
            st.cache_data.clear()
    with info_col:
        st.caption("Saved artifacts can be reloaded from the sidebar. Saving is disabled when you are already using a persisted artifact.")

    st.markdown("#### Model Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    comparison_fig = px.bar(
        comparison_df,
        x="model",
        y=["accuracy", "precision", "recall", "f1_score"],
        barmode="group",
        color_discrete_sequence=["#0f766e", "#14b8a6", "#f97316", "#334155"],
        title="Classifier Benchmark",
    )
    comparison_fig.update_layout(height=420)
    st.plotly_chart(comparison_fig, use_container_width=True)

    matrix_col, report_col = st.columns([1, 1.2])
    with matrix_col:
        st.markdown(f"#### {selected_model_name} Confusion Matrix")
        matrix_df = pd.DataFrame(
            selected_model["confusion_matrix"],
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        )
        fig = px.imshow(
            matrix_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with report_col:
        st.markdown(f"#### {selected_model_name} Classification Report")
        st.dataframe(selected_model["classification_report"], use_container_width=True)

    if not selected_model["feature_importances"].empty:
        st.markdown(f"#### {selected_model_name} Top 10 Feature Importances")
        importance_fig = px.bar(
            selected_model["feature_importances"],
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Tealgrn",
        )
        importance_fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(importance_fig, use_container_width=True)
    else:
        st.info(f"{selected_model_name} does not expose tree-based feature importances in this dashboard.")


def render_prediction_tab(
    selected_model_name: str,
    selected_model: dict,
    input_frame: pd.DataFrame,
    dataset_name: str,
) -> None:
    st.subheader("Risk Prediction")
    prediction, probability = make_prediction(
        selected_model["pipeline"],
        input_frame,
        selected_model["positive_label"],
    )

    explanation_result = explain_prediction(selected_model["pipeline"], input_frame)
    explanation_df = explanation_result["explanation"]
    report_df = build_prediction_report(
        input_frame=input_frame,
        prediction_label=prediction,
        probability=probability,
        model_name=selected_model_name,
        dataset_name=dataset_name,
    )
    report_csv, report_json = prediction_report_downloads(
        report_df,
        explanation_df,
        explanation_result["method"],
    )

    risk_label = "High Risk" if prediction == selected_model["positive_label"] else "Low Risk"
    left, middle, right = st.columns([1, 1, 1.2])
    with left:
        st.metric("Prediction", risk_label)
    with middle:
        st.metric("Probability Score", f"{probability:.2%}")
    with right:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": f"{selected_model_name} Risk Gauge"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc2626"},
                    "steps": [
                        {"range": [0, 35], "color": "#bbf7d0"},
                        {"range": [35, 70], "color": "#fde68a"},
                        {"range": [70, 100], "color": "#fecaca"},
                    ],
                },
            )
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download Prediction CSV",
            data=report_csv,
            file_name=f"prediction_report_{selected_model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "Download Prediction JSON",
            data=report_json,
            file_name=f"prediction_report_{selected_model_name.lower().replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )

    explain_col, table_col = st.columns([1.1, 1])
    with explain_col:
        st.markdown("#### SHAP-Style Local Explanation")
        fig = px.bar(
            explanation_df.sort_values("contribution"),
            x="contribution",
            y="feature",
            orientation="h",
            color="contribution",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(explanation_result["method"])
    with table_col:
        st.markdown("#### Explanation Table")
        st.dataframe(explanation_df, use_container_width=True)

    with st.expander("Submitted Applicant Profile", expanded=True):
        st.dataframe(input_frame, use_container_width=True)


def resolve_dataset():
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a credit risk dataset in CSV or Excel format.",
    )

    if uploaded_file is not None:
        df, dataset_name = get_uploaded_data(uploaded_file.name, uploaded_file.getvalue())
        return df, dataset_name, "Uploaded file"

    df, dataset_name = get_local_data()
    return df, dataset_name, "Local fallback"


def resolve_selected_model(model_results: dict):
    artifact_options = get_saved_artifact_options()
    model_source = st.sidebar.radio("Model source", ["Current training", "Saved artifact"], index=0)

    if model_source == "Saved artifact" and artifact_options:
        selected_artifact = st.sidebar.selectbox(
            "Saved model artifact",
            options=artifact_options,
            format_func=lambda value: Path(value).name,
        )
        artifact_payload = load_model_artifact(selected_artifact)
        model_name = artifact_payload["model_name"]
        model_bundle = artifact_payload["model_bundle"]
        st.sidebar.caption(f"Artifact saved at: `{artifact_payload['saved_at']}`")
        return model_name, model_bundle, "Saved artifact", True

    if model_source == "Saved artifact" and not artifact_options:
        st.sidebar.info("No saved model artifacts found yet. Using current training instead.")

    model_name = st.sidebar.selectbox(
        "Prediction model",
        options=list(model_results["models"].keys()),
        index=list(model_results["models"].keys()).index(model_results["best_model_name"]),
    )
    return model_name, model_results["models"][model_name], "Current training", False


def main() -> None:
    apply_custom_styling()
    st.sidebar.header("Dashboard Controls")

    try:
        df, dataset_name, dataset_source = resolve_dataset()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    overview = dataset_overview(df)
    model_results = get_model_bundle(df)
    selected_model_name, selected_model, model_source_label, using_saved_artifact = resolve_selected_model(model_results)

    render_hero(dataset_name, selected_model_name, model_source_label)

    st.sidebar.caption(f"Dataset source: `{dataset_source}`")
    st.sidebar.caption(f"Loaded dataset: `{dataset_name}`")

    st.sidebar.markdown("### Borrower Inputs")
    with st.sidebar.form("prediction_form"):
        input_values = {
            "age": st.number_input("Age", min_value=18, max_value=100, value=35),
            "income": st.number_input("Income", min_value=0.0, value=65000.0, step=1000.0),
            "employment_length": st.number_input("Employment Length", min_value=0.0, value=5.0, step=1.0),
            "loan_amount": st.number_input("Loan Amount", min_value=0.0, value=15000.0, step=500.0),
            "interest_rate": st.number_input("Interest Rate", min_value=0.0, value=11.5, step=0.1),
            "credit_history_length": st.number_input("Credit History Length", min_value=0.0, value=8.0, step=1.0),
            "number_of_loans": st.number_input("Number of Loans", min_value=0, value=3, step=1),
            "late_payments": st.number_input("Late Payments", min_value=0, value=1, step=1),
            "debt_to_income_ratio": st.number_input(
                "Debt-to-Income Ratio",
                min_value=0.0,
                max_value=1.5,
                value=0.32,
                step=0.01,
            ),
        }
        predict_clicked = st.form_submit_button("Predict Risk")

    tabs = st.tabs(["Overview", "Data", "Visualization", "Model", "Prediction"])

    with tabs[0]:
        render_overview_tab(df, overview, selected_model, model_results["comparison"])
    with tabs[1]:
        render_data_tab(df, overview, dataset_name)
    with tabs[2]:
        render_visualization_tab(df, selected_model["target_column"])
    with tabs[3]:
        render_model_tab(
            model_results,
            selected_model_name,
            selected_model,
            dataset_name,
            using_saved_artifact,
        )
    with tabs[4]:
        if predict_clicked:
            input_frame = build_prediction_frame(
                input_values,
                selected_model["feature_columns"],
                NUMERIC_INPUTS,
            )
            render_prediction_tab(selected_model_name, selected_model, input_frame, dataset_name)
        else:
            st.info("Use the sidebar form and click `Predict Risk` to score a borrower profile.")


if __name__ == "__main__":
    main()
