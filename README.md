# Predictive Credit Risk Modeling Dashboard

An interactive Streamlit dashboard for exploring a credit risk dataset, training a machine learning model, and predicting borrower risk in a Kaggle-style notebook interface.

## Project Structure

```text
credit-risk-dashboard/
|-- app.py
|-- model.py
|-- utils.py
|-- data/
|   |-- credit_risk_dataset.csv
|-- requirements.txt
|-- README.md
```

## Features

- Overview, Data, Visualization, Model, and Prediction tabs
- Custom styled Streamlit interface with hero banner, metric cards, and polished sidebar
- Dataset preview, schema, missing values, and descriptive statistics
- CSV/XLSX upload support from the sidebar with automatic local dataset fallback
- Interactive Plotly charts for key borrower and loan features
- Model comparison across Random Forest, Logistic Regression, and Gradient Boosting
- Accuracy, precision, recall, F1 score, confusion matrix, classification report, and feature importance
- Sidebar borrower form with probability scoring and a risk gauge chart
- Downloadable prediction reports in CSV and JSON formats
- Persistent model save/load support through local artifact files
- SHAP-style local explanation charts based on feature contribution proxies

## Dataset

The app automatically looks for either of these files:

- `data/credit_risk_dataset.csv`
- `data/External_Cibil_Dataset.xlsx`
- `credit_risk_dataset.csv`
- `External_Cibil_Dataset.xlsx`

The expected target column is `loan_status`.
You can also upload a replacement dataset directly in the dashboard sidebar.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- A sample CSV dataset is included in `data/credit_risk_dataset.csv`.
- You can replace it with your own CSV or Excel dataset as long as it includes the expected columns.
- Saved model artifacts are stored in `artifacts/` after you click `Save Selected Model`.
