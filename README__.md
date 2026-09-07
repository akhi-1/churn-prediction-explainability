# Churn Prediction with Explainability

IBM Telco dataset. 7,043 customers, 26.5% churn.

## Results

xgboost_tuned             0.846    0.665   0.640   0.163
logistic_regression       0.844    0.650   0.620   0.166
random_forest             0.844    0.649   0.643   0.156

Tuned decision threshold: 0.618

## Top features (mean |SHAP|)

Month-to-month contract - 0.6174
Tenure - months as a customer - 0.3950
Fibre optic internet - 0.2742
Two-year contract - 0.1914
Electronic check - pays by electronic check - 0.1721
Current monthly charge - 0.1456

## Stack

Python, scikit-learn, XGBoost, SHAP, MLflow, pandas, pytest

## Run

```bash
pip install -r requirements.txt
make train
make explain
make impact
make test
```

## Structure

```
src/features.py- feature engineering
src/train.py- model training
src/explain.py- SHAP explanations
src/business_impact.py- threshold economics
sql/features.sql- warehouse query
tests/- tests
reports/- output
```

Dataset: [IBM Telco Customer Churn]
