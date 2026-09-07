
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from features import load_prepared

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RANDOM_STATE = 42

# Human-readable labels for the engineered columns.
PRETTY = {
    "tenure": "months as a customer",
    "MonthlyCharges": "current monthly charge",
    "TotalCharges": "lifetime spend",
    "num_addons": "number of add-on services",
    "avg_monthly_spend": "average monthly spend",
    "charge_delta": "recent change in monthly charge",
    "is_month_to_month": "month-to-month contract",
    "is_electronic_check": "pays by electronic check",
}


def pretty(col: str) -> str:
    if col in PRETTY:
        return PRETTY[col]
    return col.replace("_", " ")


def main():
    bundle = joblib.load(REPORTS / "model.joblib")
    model, threshold = bundle["model"], bundle["threshold"]

    X, y = load_prepared()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    #Global
    importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    importance.head(20).to_csv(REPORTS / "global_importance.csv", index=False)
    print("Top 12 churn drivers (mean |SHAP|):\n")
    for _, row in importance.head(12).iterrows():
        print(f"  {pretty(row['feature']):<38} {row['mean_abs_shap']:.4f}")

    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    #Local
    proba = model.predict_proba(X_test)[:, 1]
    top_idx = np.argsort(proba)[::-1][:5]

    explanations = []
    for rank, i in enumerate(top_idx, start=1):
        contributions = sorted(
            zip(X_test.columns, shap_values[i]),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:4]

        reasons = [
            {
                "feature": pretty(col),
                "value": float(X_test.iloc[i][col]),
                "shap": float(val),
                "direction": "increases risk" if val > 0 else "reduces risk",
            }
            for col, val in contributions
        ]

        explanations.append(
            {
                "rank": rank,
                "churn_probability": float(proba[i]),
                "flagged": bool(proba[i] >= threshold),
                "actually_churned": bool(y_test.iloc[i]),
                "top_reasons": reasons,
            }
        )

    (REPORTS / "local_explanations.json").write_text(json.dumps(explanations, indent=2))
    print("\n\nHighest-risk customers in the test set:\n")
    for e in explanations:
        print(f"  #{e['rank']}  risk {e['churn_probability']:.1%}  "
              f"(actually churned: {e['actually_churned']})")
        for r in e["top_reasons"]:
            print(f"       {r['direction']:<16} {r['feature']} = {r['value']:g}")
        print()

    top_feature = importance.iloc[0]["feature"]
    plt.figure()
    shap.dependence_plot(top_feature, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_dependence_top_feature.png", dpi=130,
                bbox_inches="tight")
    plt.close()

    print(f"Saved: shap_summary.png, shap_dependence_top_feature.png, "
          f"global_importance.csv, local_explanations.json")


if __name__ == "__main__":
    main()
