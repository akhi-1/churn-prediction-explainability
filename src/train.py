import json
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import load_prepared

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, preds),
        "brier": brier_score_loss(y_test, proba),
        "threshold": threshold,
    }, proba


def best_f1_threshold(y_true, proba):
    """Churn data is imbalanced, so 0.5 is rarely the right cutoff."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[idx]), float(f1[idx])


def main():
    X, y = load_prepared()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    mlflow.set_experiment("churn-prediction")
    results = []

    # Logistic regression
    with mlflow.start_run(run_name="logistic_regression"):
        lr = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        lr.fit(X_train, y_train)
        metrics, _ = evaluate("logistic_regression", lr, X_test, y_test)
        mlflow.log_params({"model": "logistic_regression", "class_weight": "balanced"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        results.append(metrics)

    # Random forest
    with mlflow.start_run(run_name="random_forest"):
        rf = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(X_train, y_train)
        metrics, _ = evaluate("random_forest", rf, X_test, y_test)
        mlflow.log_params({"model": "random_forest", "n_estimators": 400})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        results.append(metrics)

    # XGBoost with randomised search
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.1],
        "n_estimators": [200, 400, 600],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    with mlflow.start_run(run_name="xgboost_tuned"):
        search = RandomizedSearchCV(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_distributions=param_dist,
            n_iter=30,
            scoring="average_precision",
            cv=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        xgb = search.best_estimator_
        _, proba = evaluate("xgboost_tuned", xgb, X_test, y_test)
        thr, thr_f1 = best_f1_threshold(y_test, proba)
        metrics, _ = evaluate("xgboost_tuned", xgb, X_test, y_test, threshold=thr)

        mlflow.log_params({"model": "xgboost", **search.best_params_})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        mlflow.xgboost.log_model(xgb, name="model")
        results.append(metrics)

    #report
    results.sort(key=lambda r: r["pr_auc"], reverse=True)
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'model':<22}{'ROC AUC':>9}{'PR AUC':>9}{'F1':>8}{'Brier':>8}")
    print("-" * 56)
    for r in results:
        print(
            f"{r['model']:<22}{r['roc_auc']:>9.3f}{r['pr_auc']:>9.3f}"
            f"{r['f1']:>8.3f}{r['brier']:>8.3f}"
        )

    print(f"\nTuned decision threshold: {results[0]['threshold']:.3f}")
    print("\nBest model classification report:")
    best_proba = xgb.predict_proba(X_test)[:, 1]
    print(
        classification_report(
            y_test,
            (best_proba >= results[0]["threshold"]).astype(int),
            target_names=["retained", "churned"],
        )
    )
    import joblib

    joblib.dump(
        {"model": xgb, "threshold": results[0]["threshold"], "columns": list(X.columns)},
        ROOT / "reports" / "model.joblib",
    )


if __name__ == "__main__":
    main()
