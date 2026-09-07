import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from features import load_prepared

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RANDOM_STATE = 42

OFFER_COST = 40.0        
CUSTOMER_VALUE = 500.0   
OFFER_SUCCESS = 0.30     


def main():
    bundle = joblib.load(REPORTS / "model.joblib")
    model = bundle["model"]

    X, y = load_prepared()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    proba = model.predict_proba(X_test)[:, 1]

    rows = []
    for thr in np.arange(0.10, 0.95, 0.05):
        preds = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        contacted = tp + fp
        saved = tp * OFFER_SUCCESS

        revenue = saved * CUSTOMER_VALUE
        cost = contacted * OFFER_COST
        net = revenue - cost

        rows.append(
            {
                "threshold": round(float(thr), 2),
                "contacted": int(contacted),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "missed_churners": int(fn),
                "expected_saved": round(saved, 1),
                "campaign_cost": round(cost, 0),
                "net_value": round(net, 0),
            }
        )

    df = pd.DataFrame(rows)
    best = df.loc[df["net_value"].idxmax()]

    df.to_csv(REPORTS / "threshold_economics.csv", index=False)
    (REPORTS / "best_threshold.json").write_text(
        json.dumps(
            {
                "assumptions": {
                    "offer_cost": OFFER_COST,
                    "customer_value": CUSTOMER_VALUE,
                    "offer_success_rate": OFFER_SUCCESS,
                },
                "best_threshold": float(best["threshold"]),
                "net_value_on_test_set": float(best["net_value"]),
                "customers_contacted": int(best["contacted"]),
                "churners_missed": int(best["missed_churners"]),
            },
            indent=2,
        )
    )

    print(f"Assumptions: offer ${OFFER_COST:.0f}, customer worth "
          f"${CUSTOMER_VALUE:.0f}, offer converts {OFFER_SUCCESS:.0%}\n")
    print(df.to_string(index=False))
    print(
        f"\nBest net value at threshold {best['threshold']:.2f}: "
        f"${best['net_value']:,.0f} on a test set of {len(y_test)} customers "
        f"({int(best['contacted'])} contacted, {int(best['missed_churners'])} churners missed)."
    )


if __name__ == "__main__":
    main()
