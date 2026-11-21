"""
Model search and benchmarking suite for the Freight Rate Prediction Challenge.

Evaluates multiple model families using:
1. 3-fold chronological TimeSeriesSplit on development data (Jan-Sep 2025).
2. Out-of-sample holdout validation on October 2025.
Outputs comparison rankings and metrics to model_search_results.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

from train_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features, clean_data


def preprocessor() -> ColumnTransformer:
    """Standard preprocessor with median numeric imputation and unknown-safe ordinal encoding."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models() -> list[tuple[str, any]]:
    """Define candidate regression models spanning diverse algorithm families."""
    return [
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.03,
                max_iter=300,
                max_leaf_nodes=20,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=42,
            ),
        ),
        (
            "xgboost",
            xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.04,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:absoluteerror",
                random_state=42,
                n_jobs=4,
            ),
        ),
        (
            "gradient_boosting_huber",
            GradientBoostingRegressor(
                loss="huber",
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=10,
                n_estimators=400,
                random_state=42,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=200,
                max_features=0.7,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_features=0.7,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]


def main() -> None:
    # 1. Load and clean training data
    train = pd.read_csv("train-test.csv")
    train = clean_data(train)

    # 2. Chronological development and holdout split
    development = train[train["date"] < "2025-10-01"].sort_values("date").reset_index(drop=True)
    holdout = train[train["date"] >= "2025-10-01"].reset_index(drop=True)

    X_dev = add_features(development)
    y_dev = development["posted_rate"]
    X_holdout = add_features(holdout)
    y_holdout = holdout["posted_rate"]

    prep = preprocessor()
    X_dev_trans = prep.fit_transform(X_dev)
    X_holdout_trans = prep.transform(X_holdout)

    cv = TimeSeriesSplit(n_splits=3)
    results = []

    print("=== Benchmarking Candidate Models with 3-Fold TimeSeriesSplit ===")
    for name, estimator in get_candidate_models():
        start_time = time.perf_counter()

        # Cross-validation MAE
        cv_maes = []
        for train_idx, test_idx in cv.split(X_dev_trans):
            X_cv_train, y_cv_train = X_dev_trans[train_idx], y_dev.iloc[train_idx]
            X_cv_val, y_cv_val = X_dev_trans[test_idx], y_dev.iloc[test_idx]

            est = estimator.__class__(**estimator.get_params())
            est.fit(X_cv_train, y_cv_train)
            pred_val = est.predict(X_cv_val)
            cv_maes.append(mean_absolute_error(y_cv_val, pred_val))

        mean_cv_mae = float(np.mean(cv_maes))

        # Full fit on development set and evaluation on October holdout
        estimator.fit(X_dev_trans, y_dev)
        holdout_pred = np.maximum(estimator.predict(X_holdout_trans), 0.01)

        holdout_mae = float(mean_absolute_error(y_holdout, holdout_pred))
        holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, holdout_pred)))
        holdout_r2 = float(r2_score(y_holdout, holdout_pred))
        elapsed = round(time.perf_counter() - start_time, 1)

        results.append(
            {
                "model": name,
                "cv_mae": mean_cv_mae,
                "holdout_mae": holdout_mae,
                "holdout_rmse": holdout_rmse,
                "holdout_r2": holdout_r2,
                "seconds": elapsed,
            }
        )
        print(
            f"[{name}] CV MAE: {mean_cv_mae:.2f} | Holdout MAE: {holdout_mae:.2f} | "
            f"RMSE: {holdout_rmse:.2f} | R2: {holdout_r2:.4f} ({elapsed}s)"
        )

    # Sort by holdout MAE
    results.sort(key=lambda x: x["holdout_mae"])
    summary = {
        "ranking": results,
        "selected_model": results[0]["model"],
        "selection_metric": "Holdout MAE (October 2025)",
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    output_path = Path("model_search_results.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved model search results to {output_path}")


if __name__ == "__main__":
    main()
