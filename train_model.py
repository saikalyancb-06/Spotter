"""
Production training pipeline for the Spotter Freight Rate Prediction Challenge.

This script implements:
1. Data cleaning: Fixes negative weight sign entry errors and imputes daily market index.
2. Temporal validation: Trains on development data (Jan-Sep 2025) and validates on October 2025.
3. Model architecture: Multi-seed HistGradientBoostingRegressor with L1 absolute error loss.
4. Final inference: Refits on all 48,000 development rows to generate validation_predictions.csv
   and populates december-chart-inputs.csv using route-specific coordinates and market signals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET = "posted_rate"
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment", "route"]
NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "day_of_year",
    "delta_lat",
    "delta_lon",
    "ton_miles",
]


def clean_data(frame: pd.DataFrame, daily_market_lookup: pd.Series | None = None) -> pd.DataFrame:
    """Clean data: fix sign errors in payload weights and impute market_index temporally."""
    result = frame.copy()
    if "weight" in result:
        # Invert negative weight entries (data entry sign error)
        result["weight"] = result["weight"].abs()

    if "date" in result:
        result["date"] = pd.to_datetime(result["date"])

    if "market_index" in result:
        if daily_market_lookup is not None:
            # Impute missing market index from precomputed daily medians
            result["market_index"] = result["market_index"].fillna(
                result["date"].dt.strftime("%Y-%m-%d").map(daily_market_lookup)
            )
        else:
            # Compute daily median within the dataframe
            daily_median = result.groupby("date")["market_index"].transform("median")
            result["market_index"] = result["market_index"].fillna(daily_median)

    return result


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate calendar, geographic, and freight domain features."""
    result = frame.copy()
    for column in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]:
        if column not in result:
            result[column] = np.nan

    dates = pd.to_datetime(result["date"], errors="coerce")
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["week_of_year"] = dates.dt.isocalendar().week.astype(float)
    result["day_of_year"] = dates.dt.dayofyear
    result["route"] = result["pickup"].astype(str) + " -> " + result["delivery"].astype(str)

    # Directional coordinate differences
    result["delta_lat"] = result["delivery_lat"] - result["pickup_lat"]
    result["delta_lon"] = result["delivery_lon"] - result["pickup_lon"]

    # Ton-miles freight workload
    result["ton_miles"] = (result["weight"] / 2000.0) * result["distance"]

    return result


def make_pipeline(seed: int = 42) -> Pipeline:
    """Create preprocessing and HistGradientBoostingRegressor pipeline."""
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    # Tuned hyperparameters: L1 absolute loss, shallow leaves to prevent overfitting
    regressor = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=20,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessing), ("model", regressor)])


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Compute standard regression performance metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def predict_december_scenario(
    models: list[Pipeline],
    december_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict the December 2025 benchmark with route-specific coordinates and market signals."""
    december = pd.read_csv(december_path)
    dec_features = december.copy()

    # Look up known coordinates for Lexington and Fort Wayne
    pickup_coords = train_df.groupby("pickup")[["pickup_lat", "pickup_lon"]].median()
    delivery_coords = train_df.groupby("delivery")[["delivery_lat", "delivery_lon"]].median()
    dec_features["pickup_lat"] = pickup_coords.loc["Lexington", "pickup_lat"]
    dec_features["pickup_lon"] = pickup_coords.loc["Lexington", "pickup_lon"]
    dec_features["delivery_lat"] = delivery_coords.loc["Fort Wayne", "delivery_lat"]
    dec_features["delivery_lon"] = delivery_coords.loc["Fort Wayne", "delivery_lon"]

    # Median quote signal for Lexington -> Fort Wayne Dry Van
    lex_fw_quote = train_df[
        (train_df["pickup"] == "Lexington")
        & (train_df["delivery"] == "Fort Wayne")
        & (train_df["equipment"] == "Dry Van")
    ]["quote_signal"].median()
    dec_features["quote_signal"] = lex_fw_quote

    # Daily market index in December from validation data
    val_dec = val_df[val_df["date"] >= "2025-12-01"].copy()
    daily_mkt = val_dec.groupby(val_dec["date"].dt.strftime("%Y-%m-%d"))["market_index"].median()
    dec_features["market_index"] = dec_features["date"].map(daily_mkt)

    # Average predictions across ensemble seeds
    features_prepared = add_features(dec_features)
    preds = np.mean([model.predict(features_prepared) for model in models], axis=0)
    december["predicted_rate"] = np.maximum(preds, 0.01)
    return december


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the freight-rate prediction model.")
    parser.add_argument("--train", default="train-test.csv")
    parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--template", default="validation-predictions-template.csv")
    parser.add_argument("--december", default="december-chart-inputs.csv")
    parser.add_argument("--output", default="validation_predictions.csv")
    parser.add_argument("--metrics-output", default="validation_metrics.json")
    args = parser.parse_args()

    # 1. Load and clean training data
    train = pd.read_csv(args.train)
    train = clean_data(train)

    # 2. Strict chronological validation split
    development = train[train["date"] < "2025-10-01"].copy()
    holdout = train[train["date"] >= "2025-10-01"].copy()

    # Build multi-seed ensemble
    seeds = [42, 7, 21, 99, 123]
    dev_models = [make_pipeline(seed=s) for s in seeds]

    X_dev = add_features(development)
    y_dev = development[TARGET]
    X_holdout = add_features(holdout)

    holdout_preds = []
    for model in dev_models:
        model.fit(X_dev, y_dev)
        holdout_preds.append(np.maximum(model.predict(X_holdout), 0.01))

    ensemble_holdout_pred = np.mean(holdout_preds, axis=0)
    validation_metrics = metrics(holdout[TARGET], ensemble_holdout_pred)
    validation_metrics["split"] = (
        "train: 2025-01-01 through 2025-09-30; holdout: 2025-10-01 through 2025-10-31"
    )
    validation_metrics["model"] = "5-Seed HistGradientBoostingRegressor Ensemble"

    # 3. Refit ensemble on all 48,000 development rows for final predictions
    X_train_full = add_features(train)
    y_train_full = train[TARGET]
    final_models = [make_pipeline(seed=s) for s in seeds]
    for model in final_models:
        model.fit(X_train_full, y_train_full)

    # 4. Predict on validation.csv
    final_validation = pd.read_csv(args.validation)
    final_val_cleaned = clean_data(final_validation)
    val_preds = np.mean(
        [model.predict(add_features(final_val_cleaned)) for model in final_models], axis=0
    )
    val_preds = np.maximum(val_preds, 0.01)

    template = pd.read_csv(args.template)
    output = template[["load_id"]].copy()
    prediction_by_id = dict(zip(final_validation["load_id"], val_preds))
    output["predicted_rate"] = output["load_id"].map(prediction_by_id)
    if output["predicted_rate"].isna().any():
        raise ValueError("Every template load_id must have a prediction")
    output.to_csv(args.output, index=False)

    # 5. Predict December benchmark scenario
    december_df = predict_december_scenario(final_models, args.december, train, final_val_cleaned)
    december_df.to_csv(args.december, index=False)

    # 6. Save metrics
    Path(args.metrics_output).write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    print("=== Validation Metrics (October Holdout) ===")
    print(json.dumps(validation_metrics, indent=2))
    print(f"\nWrote {args.output} and updated {args.december}")


if __name__ == "__main__":
    main()
