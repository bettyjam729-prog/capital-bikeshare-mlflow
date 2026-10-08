"""Leakage-aware Capital Bikeshare forecast experiment with MLflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "time_idx", "weekday", "month", "holiday", "workingday", "season",
    "lag_2", "lag_3", "lag_7", "rolling_7",
]


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values("dteday").reset_index(drop=True)
    if not df["dteday"].is_unique:
        raise ValueError("Duplicate dates")
    df["time_idx"] = (df["dteday"] - df["dteday"].min()).dt.days
    # At the evening of day T-1, T-1 rental totals may not yet be final.
    # Therefore only counts through T-2 are used to forecast day T.
    for lag in (2, 3, 7):
        df[f"lag_{lag}"] = df["cnt"].shift(lag)
    df["rolling_7"] = df["cnt"].shift(2).rolling(7).mean()
    return df.dropna(subset=FEATURES + ["cnt"]).reset_index(drop=True)


def score(actual, prediction):
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/day.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="bike-demand-forecast")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    needed = {"dteday", "cnt", "weekday", "mnth", "holiday", "workingday", "season"}
    if needed - set(raw.columns):
        raise ValueError(f"Missing columns: {needed - set(raw.columns)}")
    df = make_features(raw.rename(columns={"mnth": "month"}))
    test_start = int(len(df) * 0.8)
    development, test = df.iloc[:test_start], df.iloc[test_start:]
    validation_start = int(len(development) * 0.8)
    train, validation = development.iloc[:validation_start], development.iloc[validation_start:]

    candidates = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "RandomForest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=42
        ),
    }

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    runs = []
    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            estimator.fit(train[FEATURES], train["cnt"])
            metrics = score(validation["cnt"], estimator.predict(validation[FEATURES]))
            baseline = score(validation["cnt"], validation["lag_2"])
            mlflow.set_tags({"project": "Capital Bikeshare", "stage": "validation", "model_family": name})
            mlflow.log_input(mlflow.data.from_pandas(train[FEATURES + ["cnt"]], name="training-data", targets="cnt"), context="training")
            mlflow.log_input(mlflow.data.from_pandas(validation[FEATURES + ["cnt"]], name="validation-data", targets="cnt"), context="validation")
            mlflow.log_params({
                "model": name, "features": ",".join(FEATURES),
                "split": "chronological", "target_day_lag": 2,
                "training_period": f"{train.dteday.min().date()} to {train.dteday.max().date()}",
                "validation_period": f"{validation.dteday.min().date()} to {validation.dteday.max().date()}",
            })
            mlflow.log_metrics({
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_baseline_mae": baseline["mae"],
            })
            runs.append({"model": name, "validation_mae": metrics["mae"],
                         "run_id": run.info.run_id})

    winner = min(runs, key=lambda run: run["validation_mae"])
    final = clone(candidates[winner["model"]]).fit(
        development[FEATURES], development["cnt"]
    )
    predictions = final.predict(test[FEATURES])
    final_score = score(test["cnt"], predictions)
    baseline_score = score(test["cnt"], test["lag_2"])

    pd.DataFrame({
        "date": test["dteday"].dt.strftime("%Y-%m-%d"),
        "actual": test["cnt"],
        "predicted": predictions,
        "baseline_lag_2": test["lag_2"],
    }).to_csv(out / "test_predictions.csv", index=False)

    report = {
        "selected_model": winner["model"],
        "selection": "lowest validation MAE; holdout test used once",
        "validation_results": runs,
        "test_start": str(test["dteday"].min().date()),
        "test_end": str(test["dteday"].max().date()),
        "train_rows": int(len(development)),
        "test_rows": int(len(test)),
        "test_metrics": final_score,
        "baseline_metrics": baseline_score,
        "outperforms_baseline_on_mae": final_score["mae"] < baseline_score["mae"],
        "notes": "Target-day observed weather not used. Historical counts through T-2.",
    }

    with mlflow.start_run(run_name=f"FINAL_{winner['model']}") as run:
        mlflow.set_tags({"project": "Capital Bikeshare", "stage": "holdout", "selected_model": winner["model"]})
        mlflow.log_input(mlflow.data.from_pandas(development[FEATURES + ["cnt"]], name="final-training-data", targets="cnt"), context="training")
        mlflow.log_input(mlflow.data.from_pandas(test[FEATURES + ["cnt"]], name="holdout-test-data", targets="cnt"), context="testing")
        mlflow.log_params({
            "model": winner["model"], "selected_from_run": winner["run_id"],
            "features": ",".join(FEATURES), "train_rows": len(development),
            "test_rows": len(test),
        })
        mlflow.log_metrics({
            "test_mae": final_score["mae"], "test_rmse": final_score["rmse"],
            "baseline_mae": baseline_score["mae"],
            "baseline_rmse": baseline_score["rmse"],
        })
        artifact = mlflow.sklearn.log_model(
            sk_model=final, name="daily-demand-forecast",
            input_example=development[FEATURES].head(3),
        )
        report["final_run_id"] = run.info.run_id
        report["model_uri"] = artifact.model_uri
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out / "report.json"))
        mlflow.log_artifact(str(out / "test_predictions.csv"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
