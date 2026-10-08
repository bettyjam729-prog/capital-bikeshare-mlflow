# Capital Bikeshare — MLflow Experiment Tracking

Forecast daily bike rental counts using the UCI Bike Sharing `day.csv` dataset. Compare Ridge, Random Forest, and Histogram Gradient Boosting against a simple lagged-demand baseline; track training runs and register the final model with MLflow.

## Setup

```bash
python -m pip install -r requirements.txt
python download_data.py
python train_bike_demand.py
python verify_bike_registry.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Visit http://127.0.0.1:5000 to inspect MLflow runs. The data file comes from https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset .

## Evaluation and limitations

Use chronological train/validation/test periods; select on validation MAE and evaluate once on the holdout test. Because next-day observed weather is unavailable the previous evening, do not use the target day's actual weather or rental totals as features. This lab uses completed earlier-day rentals and known calendar features.

Prior local, non-MLflow offline results: selected **Ridge**, holdout MAE **930.37** vs baseline **1160.83**, and holdout RMSE **1374.47** vs baseline **1628.85**. These are *offline* results, **not verified MLflow runs**. Actual run IDs and registry versions only exist after running the scripts successfully with MLflow installed.

## Dataset attribution

UCI Machine Learning Repository, Bike Sharing Dataset (Capital Bikeshare, 2011–2012). Review the source page for dataset attribution and licensing before redistribution.
