"""Register the trained model and verify it maps to the logged MLflow run."""
import argparse
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/report.json")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--registered-name", default="capital-bikeshare-daily-demand")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if not report["outperforms_baseline_on_mae"]:
        raise SystemExit("Not registering: test MAE does not beat baseline")
    mlflow.set_tracking_uri(args.tracking_uri)
    registered = mlflow.register_model(report["model_uri"], args.registered_name)
    fetched = MlflowClient().get_model_version(args.registered_name, registered.version)
    assert fetched.run_id == report["final_run_id"], "Run lineage mismatch"
    print(json.dumps({"name": fetched.name, "version": fetched.version,
                      "run_id": fetched.run_id, "status": fetched.status}, indent=2))

if __name__ == "__main__":
    main()
