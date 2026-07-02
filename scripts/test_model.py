from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml

from titanic_mlops.data import load_titanic, split_features_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data, metrics, and candidate model before deployment.")
    parser.add_argument("--data", default="data/titanic.csv")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model", default="models/candidate/model.joblib")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--min-accuracy", type=float, default=None)
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as file_obj:
        params = yaml.safe_load(file_obj)

    min_accuracy = args.min_accuracy
    if min_accuracy is None:
        min_accuracy = float(params["training"]["min_accuracy"])

    df = load_titanic(args.data)
    X, _ = split_features_target(df)

    model_path = Path(args.model)
    metrics_path = Path(args.metrics)

    if not model_path.exists():
        raise FileNotFoundError(f"Candidate model missing: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file missing: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)

    accuracy = float(metrics["accuracy"])
    if accuracy < min_accuracy:
        raise AssertionError(f"Accuracy {accuracy:.3f} is below required minimum {min_accuracy:.3f}")

    model = joblib.load(model_path)
    sample_predictions = model.predict(X.head(5))
    if len(sample_predictions) != 5:
        raise AssertionError("Prediction smoke test failed: expected 5 predictions.")

    print("Model validation passed.")
    print(f"Accuracy: {accuracy:.3f} >= minimum: {min_accuracy:.3f}")


if __name__ == "__main__":
    main()
