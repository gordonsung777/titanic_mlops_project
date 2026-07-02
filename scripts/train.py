from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from git import InvalidGitRepositoryError, Repo
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from titanic_mlops.data import load_titanic, sha256_file, split_features_target
from titanic_mlops.modeling import build_model


def get_git_commit() -> str:
    """Return the current git commit SHA, or 'not-a-git-repo' during early local testing."""
    try:
        repo = Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except (InvalidGitRepositoryError, ValueError):
        return "not-a-git-repo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Titanic survival model and log it to MLflow.")
    parser.add_argument("--data", default="data/titanic.csv", help="Path to training CSV")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--output-dir", default="models/candidate", help="Where to save the candidate model")
    parser.add_argument("--reports-dir", default="reports", help="Where to save metrics JSON")
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as file_obj:
        params = yaml.safe_load(file_obj)

    model_params = params["model"]
    training_params = params["training"]

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_titanic(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_params["test_size"],
        random_state=model_params["random_state"],
        stratify=y,
    )

    model = build_model(
        C=float(model_params["C"]),
        max_iter=int(model_params["max_iter"]),
        random_state=int(model_params["random_state"]),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    data_hash = sha256_file(data_path)
    git_commit = get_git_commit()

    local_model_path = output_dir / "model.joblib"
    joblib.dump(model, local_model_path)

    metadata = {
        "git_commit": git_commit,
        "data_path": str(data_path),
        "data_sha256": data_hash,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "features": list(X.columns),
        "metrics": metrics,
        "params": params,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    mlflow.set_experiment("titanic-survival")
    with mlflow.start_run(run_name=f"train-{git_commit[:7]}"):
        mlflow.log_params(
            {
                "model_type": model_params["type"],
                "C": model_params["C"],
                "max_iter": model_params["max_iter"],
                "random_state": model_params["random_state"],
                "test_size": training_params["test_size"],
                "data_sha256": data_hash,
                "git_commit": git_commit,
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_dir / "metadata.json"), artifact_path="metadata")
        mlflow.log_artifact(str(reports_dir / "metrics.json"), artifact_path="reports")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=training_params.get("registered_model_name"),
        )

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved local candidate model to: {local_model_path}")
    print("Logged params, metrics, metadata, and model artifact to MLflow.")


if __name__ == "__main__":
    main()
