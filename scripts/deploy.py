from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the validated candidate model to a local deployment folder.")
    parser.add_argument("--candidate-dir", default="models/candidate")
    parser.add_argument("--deployment-dir", default="deployment")
    parser.add_argument("--metrics", default="reports/metrics.json")
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    deployment_dir = Path(args.deployment_dir)
    deployment_dir.mkdir(parents=True, exist_ok=True)

    model_source = candidate_dir / "model.joblib"
    metadata_source = candidate_dir / "metadata.json"
    metrics_source = Path(args.metrics)

    for required in [model_source, metadata_source, metrics_source]:
        if not required.exists():
            raise FileNotFoundError(f"Cannot deploy because this file is missing: {required}")

    shutil.copy2(model_source, deployment_dir / "model.joblib")
    shutil.copy2(metadata_source, deployment_dir / "metadata.json")
    shutil.copy2(metrics_source, deployment_dir / "metrics.json")

    with metrics_source.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)
    with metadata_source.open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    manifest = {
        "deployed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": metadata.get("git_commit"),
        "source_data_sha256": metadata.get("data_sha256"),
        "metrics": metrics,
    }

    with (deployment_dir / "manifest.json").open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)

    print("Deployment complete.")
    print(f"Production-ready model files are in: {deployment_dir}")


if __name__ == "__main__":
    main()
