from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    print("\n>>> " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train -> test -> deploy as one reproducible pipeline.")
    parser.add_argument("--data", default="data/titanic.csv")
    parser.add_argument("--min-accuracy", default=None)
    args = parser.parse_args()

    run([sys.executable, "scripts/train.py", "--data", args.data])

    test_command = [sys.executable, "scripts/test_model.py", "--data", args.data]
    if args.min_accuracy is not None:
        test_command += ["--min-accuracy", str(args.min_accuracy)]
    run(test_command)

    run([sys.executable, "scripts/deploy.py"])

    print("\nFull pipeline succeeded: training, testing, and deployment all passed.")


if __name__ == "__main__":
    main()
