# Titanic MLOps Project: MLflow + DVC + CI/CD

This project trains a simple Titanic survival model and shows beginner-friendly MLOps practices:

- MLflow tracks model parameters, metrics, metadata, and model artifacts.
- DVC tracks data and pipeline dependencies.
- GitHub Actions reruns the pipeline when code or data changes.
- Python scripts automate train -> test -> deploy.

## Project structure

```text
data/titanic.csv                         # training data
src/titanic_mlops/data.py                # data loading and validation
src/titanic_mlops/modeling.py            # sklearn preprocessing + model pipeline
scripts/train.py                         # trains model and logs to MLflow
scripts/test_model.py                    # validates model before deployment
scripts/deploy.py                        # promotes candidate model to deployment folder
scripts/run_pipeline.py                  # runs training, test, deploy in order
tests/                                   # unit tests
params.yaml                              # versioned training parameters
dvc.yaml                                 # DVC pipeline definition
.github/workflows/mlops-pipeline.yml     # CI/CD pipeline
```

## Local quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_pipeline.py --data data/titanic.csv
mlflow ui
```

Then open the MLflow URL shown in the terminal, usually http://127.0.0.1:5000.

## Git and DVC setup

```bash
git init
dvc init
dvc add data/titanic.csv
git add .gitignore .dvcignore .dvc dvc.yaml params.yaml requirements.txt data/titanic.csv.dvc src scripts tests .github README.md
git commit -m "Initial MLOps pipeline with MLflow and DVC"
```

After editing data or code:

```bash
dvc add data/titanic.csv
git add data/titanic.csv.dvc dvc.lock params.yaml src scripts tests .github
git commit -m "Update data or model pipeline"
```

## Run with DVC

```bash
dvc repro
```

DVC will rerun stages when tracked dependencies change.

## Manual MLflow commands

```bash
python scripts/train.py --data data/titanic.csv
mlflow ui
```

Look for:

- parameters: C, max_iter, test_size, data_sha256, git_commit
- metrics: accuracy, precision, recall, f1, roc_auc
- artifacts: model, reports, metadata

## CI/CD behavior

GitHub Actions runs automatically when you push changes to:

- data files
- source code
- scripts
- tests
- params.yaml
- dvc.yaml
- requirements.txt
- the workflow file itself

It also runs weekly and can be started manually from the GitHub Actions tab.
