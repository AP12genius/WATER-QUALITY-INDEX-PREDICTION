# my-ml-model

Water Quality Index (WQI) prediction from water-sample measurements
(pH, dissolved oxygen, BOD, coliform counts, nitrate), using a stacked
ensemble (Random Forest + XGBoost + LightGBM + SVR, with a Ridge
meta-learner) and a simple RNN baseline, plus SHAP-based explainability.

## Project structure

```
my-ml-model/
├── README.md
├── pyproject.toml
├── data/
│   └── dataset.csv        # raw water quality samples
├── assets/
│   └── model.png          # reference diagram / screenshot
├── src/
│   └── model.py           # data loading, feature engineering, model pipeline
├── tests/
│   └── test_model.py      # unit tests for the pipeline
└── notebook.ipynb         # original exploratory notebook
```

## Setup

This project pins its Python version and dependencies in `pyproject.toml`
so everyone gets the same environment.

```bash
# 1. Create a virtual environment (uses the version pinned in pyproject.toml)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install the project (+ dev tools like pytest)
pip install -e ".[dev]"
```

`.venv/` is git-ignored — every teammate creates their own local copy from
`pyproject.toml`, so nobody commits environment-specific files.

## Usage

Run the full training pipeline:

```bash
python -m src.model
```

This loads `data/dataset.csv`, computes the WQI ground truth, engineers
features, trains the stacked ensemble, and prints test-set R², MAE, and RMSE.

## Tests

```bash
pytest
```

## Notebook

`notebook.ipynb` contains the original exploratory work: model comparison,
SHAP explainability, sanity checks on extreme cases, and training-curve
diagnostics (RNN learning curve, XGBoost complexity curve). The reusable
parts of this workflow have been extracted into `src/model.py`.
