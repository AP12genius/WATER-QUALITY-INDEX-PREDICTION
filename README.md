# Water Quality Index Prediction Handbook

## What this project achieves

This project forecasts a river monitoring station's **next-year Water Quality
Index (WQI)** from the measurements available in its current year. It provides
a reproducible, leakage-safe benchmark for Indian river-water observations and
compares several machine-learning approaches against simple, meaningful
baselines.

The model is intended to support water-quality monitoring, not replace field
sampling or regulatory assessment.

## Data and parameters

The dataset in `data/dataset.csv` contains nationwide observations from
2003–2014. The forecasting inputs are:

| Parameter | Meaning |
| --- | --- |
| pH | Acidity/alkalinity of water |
| DO | Dissolved oxygen (mg/L) |
| BOD | Biochemical oxygen demand (mg/L) |
| TC / FC | Total and fecal coliform counts (MPN/100 mL) |
| Nitrate | Nitrate/nitrite concentration (mg/L) |

Duplicate measurements for the same station and year are averaged. The WQI
target uses a documented five-input approximation: DO (0.31), pH (0.11), BOD
(0.19), fecal coliform (0.28), and nitrate (0.11). The target is the following
calendar year's WQI, never the same row's WQI.

## Before and after

The original notebook calculated WQI from a row and trained models on that
same row. Because the target was built from the input measurements, this was
target leakage and could make R² look unrealistically high. It also used a
random data split, which allows future years to influence training.

The current workflow creates only consecutive station-year pairs, fits
imputation/scaling on training data alone, and splits chronologically:

| Partition | Target year |
| --- | --- |
| Training | 2007–2012 |
| Validation | 2013 |
| Final test | 2014 |

This makes the reported test result a realistic estimate of forecasting a
future annual observation.

## Models and evaluation

The benchmark evaluates a median baseline, a persistence baseline (current WQI),
Random Forest, XGBoost, LightGBM, SVR, and a stacked ensemble. The ensemble's
Ridge meta-model is trained from chronological out-of-fold base-model
predictions, not random cross-validation. Metrics are R², MAE, RMSE, and MSE.

The included 2025 Haridwar research paper is the methodological reference. It
reports individual XGBR, RFR, SVR, and RNN scores on a different monthly
2017–2022 dataset. LightGBM and stacking are project extensions. Paper results
are context, not a performance target for this different dataset and task.

Current final-test results are:

| Model | R² | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Persistence baseline | 0.9076 | 3.1405 | 4.3294 |
| Random Forest | 0.8648 | 3.9590 | 5.2386 |
| XGBoost | 0.8562 | 3.9517 | 5.4021 |
| LightGBM | 0.8521 | 4.0291 | 5.4795 |
| Stacked ensemble | 0.8554 | 4.0033 | 5.4168 |
| SVR | 0.7517 | 5.0512 | 7.0992 |

The persistence baseline is currently best. This is an important finding, not
a failure to hide: it establishes the standard that future models must beat.
The runner flags any train-to-validation R² drop above 0.10; the current
XGBoost configuration is flagged, so it must not be presented as
fully generalised without further validation tuning.

## Setup and usage

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv.lock`
is the exact dependency lock; `requirements.txt` is its pip-compatible export.

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run python -m src.model
```

Open `main.ipynb` to see the full methodology with Markdown explanations,
data preparation, benchmark execution, and SHAP interpretation. Generate a
Random Forest SHAP plot explicitly when required:

```bash
uv run python -m src.model --shap-output /tmp/wqi-shap.png
```

## RNN status and next steps

TensorFlow is available through `uv sync --group rnn`, but the RNN is deferred.
The paper used regular monthly sequences; this dataset is annual and contains
irregular station histories. Before adding an RNN, define a sequence window,
handle missing years, and prove that every history window precedes its target.
The next improvement should be validation-only feature and hyperparameter work,
with the 2014 test set remaining untouched until a final comparison.
