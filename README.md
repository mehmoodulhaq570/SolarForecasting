# Solar Radiation Forecasting for Lahore, Pakistan

**Final Year Project** — A comparative study of statistical, machine
learning, and deep learning models for short-term solar radiation
forecasting, using NASA POWER hourly meteorological data, deployed as an
interactive Streamlit dashboard.

## Overview

Reliable solar radiation forecasts are essential for planning and
operating photovoltaic (PV) systems — from sizing installations to
scheduling grid dispatch. This project builds and benchmarks six
forecasting models against ~7 years of hourly meteorological and solar
radiation data for Lahore, Pakistan, and packages the best-performing
models into a dashboard for interactive hourly and long-range forecasts.

**Target variable:** hourly global solar radiation (W/m²)
**Location:** Lahore, Pakistan (31.56°N, 74.35°E)
**Data period:** January 2018 – October 2025 (~68,000 hourly records)
**Data source:** [NASA POWER](https://power.larc.nasa.gov/) hourly API

## Models compared

| Model | Type |
|---|---|
| Random Forest | Ensemble tree-based ML |
| XGBoost | Gradient-boosted tree-based ML |
| LSTM | Recurrent deep learning |
| CNN-LSTM | Convolutional + recurrent deep learning |
| TCN | Temporal Convolutional Network |
| TFT | Temporal Fusion Transformer |

### Results (test set)

| Model | R² (test) | MAE | RMSE |
|---|---|---|---|
| XGBoost | 0.993 | 9.49 | 22.26 |
| Random Forest | 0.993 | 9.88 | 22.78 |
| CNN-LSTM | 0.988 | 18.27 | 30.19 |
| LSTM | 0.987 | 18.05 | 31.06 |
| TFT | 0.947 | 28.58 | 63.69 |
| TCN | 0.946 | 28.83 | 64.38 |

*MAE/RMSE in W/m². See [`model_scores/`](model_scores/) for full metrics,
residual analysis, ACF/PACF diagnostics, and actual-vs-predicted plots
per model.*

## Features used

Clear-sky radiation, direct radiation, diffuse radiation, solar zenith
angle, temperature, specific & relative humidity, pressure, wind speed,
and wind direction — all at hourly resolution.

## Project structure

```
frontend/                          Streamlit dashboard package (entry point: frontend/app.py)
  app.py                             Main app / page layout
  api.py                             Real-time weather API integration (Open-Meteo comparison)
  config.py                          Paths, model registry, feature/target config
  data.py, models.py, prediction.py, visualization.py

tft/                                TCN and TFT model definitions + training scripts

nasa_data/                         Data collection & preprocessing
  API_call.py                        NASA POWER API data-pull script
  raw/                                Raw NASA POWER exports (UTC & local time)
  lahore_hourly_filled.csv           Cleaned, gap-filled hourly dataset used for training
  Solar Project.ipynb                Data collection / cleaning notebook

saved_models/                      Random Forest, XGBoost, LSTM, CNN-LSTM (current models)
saved_models_lstm/                 LSTM/CNN-LSTM training run artifacts
saved_models_tft/                  TCN/TFT checkpoints and weights

model_scores/                      Evaluation results
  scores/                            Metrics table (All Model Results.xlsx) + comparison chart
  acf_pacf/                          Autocorrelation diagnostics per variable
  actual_vs_predicted/               Actual vs. predicted plots (hourly + trend), per model
  residual_plots/                    Residual distribution plots, per model

frequency_plots/                   Frequency distribution plots per input variable
trends/                            Generated long-range forecast outputs (CSV + PNG), by date
docs/                              Technical report, model explanations, user manual,
                                    statistical analysis of the input data

frontend.py, training.py, trend.py  Standalone scripts (pre-refactor entry points)
run_frontend.py                    Convenience launcher for the dashboard
convert_models.py                  Keras 2.x -> 3.x model conversion utility
```

## Getting started

### Requirements

- Python 3.11
- See [`requirements.txt`](requirements.txt) (Streamlit, pandas, scikit-learn,
  XGBoost, TensorFlow, PyTorch, PyTorch Lightning)

### Setup

```bash
python -m venv solar_env
solar_env\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run frontend/app.py
```

or

```bash
python run_frontend.py
```

The dashboard provides hourly and multi-day forecasts, model comparison
views, and a live comparison against a real-time weather API.

### Retrain models

```bash
python training.py          # Random Forest, XGBoost, LSTM, CNN-LSTM
python tft/train.py          # TCN / TFT
```

## Documentation

Full write-ups — technical report, per-model explanations, statistical
analysis of the input variables, and a user manual for training and
running the dashboard — are in [`docs/`](docs/).

## Author

Faisal — Final Year Project, [add department / university here]
