# Solar Radiation Forecasting (FYP)

Final Year Project forecasting solar radiation and related meteorological
variables for Lahore, using NASA POWER hourly data and a set of ML/DL models
(Random Forest, XGBoost, LSTM, CNN-LSTM, TCN, TFT), served through a
Streamlit dashboard.

## Project layout

```
frontend/                                Streamlit dashboard (live app: frontend/app.py)
tft/                                     TCN / TFT model definitions and training scripts
NASA Meteorological and Solar Radiation Data/   Raw + processed NASA POWER data, data-collection notebook
saved_models/                            Trained model artifacts (current models)
saved_models_lstm/                       LSTM/CNN-LSTM training run artifacts
saved_models_tft/                        TCN/TFT training run artifacts
models_scores_and_plots/                 Model evaluation: scores, ACF/PACF, residuals, actual-vs-predicted
Frequency Distribution Plots/            Per-variable frequency distribution plots
trends/                                  Generated forecast outputs (CSV + PNG) by date
docs/                                    Technical report, model explanations, user manual, statistical analysis
frontend.py, training.py, trend.py       Standalone scripts (pre-refactor entry points)
run_frontend.py                          Convenience launcher for the Streamlit app
convert_models.py                        Keras 2.x -> 3.x model conversion utility
```

## Setup

```bash
python -m venv solar_env
solar_env\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Running the dashboard

```bash
streamlit run frontend/app.py
```

or

```bash
python run_frontend.py
```

## Data

Source: NASA POWER hourly meteorological and solar radiation data for
Lahore (2018–present). See
`NASA Meteorological and Solar Radiation Data/API_call.py` for the data
pull, and `lahore_hourly_filled.csv` for the cleaned dataset used for
training.

## Models

Five model families are trained and compared: Random Forest, XGBoost,
LSTM, CNN-LSTM, TCN, and TFT. Evaluation artifacts (metrics, residual
plots, actual-vs-predicted plots) are under `models_scores_and_plots/`.

## Docs

Full write-ups (technical report, model explanations, user manual, and
statistical analysis of the input data) live under `docs/`.
