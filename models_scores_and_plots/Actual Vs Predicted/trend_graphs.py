# ==========================================
# 📈 Trend Graphs: Actual vs Predicted
# All 6 Models — Daily Solar Energy (Test Set)
# ==========================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_ROOT, "NASA meteriological and solar radiaton data", "lahore_hourly_filled.csv")
LSTM_DIR = os.path.join(PROJECT_ROOT, "saved_models_lstm")
TFT_DIR = os.path.join(PROJECT_ROOT, "saved_models_tft")
OUT_DIR = SCRIPT_DIR
PREDICTIONS_CACHE_PATH = os.path.join(OUT_DIR, "trend_predictions_cache.pkl")

sys.path.insert(0, PROJECT_ROOT)   # for tft/ imports

# ── Global plot style (same as scores_graphs.py) ───────────────────────────────
plt.rcParams.update({"font.family": "serif", "font.size": 13})
COLOR_ACTUAL    = "#32CD32"   # green (True / Actual)
COLOR_PREDICTED = "#FF2400"   # scarlet red (Predicted)

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 13
LEGEND_SIZE_MAIN = 14
LEGEND_SIZE_COMBINED = 11

SEQ_LEN  = 24
MAX_LAG  = 24
TARGET   = "SolarRadiation"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load & engineer features  (mirrors training.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Normalize alternate column names to match training-time names.
column_aliases = {
    "SolarZenithAngle": "SolarZenith",
    "SpecificHumidity": "HumiditySpecific",
    "RelativeHumidity": "HumidityRelative",
}
df.rename(columns=column_aliases, inplace=True)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_index()
df.set_index("datetime", inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")

df_day = df.copy()
df_day.dropna(subset=[TARGET], inplace=True)

df_day["hour"]       = df_day.index.hour
df_day["month"]      = df_day.index.month
df_day["day_of_year"]= df_day.index.dayofyear
df_day["hour_sin"]   = np.sin(2 * np.pi * df_day["hour"]        / 24)
df_day["hour_cos"]   = np.cos(2 * np.pi * df_day["hour"]        / 24)
df_day["month_sin"]  = np.sin(2 * np.pi * df_day["month"]       / 12)
df_day["month_cos"]  = np.cos(2 * np.pi * df_day["month"]       / 12)
df_day["doy_sin"]    = np.sin(2 * np.pi * df_day["day_of_year"] / 365)
df_day["doy_cos"]    = np.cos(2 * np.pi * df_day["day_of_year"] / 365)
df_day["ClearnessIndex"] = np.where(
    df_day["ClearSkyRadiation"] > 0,
    (df_day[TARGET] / df_day["ClearSkyRadiation"]).clip(0, 1), 0
)

split_idx = int(len(df_day) * 0.8)
train_df  = df_day.iloc[:split_idx]
test_df   = df_day.iloc[split_idx:]
print(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()}")
print(f"Test : {test_df.index.min().date()}  → {test_df.index.max().date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Tree-model features (XGBoost & Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
TREE_BASE_COLS = [
    TARGET,
    "hour", "month", "day_of_year",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "SolarZenith", "ClearSkyRadiation",
    "Temperature", "HumiditySpecific", "HumidityRelative",
    "Pressure", "WindSpeed", "WindDirection",
]

df_lag = df_day[TREE_BASE_COLS].copy()
for lag in range(1, MAX_LAG + 1):
    df_lag[f"lag_{lag}"] = df_lag[TARGET].shift(lag)
df_lag.dropna(inplace=True)

test_lag  = df_lag.loc[df_lag.index.intersection(test_df.index)]
X_test_tree = test_lag.drop(columns=[TARGET])
y_test_tree = test_lag[TARGET]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Sequence features for LSTM / CNN-LSTM  (uses scaler_seq — 15 features)
# ══════════════════════════════════════════════════════════════════════════════
SEQ_FEATURES_LSTM = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "SolarZenith", "ClearSkyRadiation",
    "Temperature", "HumiditySpecific", "HumidityRelative",
    "Pressure", "WindSpeed", "WindDirection",
    TARGET,   # 15th — past radiation in the window
]

scaler_seq = joblib.load(os.path.join(LSTM_DIR, "scaler_seq.pkl"))
scaler_y_lstm = joblib.load(os.path.join(LSTM_DIR, "scaler_y.pkl"))

df_seq_lstm = df_day[SEQ_FEATURES_LSTM].dropna()
all_arr_lstm = scaler_seq.transform(df_seq_lstm)

Xs_lstm, ys_lstm, idxs_lstm = [], [], []
for i in range(SEQ_LEN, len(df_seq_lstm)):
    Xs_lstm.append(all_arr_lstm[i - SEQ_LEN : i])
    ys_lstm.append(all_arr_lstm[i, -1])
    idxs_lstm.append(df_seq_lstm.index[i])

Xs_lstm   = np.array(Xs_lstm)
ys_lstm   = np.array(ys_lstm)
idxs_lstm = np.array(idxs_lstm)

test_set = set(test_df.index)
test_mask_lstm = np.array([idx in test_set for idx in idxs_lstm])
X_test_lstm = Xs_lstm[test_mask_lstm]
idx_test_lstm = idxs_lstm[test_mask_lstm]
y_test_lstm_unscaled = scaler_y_lstm.inverse_transform(
    ys_lstm[test_mask_lstm].reshape(-1, 1)
).flatten()

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Sequence features for TFT / TCN  (uses scaler_X — 17 features, no target)
#     The saved TFT/TCN models were trained with raw time + cyclical + weather
#     (same as TREE_FEATURES), fitted via scaler_X in saved_models_tft.
# ══════════════════════════════════════════════════════════════════════════════
SEQ_FEATURES_TFT = [
    "hour", "month", "day_of_year",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "SolarZenith", "ClearSkyRadiation",
    "Temperature", "HumiditySpecific", "HumidityRelative",
    "Pressure", "WindSpeed", "WindDirection",
]   # 17 features — matches scaler_X fitted shape in saved_models_tft

scaler_X_tft  = joblib.load(os.path.join(TFT_DIR, "scaler_X.pkl"))
scaler_y_tft  = joblib.load(os.path.join(TFT_DIR, "scaler_y.pkl"))

df_seq_tft = df_day[SEQ_FEATURES_TFT + [TARGET]].dropna()
X_arr_tft  = scaler_X_tft.transform(df_seq_tft[SEQ_FEATURES_TFT])
y_arr_tft  = scaler_y_tft.transform(df_seq_tft[[TARGET]]).flatten()

Xs_tft, ys_tft, idxs_tft = [], [], []
for i in range(SEQ_LEN, len(df_seq_tft)):
    Xs_tft.append(X_arr_tft[i - SEQ_LEN : i])
    ys_tft.append(y_arr_tft[i])
    idxs_tft.append(df_seq_tft.index[i])

Xs_tft   = np.array(Xs_tft)
idxs_tft = np.array(idxs_tft)
ys_tft   = np.array(ys_tft)

test_mask_tft = np.array([idx in test_set for idx in idxs_tft])
X_test_tft   = Xs_tft[test_mask_tft]
idx_test_tft = idxs_tft[test_mask_tft]
y_test_tft_unscaled = scaler_y_tft.inverse_transform(
    ys_tft[test_mask_tft].reshape(-1, 1)
).flatten()

print(f"Test samples — tree: {len(X_test_tree)}, LSTM-seq: {len(X_test_lstm)}, TFT-seq: {len(X_test_tft)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Run model predictions
# ══════════════════════════════════════════════════════════════════════════════
predictions        = None   # model_name → (pd.Series actual_daily, pd.Series pred_daily)
predictions_hourly = None   # model_name → (pd.Series actual_hourly, pd.Series pred_hourly)

def to_daily(index_arr, actual_arr, pred_arr):
    """Aggregate hourly arrays to daily sums."""
    df_h = pd.DataFrame({"actual": actual_arr, "predicted": pred_arr}, index=pd.DatetimeIndex(index_arr))
    daily = df_h.resample("D").sum()
    counts = df_h.resample("D").count()
    daily = daily[counts["actual"] >= 20]
    return daily["actual"], daily["predicted"]

def to_hourly(index_arr, actual_arr, pred_arr):
    """Return hourly Series (clamp negatives to 0)."""
    idx = pd.DatetimeIndex(index_arr)
    act = pd.Series(np.clip(actual_arr,    0, None), index=idx)
    prd = pd.Series(np.clip(pred_arr,      0, None), index=idx)
    return act, prd

model_order = ["XGBoost", "Random Forest", "LSTM", "CNN-LSTM", "TFT", "TCN"]

if os.path.exists(PREDICTIONS_CACHE_PATH):
    try:
        cached = joblib.load(PREDICTIONS_CACHE_PATH)
        daily_cached = cached.get("predictions") if isinstance(cached, dict) else None
        hourly_cached = (
            cached.get("predictions_hourly") if isinstance(cached, dict) else None
        )
        if (
            isinstance(daily_cached, dict)
            and isinstance(hourly_cached, dict)
            and all(m in daily_cached for m in model_order)
            and all(m in hourly_cached for m in model_order)
        ):
            predictions = daily_cached
            predictions_hourly = hourly_cached
            print(f"Loaded cached predictions from: {PREDICTIONS_CACHE_PATH}")
            print("Skipping model prediction and using cached series.")
    except Exception as exc:
        print(f"Could not load prediction cache ({exc}). Recomputing predictions...")

if predictions is None or predictions_hourly is None:
    predictions = {}
    predictions_hourly = {}

    # ── 5a. XGBoost ───────────────────────────────────────────────────────────
    print("Predicting XGBoost …")
    import xgboost as xgb

    xgb_model = joblib.load(os.path.join(LSTM_DIR, "xgboost_model.pkl"))
    xgb_pred = xgb_model.predict(X_test_tree)
    predictions["XGBoost"] = to_daily(test_lag.index, y_test_tree.values, xgb_pred)
    predictions_hourly["XGBoost"] = to_hourly(
        test_lag.index, y_test_tree.values, xgb_pred
    )

    # ── 5b. Random Forest ─────────────────────────────────────────────────────
    print("Predicting Random Forest …")
    rf_model = joblib.load(os.path.join(LSTM_DIR, "random_forest_model.pkl"))
    # Patch for scikit-learn version mismatch (1.3.2 → 1.7.2)
    for est in rf_model.estimators_:
        if not hasattr(est, "monotonic_cst"):
            est.monotonic_cst = None
    rf_pred = rf_model.predict(X_test_tree)
    predictions["Random Forest"] = to_daily(
        test_lag.index, y_test_tree.values, rf_pred
    )
    predictions_hourly["Random Forest"] = to_hourly(
        test_lag.index, y_test_tree.values, rf_pred
    )

    # ── 5c. LSTM ──────────────────────────────────────────────────────────────
    print("Predicting LSTM …")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    keras_path_lstm = os.path.join(LSTM_DIR, "lstm_model_v3.keras")
    if not os.path.exists(keras_path_lstm):
        keras_path_lstm = os.path.join(LSTM_DIR, "lstm_model.h5")
    lstm_model = tf.keras.models.load_model(keras_path_lstm)
    lstm_scaled = lstm_model.predict(X_test_lstm, verbose=0)
    lstm_pred = scaler_y_lstm.inverse_transform(lstm_scaled).flatten()
    predictions["LSTM"] = to_daily(idx_test_lstm, y_test_lstm_unscaled, lstm_pred)
    predictions_hourly["LSTM"] = to_hourly(
        idx_test_lstm, y_test_lstm_unscaled, lstm_pred
    )

    # ── 5d. CNN-LSTM ──────────────────────────────────────────────────────────
    print("Predicting CNN-LSTM …")
    keras_path_cnn = os.path.join(LSTM_DIR, "cnn_lstm_model_v3.keras")
    if not os.path.exists(keras_path_cnn):
        keras_path_cnn = os.path.join(LSTM_DIR, "cnn_lstm_model.h5")
    cnn_model = tf.keras.models.load_model(keras_path_cnn)
    cnn_scaled = cnn_model.predict(X_test_lstm, verbose=0)
    cnn_pred = scaler_y_lstm.inverse_transform(cnn_scaled).flatten()
    predictions["CNN-LSTM"] = to_daily(idx_test_lstm, y_test_lstm_unscaled, cnn_pred)
    predictions_hourly["CNN-LSTM"] = to_hourly(
        idx_test_lstm, y_test_lstm_unscaled, cnn_pred
    )

    # ── 5e. TFT ───────────────────────────────────────────────────────────────
    print("Predicting TFT …")
    import torch
    from tft.tft import LitTFT

    tft_input_size = X_test_tft.shape[2]  # 17
    tft_lit = LitTFT(
        input_size=tft_input_size,
        hidden_size=64,
        output_size=1,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    )
    tft_state = torch.load(os.path.join(TFT_DIR, "tft_model.pt"), map_location="cpu")
    tft_lit.load_state_dict(tft_state)
    tft_lit.eval()
    with torch.no_grad():
        tft_tensor = torch.tensor(X_test_tft, dtype=torch.float32)
        tft_out = tft_lit(tft_tensor).numpy().flatten()
    tft_pred = scaler_y_tft.inverse_transform(tft_out.reshape(-1, 1)).flatten()
    predictions["TFT"] = to_daily(idx_test_tft, y_test_tft_unscaled, tft_pred)
    predictions_hourly["TFT"] = to_hourly(idx_test_tft, y_test_tft_unscaled, tft_pred)

    # ── 5f. TCN ───────────────────────────────────────────────────────────────
    print("Predicting TCN …")
    from tft.tcn import LitTCN

    tcn_input_size = X_test_tft.shape[2]  # 17
    tcn_lit = LitTCN(
        input_size=tcn_input_size,
        hidden_size=64,
        output_size=1,
        num_layers=4,
        kernel_size=3,
        dropout=0.2,
    )
    tcn_state = torch.load(os.path.join(TFT_DIR, "tcn_model.pt"), map_location="cpu")
    tcn_lit.load_state_dict(tcn_state)
    tcn_lit.eval()
    with torch.no_grad():
        tcn_tensor = torch.tensor(X_test_tft, dtype=torch.float32)
        tcn_out = tcn_lit(tcn_tensor).numpy().flatten()
    tcn_pred = scaler_y_tft.inverse_transform(tcn_out.reshape(-1, 1)).flatten()
    predictions["TCN"] = to_daily(idx_test_tft, y_test_tft_unscaled, tcn_pred)
    predictions_hourly["TCN"] = to_hourly(idx_test_tft, y_test_tft_unscaled, tcn_pred)

    print("All predictions complete.")
    joblib.dump(
        {
            "predictions": predictions,
            "predictions_hourly": predictions_hourly,
        },
        PREDICTIONS_CACHE_PATH,
    )
    print(f"Saved prediction cache to: {PREDICTIONS_CACHE_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Plotting helper
# ══════════════════════════════════════════════════════════════════════════════
def plot_trend(model_name, actual, predicted, save_path):
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(actual.index,    actual.values,    color=COLOR_ACTUAL,
            linewidth=1.5, label="True (Actual)",  zorder=2)
    ax.plot(predicted.index, predicted.values, color=COLOR_PREDICTED,
            linewidth=1.5, label="Predicted",       zorder=3, alpha=0.85)

    # ── Axis labels & title ──────────────────────────────────────────────────
    ax.set_title(
        f"Daily Solar Energy Forecast using {model_name}",
        fontsize=TITLE_SIZE, fontweight="bold", pad=12
    )
    ax.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Daily Energy (Wh/m²)", fontsize=LABEL_SIZE, fontweight="bold")

    # ── Tick formatting ──────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE, width=2)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=2)

    # Bold tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # ── Grid ─────────────────────────────────────────────────────────────────
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend = ax.legend(fontsize=LEGEND_SIZE_MAIN, frameon=True, loc="upper right")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.basename(save_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Save individual plots
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving individual trend plots …")
for model_name, (actual, predicted) in predictions.items():
    fname = f"trend_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plot_trend(model_name, actual, predicted, os.path.join(OUT_DIR, fname))

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Combined 3×2 overview figure
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving combined 3×2 overview …")
fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=False)
axes = axes.flatten()

for ax, model_name in zip(axes, model_order):
    actual, predicted = predictions[model_name]

    ax.plot(actual.index,    actual.values,    color=COLOR_ACTUAL,
            linewidth=1.2, label="True (Actual)", zorder=2)
    ax.plot(predicted.index, predicted.values, color=COLOR_PREDICTED,
            linewidth=1.2, label="Predicted",      zorder=3, alpha=0.85)

    ax.set_title(f"{model_name}", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Daily Energy (Wh/m²)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE, width=1.5)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    legend = ax.legend(fontsize=LEGEND_SIZE_COMBINED, frameon=True, loc="upper right")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

fig.suptitle(
    "Daily Solar Energy Forecast — Actual vs Predicted (All Models)",
    fontsize=TITLE_SIZE, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trend_all_models.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved → trend_all_models.png")
# ══════════════════════════════════════════════════════════════════════════════
# 9.  Hourly trend — individual plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_hourly_trend(model_name, actual, predicted, save_path):
    fig, ax = plt.subplots(figsize=(18, 5))

    ax.plot(actual.index,    actual.values,    color=COLOR_ACTUAL,
            linewidth=0.8, label="True (Actual)",  zorder=2)
    ax.plot(predicted.index, predicted.values, color=COLOR_PREDICTED,
            linewidth=0.8, label="Predicted",       zorder=3, alpha=0.85)

    ax.set_title(
        f"Hourly Solar Radiation Forecast using {model_name}",
        fontsize=TITLE_SIZE, fontweight="bold", pad=12
    )
    ax.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Solar Radiation (W/m²)", fontsize=LABEL_SIZE, fontweight="bold")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE, width=2)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=2)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(fontsize=LEGEND_SIZE_MAIN, frameon=True, loc="upper right")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.basename(save_path)}")


print("\nSaving individual hourly trend plots …")
for model_name, (actual, predicted) in predictions_hourly.items():
    fname = f"hourly_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plot_hourly_trend(model_name, actual, predicted, os.path.join(OUT_DIR, fname))

# ══════════════════════════════════════════════════════════════════════════════
# 10.  Hourly trend — combined 3×2 overview
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving combined hourly 3×2 overview …")
fig, axes = plt.subplots(3, 2, figsize=(22, 15), sharex=False)
axes = axes.flatten()

for ax, model_name in zip(axes, model_order):
    actual, predicted = predictions_hourly[model_name]

    ax.plot(actual.index,    actual.values,    color=COLOR_ACTUAL,
            linewidth=0.6, label="True (Actual)", zorder=2)
    ax.plot(predicted.index, predicted.values, color=COLOR_PREDICTED,
            linewidth=0.6, label="Predicted",      zorder=3, alpha=0.85)

    ax.set_title(f"{model_name}", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Solar Radiation (W/m²)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30, labelsize=TICK_SIZE, width=1.5)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(fontsize=LEGEND_SIZE_COMBINED, frameon=True, loc="upper right")
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

fig.suptitle(
    "Hourly Solar Radiation Forecast — Actual vs Predicted (All Models)",
    fontsize=TITLE_SIZE, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hourly_all_models.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved → hourly_all_models.png")
print("\nDone! All trend graphs saved to:", OUT_DIR)
