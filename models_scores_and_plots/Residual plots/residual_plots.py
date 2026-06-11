# ============================================================
# Residual Histogram Plots — All 6 Models (Hourly Test Set)
# ============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "NASA meteriological and solar radiaton data",
    "lahore_hourly_filled.csv",
)
LSTM_DIR = os.path.join(PROJECT_ROOT, "saved_models_lstm")
TFT_DIR = os.path.join(PROJECT_ROOT, "saved_models_tft")
OUT_DIR = SCRIPT_DIR
RESIDUALS_CACHE_PATH = os.path.join(OUT_DIR, "residuals_cache.pkl")
sys.path.insert(0, PROJECT_ROOT)

# ── Style ─────────────────────────────────────────────────────────────────────
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 18
LEGEND_SIZE = 16

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
    }
)


SEQ_LEN = 24
MAX_LAG = 24
TARGET  = "SolarRadiation"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load & engineer features
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Normalize common alternate column names to training-time names.
column_aliases = {
    "SolarZenithAngle": "SolarZenith",
    "SpecificHumidity": "HumiditySpecific",
    "RelativeHumidity": "HumidityRelative",
}
df.rename(columns=column_aliases, inplace=True)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime")
df.set_index("datetime", inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")

df_day = df.copy()
df_day.dropna(subset=[TARGET], inplace=True)
df_day["hour"]        = df_day.index.hour
df_day["month"]       = df_day.index.month
df_day["day_of_year"] = df_day.index.dayofyear
df_day["hour_sin"]    = np.sin(2 * np.pi * df_day["hour"]        / 24)
df_day["hour_cos"]    = np.cos(2 * np.pi * df_day["hour"]        / 24)
df_day["month_sin"]   = np.sin(2 * np.pi * df_day["month"]       / 12)
df_day["month_cos"]   = np.cos(2 * np.pi * df_day["month"]       / 12)
df_day["doy_sin"]     = np.sin(2 * np.pi * df_day["day_of_year"] / 365)
df_day["doy_cos"]     = np.cos(2 * np.pi * df_day["day_of_year"] / 365)
df_day["ClearnessIndex"] = np.where(
    df_day["ClearSkyRadiation"] > 0,
    (df_day[TARGET] / df_day["ClearSkyRadiation"]).clip(0, 1), 0)

split_idx = int(len(df_day) * 0.8)
train_df  = df_day.iloc[:split_idx]
test_df   = df_day.iloc[split_idx:]
test_set  = set(test_df.index)

print(f"Train: {train_df.index.min().date()} -> {train_df.index.max().date()}")
print(f"Test : {test_df.index.min().date()}  -> {test_df.index.max().date()}")

# ── Tree features ─────────────────────────────────────────────────────────────
TREE_BASE_COLS = [
    TARGET, "hour", "month", "day_of_year",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "SolarZenith", "ClearSkyRadiation", "Temperature", "HumiditySpecific",
    "HumidityRelative", "Pressure", "WindSpeed", "WindDirection",
]
df_lag = df_day[TREE_BASE_COLS].copy()
for lag in range(1, MAX_LAG + 1):
    df_lag[f"lag_{lag}"] = df_lag[TARGET].shift(lag)
df_lag.dropna(inplace=True)
test_lag    = df_lag.loc[df_lag.index.intersection(test_df.index)]
X_test_tree = test_lag.drop(columns=[TARGET])
y_test_tree = test_lag[TARGET]

# ── LSTM / CNN-LSTM features ──────────────────────────────────────────────────
SEQ_FEATURES_LSTM = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "SolarZenith", "ClearSkyRadiation", "Temperature", "HumiditySpecific",
    "HumidityRelative", "Pressure", "WindSpeed", "WindDirection", TARGET,
]
scaler_seq    = joblib.load(os.path.join(LSTM_DIR, "scaler_seq.pkl"))
scaler_y_lstm = joblib.load(os.path.join(LSTM_DIR, "scaler_y.pkl"))

df_seq_lstm   = df_day[SEQ_FEATURES_LSTM].dropna()
all_arr_lstm  = scaler_seq.transform(df_seq_lstm)
Xs_lstm, ys_lstm, idxs_lstm = [], [], []
for i in range(SEQ_LEN, len(df_seq_lstm)):
    Xs_lstm.append(all_arr_lstm[i - SEQ_LEN:i])
    ys_lstm.append(all_arr_lstm[i, -1])
    idxs_lstm.append(df_seq_lstm.index[i])
Xs_lstm   = np.array(Xs_lstm)
ys_lstm   = np.array(ys_lstm)
idxs_lstm = np.array(idxs_lstm)
test_mask_lstm      = np.array([idx in test_set for idx in idxs_lstm])
X_test_lstm         = Xs_lstm[test_mask_lstm]
idx_test_lstm       = idxs_lstm[test_mask_lstm]
y_test_lstm_unscaled = scaler_y_lstm.inverse_transform(
    ys_lstm[test_mask_lstm].reshape(-1, 1)).flatten()

# ── TFT / TCN features ────────────────────────────────────────────────────────
SEQ_FEATURES_TFT = [
    "hour", "month", "day_of_year",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "SolarZenith", "ClearSkyRadiation", "Temperature", "HumiditySpecific",
    "HumidityRelative", "Pressure", "WindSpeed", "WindDirection",
]
scaler_X_tft = joblib.load(os.path.join(TFT_DIR, "scaler_X.pkl"))
scaler_y_tft = joblib.load(os.path.join(TFT_DIR, "scaler_y.pkl"))

df_seq_tft  = df_day[SEQ_FEATURES_TFT + [TARGET]].dropna()
X_arr_tft   = scaler_X_tft.transform(df_seq_tft[SEQ_FEATURES_TFT])
y_arr_tft   = scaler_y_tft.transform(df_seq_tft[[TARGET]]).flatten()
Xs_tft, ys_tft, idxs_tft = [], [], []
for i in range(SEQ_LEN, len(df_seq_tft)):
    Xs_tft.append(X_arr_tft[i - SEQ_LEN:i])
    ys_tft.append(y_arr_tft[i])
    idxs_tft.append(df_seq_tft.index[i])
Xs_tft   = np.array(Xs_tft)
idxs_tft = np.array(idxs_tft)
ys_tft   = np.array(ys_tft)
test_mask_tft       = np.array([idx in test_set for idx in idxs_tft])
X_test_tft          = Xs_tft[test_mask_tft]
idx_test_tft        = idxs_tft[test_mask_tft]
y_test_tft_unscaled = scaler_y_tft.inverse_transform(
    ys_tft[test_mask_tft].reshape(-1, 1)).flatten()

MODEL_ORDER = ["XGBoost", "Random Forest", "LSTM", "CNN-LSTM", "TFT", "TCN"]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Run model predictions (with cache)
# ══════════════════════════════════════════════════════════════════════════════
residuals = None  # model_name -> dict(index, actual, predicted, residual)

if os.path.exists(RESIDUALS_CACHE_PATH):
    try:
        cached = joblib.load(RESIDUALS_CACHE_PATH)
        if isinstance(cached, dict) and all(m in cached for m in MODEL_ORDER):
            residuals = cached
            print(f"Loaded cached residuals from: {RESIDUALS_CACHE_PATH}")
            print("Skipping model prediction and using cached residuals.\n")
    except Exception as exc:
        print(f"Could not load residual cache ({exc}). Recomputing residuals...")

if residuals is None:
    residuals = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("Predicting XGBoost …")
    import xgboost as xgb

    xgb_model = joblib.load(os.path.join(LSTM_DIR, "xgboost_model.pkl"))
    xgb_pred = xgb_model.predict(X_test_tree)
    residuals["XGBoost"] = dict(
        index=test_lag.index,
        actual=y_test_tree.values,
        predicted=xgb_pred,
        residual=y_test_tree.values - xgb_pred,
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("Predicting Random Forest …")
    rf_model = joblib.load(os.path.join(LSTM_DIR, "random_forest_model.pkl"))
    for est in rf_model.estimators_:
        if not hasattr(est, "monotonic_cst"):
            est.monotonic_cst = None
    rf_pred = rf_model.predict(X_test_tree)
    residuals["Random Forest"] = dict(
        index=test_lag.index,
        actual=y_test_tree.values,
        predicted=rf_pred,
        residual=y_test_tree.values - rf_pred,
    )

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("Predicting LSTM …")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    keras_path = os.path.join(LSTM_DIR, "lstm_model_v3.keras")
    if not os.path.exists(keras_path):
        keras_path = os.path.join(LSTM_DIR, "lstm_model.h5")
    lstm_model = tf.keras.models.load_model(keras_path)
    lstm_pred = scaler_y_lstm.inverse_transform(
        lstm_model.predict(X_test_lstm, verbose=0)
    ).flatten()
    residuals["LSTM"] = dict(
        index=idx_test_lstm,
        actual=y_test_lstm_unscaled,
        predicted=lstm_pred,
        residual=y_test_lstm_unscaled - lstm_pred,
    )

    # ── CNN-LSTM ──────────────────────────────────────────────────────────────
    print("Predicting CNN-LSTM …")
    keras_path = os.path.join(LSTM_DIR, "cnn_lstm_model_v3.keras")
    if not os.path.exists(keras_path):
        keras_path = os.path.join(LSTM_DIR, "cnn_lstm_model.h5")
    cnn_model = tf.keras.models.load_model(keras_path)
    cnn_pred = scaler_y_lstm.inverse_transform(
        cnn_model.predict(X_test_lstm, verbose=0)
    ).flatten()
    residuals["CNN-LSTM"] = dict(
        index=idx_test_lstm,
        actual=y_test_lstm_unscaled,
        predicted=cnn_pred,
        residual=y_test_lstm_unscaled - cnn_pred,
    )

    # ── TFT ───────────────────────────────────────────────────────────────────
    print("Predicting TFT …")
    import torch
    from tft.tft import LitTFT

    tft_lit = LitTFT(
        input_size=X_test_tft.shape[2],
        hidden_size=64,
        output_size=1,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    )
    tft_lit.load_state_dict(
        torch.load(os.path.join(TFT_DIR, "tft_model.pt"), map_location="cpu")
    )
    tft_lit.eval()
    with torch.no_grad():
        tft_out = tft_lit(torch.tensor(X_test_tft, dtype=torch.float32)).numpy().flatten()
    tft_pred = scaler_y_tft.inverse_transform(tft_out.reshape(-1, 1)).flatten()
    residuals["TFT"] = dict(
        index=idx_test_tft,
        actual=y_test_tft_unscaled,
        predicted=tft_pred,
        residual=y_test_tft_unscaled - tft_pred,
    )

    # ── TCN ───────────────────────────────────────────────────────────────────
    print("Predicting TCN …")
    from tft.tcn import LitTCN

    tcn_lit = LitTCN(
        input_size=X_test_tft.shape[2],
        hidden_size=64,
        output_size=1,
        num_layers=4,
        kernel_size=3,
        dropout=0.2,
    )
    tcn_lit.load_state_dict(
        torch.load(os.path.join(TFT_DIR, "tcn_model.pt"), map_location="cpu")
    )
    tcn_lit.eval()
    with torch.no_grad():
        tcn_out = tcn_lit(torch.tensor(X_test_tft, dtype=torch.float32)).numpy().flatten()
    tcn_pred = scaler_y_tft.inverse_transform(tcn_out.reshape(-1, 1)).flatten()
    residuals["TCN"] = dict(
        index=idx_test_tft,
        actual=y_test_tft_unscaled,
        predicted=tcn_pred,
        residual=y_test_tft_unscaled - tcn_pred,
    )

    print("All predictions complete.\n")
    joblib.dump(residuals, RESIDUALS_CACHE_PATH)
    print(f"Saved residual cache to: {RESIDUALS_CACHE_PATH}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Residual Histogram — individual + combined
# ══════════════════════════════════════════════════════════════════════════════
C_BAR  = "#1565C0"   # deep blue bars
C_ZERO = "#2E7D32"   # green zero line


def apply_axis_typography(ax):
    ax.tick_params(labelsize=TICK_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

print("\nSaving individual residual histogram plots …")
for model_name in MODEL_ORDER:
    res  = residuals[model_name]["residual"]
    mu   = res.mean()
    sigma = res.std()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.hist(res, bins=80, color=C_BAR, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.axvline(0,  color=C_ZERO,   linewidth=1.8, linestyle="--", label="Zero")
    ax.axvline(mu, color="#B71C1C", linewidth=1.8, linestyle="-",  label=f"Mean = {mu:.2f}")

    ax.set_title(f"Residual Distribution  —  {model_name}",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_xlabel("Residual (W/m\u00b2)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Count (log scale)", fontsize=LABEL_SIZE, fontweight="bold")
    apply_axis_typography(ax)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="best", frameon=True, fontsize=LEGEND_SIZE)
    leg.get_frame().set_alpha(0.9)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    ax.text(0.97, 0.95, f"\u03bc = {mu:.2f}\n\u03c3 = {sigma:.2f}",
            transform=ax.transAxes, fontsize=LEGEND_SIZE, va="top", ha="right",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

    plt.tight_layout()
    fname = f"hist_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fname}")

# ── Combined 2×3 overview (3 plots per row) ──────────────────────────────────
print("\nSaving combined histogram overview …")
fig, axes = plt.subplots(2, 3, figsize=(30, 14),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.28})
axes = axes.flatten()
for ax, model_name in zip(axes, MODEL_ORDER):
    res   = residuals[model_name]["residual"]
    mu    = res.mean()
    sigma = res.std()
    ax.hist(res, bins=80, color=C_BAR, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.axvline(0,  color=C_ZERO,   linewidth=1.5, linestyle="--", label="Zero")
    ax.axvline(mu, color="#B71C1C", linewidth=1.5, linestyle="-", label=f"Mean = {mu:.2f}")
    ax.set_title(model_name, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Residual (W/m\u00b2)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Count (log scale)", fontsize=LABEL_SIZE, fontweight="bold")
    apply_axis_typography(ax)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="best", frameon=True, fontsize=LEGEND_SIZE)
    leg.get_frame().set_alpha(0.9)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    ax.text(0.97, 0.95, f"\u03bc={mu:.2f}  \u03c3={sigma:.2f}",
            transform=ax.transAxes, fontsize=LEGEND_SIZE, va="top", ha="right",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

fig.suptitle("Residual Histogram — All Models",
             fontsize=TITLE_SIZE, fontweight="bold", y=1.04)
fig.subplots_adjust(top=0.88)
plt.savefig(os.path.join(OUT_DIR, "hist_all_models.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved -> hist_all_models.png")

print("\nDone! Residual histograms saved to:", OUT_DIR)
