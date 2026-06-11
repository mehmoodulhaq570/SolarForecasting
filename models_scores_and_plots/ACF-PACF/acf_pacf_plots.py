# ============================================================
# ACF & PACF Plots — All Input Parameters
# Data: lahore_hourly_filled.csv
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "NASA meteriological and solar radiaton data",
    "lahore_hourly_filled.csv",
)
OUT_DIR = SCRIPT_DIR

# ── Style ────────────────────────────────────────────────────
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   15,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
})

COLOR_ACF  = "#1f77b4"   # blue
COLOR_PACF = "#d62728"   # red
COLOR_CI   = "#aec7e8"   # light-blue shading


def style_axes(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

# ── Load data ────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
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

PARAMETERS = [
    "SolarRadiation",
    "ClearSkyRadiation",
    "DirectRadiation",
    "DiffuseRadiation",
    "SolarZenith",
    "Temperature",
    "HumiditySpecific",
    "HumidityRelative",
    "Pressure",
    "WindSpeed",
    "WindDirection",
]

# ── Settings ─────────────────────────────────────────────────
LAGS        = 150    # 150 hourly lags
SAMPLE_SIZE = 8760   # 1 year — reliable ACF, fast rendering
ALPHA       = 0.05   # 95 % confidence interval

print(f"Parameters  : {len(PARAMETERS)}")
print(f"Lags shown  : {LAGS}")
print(f"Sample size : {SAMPLE_SIZE} hours")

# ============================================================
# 1.  Individual ACF + PACF per parameter  (2-panel figure)
# ============================================================
print("\nGenerating individual ACF/PACF plots …")
for param in PARAMETERS:
    series = df[param].dropna()
    if len(series) > SAMPLE_SIZE:
        series = series.iloc[:SAMPLE_SIZE]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    fig.suptitle(
        f"ACF & PACF  —  {param}",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
    )

    # ACF ─────────────────────────────────────────────────────
    plot_acf(
        series, lags=LAGS, alpha=ALPHA,
        ax=ax1, color=COLOR_ACF, vlines_kwargs={"colors": COLOR_ACF},
    )
    ax1.set_title("Autocorrelation Function (ACF)", fontsize=TITLE_SIZE, fontweight="bold")
    ax1.set_xlabel("Lag (hours)", fontsize=LABEL_SIZE, fontweight="bold")
    ax1.set_ylabel(param, fontsize=LABEL_SIZE, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=TICK_SIZE)
    style_axes(ax1)
    ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.set_axisbelow(True)

    # PACF ────────────────────────────────────────────────────
    plot_pacf(
        series, lags=LAGS, alpha=ALPHA, method="ywm",
        ax=ax2, color=COLOR_PACF, vlines_kwargs={"colors": COLOR_PACF},
    )
    ax2.set_title("Partial Autocorrelation Function (PACF)", fontsize=TITLE_SIZE, fontweight="bold")
    ax2.set_xlabel("Lag (hours)", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.set_ylabel(param, fontsize=LABEL_SIZE, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=TICK_SIZE)
    style_axes(ax2)
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fname = f"acf_pacf_{param.lower()}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")

# ============================================================
# 2.  Combined overview — ACF only (all 11 parameters, 3×4 grid)
# ============================================================
print("\nGenerating combined ACF overview …")
n_cols = 4
n_rows = int(np.ceil(len(PARAMETERS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, n_rows * 4.5), sharex=False)
axes = axes.flatten()

for ax, param in zip(axes, PARAMETERS):
    series = df[param].dropna()
    if len(series) > SAMPLE_SIZE:
        series = series.iloc[:SAMPLE_SIZE]
    plot_acf(series, lags=LAGS, alpha=ALPHA, ax=ax,
             color=COLOR_ACF, vlines_kwargs={"colors": COLOR_ACF})
    ax.set_title(param, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Lag (hours)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(param, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    style_axes(ax)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)

# Hide any spare axes
for ax in axes[len(PARAMETERS):]:
    ax.set_visible(False)

fig.suptitle(
    "Autocorrelation Function (ACF) — All Input Parameters",
    fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "acf_all_parameters.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → acf_all_parameters.png")

# ============================================================
# 3.  Combined overview — PACF only (all 11 parameters, 3×4 grid)
# ============================================================
print("\nGenerating combined PACF overview …")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, n_rows * 4.5), sharex=False)
axes = axes.flatten()

for ax, param in zip(axes, PARAMETERS):
    series = df[param].dropna()
    if len(series) > SAMPLE_SIZE:
        series = series.iloc[:SAMPLE_SIZE]
    plot_pacf(series, lags=LAGS, alpha=ALPHA, method="ywm", ax=ax,
              color=COLOR_PACF, vlines_kwargs={"colors": COLOR_PACF})
    ax.set_title(param, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Lag (hours)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(param, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    style_axes(ax)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)

for ax in axes[len(PARAMETERS):]:
    ax.set_visible(False)

fig.suptitle(
    "Partial Autocorrelation Function (PACF) — All Input Parameters",
    fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pacf_all_parameters.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → pacf_all_parameters.png")

print("\nDone! All ACF/PACF plots saved to:", OUT_DIR)
