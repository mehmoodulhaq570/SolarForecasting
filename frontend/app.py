# ===============================================
# Solar Radiation Prediction Dashboard
# Streamlit Frontend with TFT & TCN Integration
# ===============================================

import streamlit as st
import os
import sys
import platform
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import frontend modules
from frontend.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    MODEL_COLORS,
    LOCATION_NAME,
    BASE_DIR,
)
from frontend.api import fetch_api_forecast, get_nasa_power_availability_note
from frontend.models import (
    load_traditional_models,
    load_keras_models,
    load_pytorch_models,
    load_scalers,
    TENSORFLOW_AVAILABLE,
    PYTORCH_AVAILABLE,
    XGBOOST_AVAILABLE,
)
from frontend.data import load_historical_data, get_data_summary
from frontend.prediction import PredictionEngine
from frontend.prediction import calibrate_ensemble_to_api
from frontend.visualization import (
    create_forecast_plots,
    create_deep_learning_comparison_plot,
    calculate_metrics,
    create_model_architecture_info,
)

# ============== Page Configuration ==============
st.set_page_config(page_title="Solar Radiation Forecast", page_icon="☀️", layout="wide")

# ============== Sidebar ==============
st.sidebar.image("https://img.icons8.com/fluency/96/solar-panel.png", width=80)
st.sidebar.title("☀️ Solar Power Prediction")
st.sidebar.markdown("---")

# Date input
selected_date = st.sidebar.date_input(
    "📅 Forecast Date",
    datetime.today(),
    help="Select the date for solar radiation forecast",
)

# Model folder paths
traditional_folder = os.path.join(BASE_DIR, "saved_models")
lstm_folder = os.path.join(BASE_DIR, "saved_models")  # LSTM models are in saved_models
tft_folder = os.path.join(BASE_DIR, "saved_models_tft")

# Model selection with availability indicators
st.sidebar.markdown("### 🤖 Model Selection")

# Check model availability - check both saved_models and saved_models_lstm
available_model_status = {
    "XGBoost": XGBOOST_AVAILABLE
    and os.path.exists(os.path.join(traditional_folder, "xgboost_model.pkl")),
    "Random Forest": os.path.exists(
        os.path.join(traditional_folder, "random_forest_model.pkl")
    ),
    "LSTM": TENSORFLOW_AVAILABLE
    and (
        os.path.exists(os.path.join(traditional_folder, "lstm_model.h5"))
        or os.path.exists(os.path.join(BASE_DIR, "saved_models_lstm", "lstm_model.h5"))
    ),
    "CNN-LSTM": TENSORFLOW_AVAILABLE
    and (
        os.path.exists(os.path.join(traditional_folder, "cnn_lstm_model.h5"))
        or os.path.exists(
            os.path.join(BASE_DIR, "saved_models_lstm", "cnn_lstm_model.h5")
        )
    ),
    "TFT": PYTORCH_AVAILABLE
    and (
        os.path.exists(os.path.join(tft_folder, "tft_model.pt"))
        or os.path.exists(os.path.join(tft_folder, "tft_model_best.ckpt"))
    ),
    "TCN": PYTORCH_AVAILABLE
    and (
        os.path.exists(os.path.join(tft_folder, "tcn_model.pt"))
        or os.path.exists(os.path.join(tft_folder, "tcn_model_best.ckpt"))
    ),
    "Ensemble": True,  # Always available
}

# Display model availability
with st.sidebar.expander("📊 Model Availability", expanded=False):
    for model, available in available_model_status.items():
        if model != "Ensemble":
            status = "✅" if available else "❌"
            st.write(f"{status} {model}")

    st.markdown("---")
    st.caption("Framework Status:")
    st.write(f"{'✅' if TENSORFLOW_AVAILABLE else '❌'} TensorFlow")
    st.write(f"{'✅' if PYTORCH_AVAILABLE else '❌'} PyTorch")
    st.write(f"{'✅' if XGBOOST_AVAILABLE else '❌'} XGBoost")
    st.caption(f"Python: {platform.python_version()}")
    if not TENSORFLOW_AVAILABLE and sys.version_info >= (3, 14):
        st.info("TensorFlow models require Python 3.13 or lower.")

# Filter available models for selection
selectable_models = [
    m for m in AVAILABLE_MODELS if available_model_status.get(m, False)
]
default_selection = [m for m in DEFAULT_MODELS if m in selectable_models]

selected_models = st.sidebar.multiselect(
    "Select Models",
    selectable_models,
    default=default_selection,
    help="Choose which models to use for prediction",
)

st.sidebar.markdown("---")

# API Comparison toggle
enable_api = st.sidebar.checkbox(
    "🌐 Compare with API",
    value=True,
    help="Fetch API forecast for comparison",
)

# API Selection
selected_api = st.sidebar.selectbox(
    "📡 Select API",
    ["NASA POWER", "Open-Meteo"],
    index=0,
    help="NASA POWER: Historical data (same source as training, ~1 week delay)\nOpen-Meteo: Real-time forecast (today + 16 days)",
    disabled=not enable_api,
)

# Show API info
if enable_api:
    if selected_api == "NASA POWER":
        st.sidebar.info(
            "🛰️ **NASA POWER**: Best accuracy (R²=0.96)\nUse dates ≥1 week old"
        )
    else:
        st.sidebar.info(
            "🌤️ **Open-Meteo**: Real-time forecast\nUse today or future dates"
        )

st.sidebar.markdown("---")

# Model info
st.sidebar.markdown("### 📊 Model Info")
with st.sidebar.expander("Model Details", expanded=False):
    model_info = create_model_architecture_info()
    for model, info in model_info.items():
        if model in selectable_models:
            st.markdown(f"**{model}**")
            st.caption(f"{info['type']}: {info['description']}")

# ============== Main Content ==============
st.markdown(
    "<h1 style='text-align:center;'>☀️ Solar Radiation Forecast Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:gray;'>Calibrated hourly predictions for {LOCATION_NAME} using ML + Deep Learning</p>",
    unsafe_allow_html=True,
)

# Framework status banner
if PYTORCH_AVAILABLE:
    framework_status = "🚀 TFT & TCN models available (PyTorch)"
else:
    framework_status = "⚠️ Install PyTorch for TFT & TCN support"

st.info(framework_status)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Prediction", "🔬 Model Comparison", "📋 Data Table", "ℹ️ About"]
)

# ============== Prediction Tab ==============
with tab1:
    col1, col2 = st.columns([3, 1])

    with col2:
        predict_button = st.button(
            "🔮 Generate Forecast", type="primary", use_container_width=True
        )

    with col1:
        st.markdown(f"**Selected Date:** {selected_date.strftime('%B %d, %Y')}")
        st.markdown(
            f"**Models:** {', '.join(selected_models) if selected_models else 'None selected'}"
        )

    if predict_button:
        if not selected_models:
            st.error("❌ Please select at least one model.")
            st.stop()

        # Progress bar
        progress = st.progress(0, text="Loading data...")

        # ---------- 1. Load Historical Data ----------
        try:
            df_day = load_historical_data()
            progress.progress(10, text="Data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()

        # ---------- 2. Load Models ----------
        progress.progress(20, text="Loading models...")

        loaded_models = {}

        # Load traditional models
        if any(m in selected_models for m in ["XGBoost", "Random Forest", "Ensemble"]):
            traditional_models = load_traditional_models(traditional_folder)
            loaded_models.update(traditional_models)

        # Load Keras models
        if any(m in selected_models for m in ["LSTM", "CNN-LSTM", "Ensemble"]):
            keras_models = load_keras_models(lstm_folder)
            loaded_models.update(keras_models)

        # Load PyTorch models (TFT, TCN)
        if any(m in selected_models for m in ["TFT", "TCN", "Ensemble"]):
            pytorch_models = load_pytorch_models(tft_folder)
            loaded_models.update(pytorch_models)

        # Load scalers (try TFT folder first, then traditional)
        scaler_X, scaler_y = load_scalers(tft_folder)
        if scaler_X is None:
            scaler_X, scaler_y = load_scalers(traditional_folder)

        if not loaded_models:
            st.error("❌ No models could be loaded. Check model folders.")
            st.stop()

        progress.progress(40, text=f"Loaded {len(loaded_models)} models!")

        # Display loaded models
        model_badges = " | ".join([f"✅ {m}" for m in loaded_models.keys()])
        st.success(f"Loaded: {model_badges}")

        # ---------- 3. Generate Predictions ----------
        progress.progress(50, text="Generating predictions...")

        year = selected_date.year
        month = selected_date.month
        day = selected_date.day

        # Fetch API data first to determine weights
        api_preds = None
        api_name = "API"
        use_openmeteo_weights = False

        if enable_api:
            progress.progress(60, text=f"Fetching {selected_api} forecast...")
            api_preds, api_name, use_openmeteo_weights = fetch_api_forecast(
                selected_api, year, month, day
            )

            if api_preds:
                st.sidebar.success(f"✅ {api_name} data fetched!")
                if api_name == "Open-Meteo (fallback)" and selected_api == "NASA POWER":
                    nasa_note = get_nasa_power_availability_note(year, month, day)
                    if nasa_note:
                        st.sidebar.warning(f"NASA POWER unavailable: {nasa_note}")
                    else:
                        st.sidebar.warning(
                            "NASA POWER did not return a usable 24-hour solar curve."
                        )
            else:
                st.sidebar.warning("⚠️ API data not available for this date")
        else:
            use_openmeteo_weights = True

        # Initialize prediction engine
        engine = PredictionEngine(
            models=loaded_models, scaler_X=scaler_X, scaler_y=scaler_y, df_day=df_day
        )

        progress.progress(70, text="Running model predictions...")

        # Generate predictions
        all_predictions = engine.predict_all(year, month, day, use_openmeteo_weights)

        if api_preds and "Ensemble" in all_predictions:
            all_predictions["Ensemble"] = calibrate_ensemble_to_api(
                all_predictions["Ensemble"], api_preds, use_openmeteo_weights
            )
            st.info(
                f"Applied API-guided ensemble calibration using {api_name} "
                "to reduce same-day bias and timing drift."
            )

        progress.progress(90, text="Creating visualizations...")

        # ---------- 4. Build Results DataFrame ----------
        date_index = pd.date_range(
            start=f"{year}-{month:02d}-{day:02d} 00:00",
            end=f"{year}-{month:02d}-{day:02d} 23:00",
            freq="h",
        )

        results_df = pd.DataFrame(
            {
                "datetime": date_index,
                "Hour": [f"{h:02d}:00" for h in range(24)],
            }
        )

        # Add model predictions
        for model_name, preds in all_predictions.items():
            if preds and model_name in selected_models:
                results_df[model_name] = preds

        # Add ensemble
        if "Ensemble" in selected_models:
            results_df["Ensemble"] = all_predictions.get("Ensemble", [0] * 24)

        # Add API predictions
        if api_preds:
            results_df[api_name] = api_preds

        progress.progress(100, text="Complete!")

        # ---------- 5. Display Results ----------
        st.success(f"✅ Forecast generated for {selected_date.strftime('%B %d, %Y')}")

        # Create and display plots
        ensemble_preds = all_predictions.get("Ensemble", [0] * 24)
        fig = create_forecast_plots(
            results_df=results_df,
            selected_models=selected_models,
            api_preds=api_preds,
            api_name=api_name,
            ensemble_preds=ensemble_preds,
            forecast_date=selected_date.strftime("%Y-%m-%d"),
        )
        st.pyplot(fig)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "🌅 Sunrise Peak", f"{max(ensemble_preds[5:9]):.0f} W/m²", "05:00-08:00"
            )
        with col2:
            st.metric(
                "☀️ Midday Peak", f"{max(ensemble_preds[10:14]):.0f} W/m²", "10:00-13:00"
            )
        with col3:
            st.metric(
                "🌇 Sunset", f"{max(ensemble_preds[16:19]):.0f} W/m²", "16:00-18:00"
            )
        with col4:
            daily_total = sum(ensemble_preds)
            st.metric("📊 Daily Total", f"{daily_total:.0f} Wh/m²", "Energy")

        # API Comparison Metrics
        if api_preds:
            st.markdown("---")
            st.subheader("📊 Model vs API Comparison Metrics")

            metrics_df = calculate_metrics(all_predictions, api_preds)

            if not metrics_df.empty:
                st.dataframe(
                    metrics_df.style.format(
                        {
                            "MAE (W/m²)": "{:.2f}",
                            "RMSE (W/m²)": "{:.2f}",
                            "R²": "{:.4f}",
                            "Bias (W/m²)": "{:.2f}",
                        }
                    )
                    .highlight_min(
                        subset=["MAE (W/m²)", "RMSE (W/m²)"], color="lightgreen"
                    )
                    .highlight_max(subset=["R²"], color="lightgreen"),
                    use_container_width=True,
                )

        # Store results for other tabs
        st.session_state["results_df"] = results_df
        st.session_state["all_predictions"] = all_predictions
        st.session_state["forecast_date"] = selected_date
        st.session_state["api_preds"] = api_preds
        st.session_state["api_name"] = api_name

# ============== Model Comparison Tab ==============
with tab2:
    st.subheader("🔬 Deep Learning Model Comparison")

    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        api_preds = st.session_state.get("api_preds")
        api_name = st.session_state.get("api_name", "API")

        # Create deep learning comparison plot
        fig = create_deep_learning_comparison_plot(results_df, api_preds, api_name)
        st.pyplot(fig)

        # Model architecture comparison
        st.markdown("---")
        st.subheader("🏗️ Model Architecture Comparison")

        model_info = create_model_architecture_info()

        cols = st.columns(3)
        for i, (model, info) in enumerate(model_info.items()):
            if model in ["TFT", "TCN", "LSTM", "CNN-LSTM", "Ensemble"]:
                with cols[i % 3]:
                    with st.expander(f"**{model}**", expanded=True):
                        st.markdown(f"**Type:** {info['type']}")
                        st.markdown(f"**Description:** {info['description']}")
                        st.markdown(f"**Features:** {info['features']}")
                        st.markdown(f"**Strength:** {info['strength']}")
    else:
        st.info("👆 Generate a forecast first to see model comparisons.")

# ============== Data Table Tab ==============
with tab3:
    if "results_df" in st.session_state:
        st.subheader(
            f"📋 Hourly Predictions for {st.session_state['forecast_date'].strftime('%B %d, %Y')}"
        )

        display_df = (
            st.session_state["results_df"].drop(columns=["datetime"]).set_index("Hour")
        )
        st.dataframe(
            display_df.style.format("{:.2f}").background_gradient(
                cmap="YlOrRd", axis=0
            ),
            use_container_width=True,
        )

        # Download button
        csv = st.session_state["results_df"].to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"solar_forecast_{st.session_state['forecast_date'].strftime('%Y_%m_%d')}.csv",
            "text/csv",
        )
    else:
        st.info("👆 Click 'Generate Forecast' to see predictions here.")

# ============== About Tab ==============
with tab4:
    st.subheader("ℹ️ About This Dashboard")
    st.markdown(
        f"""
    ### 🌤 Solar Radiation Forecasting System
    
    This dashboard uses machine learning models trained on **NASA POWER** meteorological data 
    from 2018-2025 to predict hourly solar radiation for **{LOCATION_NAME}**.
    
    #### 🤖 Models Used:
    
    **Traditional ML:**
    - **XGBoost**: Gradient boosting with decision trees
    - **Random Forest**: Bootstrap aggregated decision trees
    
    **Deep Learning (TensorFlow/Keras):**
    - **LSTM**: Bidirectional Long Short-Term Memory neural network
    - **CNN-LSTM**: Convolutional + LSTM hybrid architecture
    
    **Advanced Deep Learning (PyTorch):**
    - **TFT (Temporal Fusion Transformer)**: State-of-the-art attention-based model
    - **TCN (Temporal Convolutional Network)**: Dilated causal convolutions
    
    **Ensemble:**
    - **Calibrated Ensemble**: Weighted combination with hour-specific calibration
    
    #### 📊 Features Used:
    - ⏰ **Time**: Hour, Month, Day of Year (with cyclical encoding)
    - ☀️ **Solar**: Solar Zenith Angle, Clear Sky Radiation
    - 🌡️ **Weather**: Temperature, Humidity, Pressure, Wind
    - 📈 **Historical**: 24-hour lag values (tree models)
    - 🔄 **Sequences**: 24-step sequences (deep learning models)
    
    #### 🌐 API Sources:
    - **NASA POWER**: Prediction Of Worldwide Energy Resources (Historical)
    - **Open-Meteo**: Real-time weather forecasts
    
    #### 🆕 New in v2.0:
    - TFT (Temporal Fusion Transformer) integration
    - TCN (Temporal Convolutional Network) integration
    - Improved ensemble with 6 models
    - Deep learning comparison visualizations
    - Model availability status
    """
    )

    # Version info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Version", "2.0.0")
    with col2:
        st.metric(
            "Models",
            f"{len([m for m, a in available_model_status.items() if a])} Available",
        )
    with col3:
        st.metric("Location", LOCATION_NAME)
