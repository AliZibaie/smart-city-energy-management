import streamlit as st
import pandas as pd
import numpy as np
import pymysql
import joblib
import os
import sys
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import notebooks.db_helpers as dbh

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))

st.set_page_config(page_title="Smart City Energy Dashboard", layout="wide")

# -------------------------
# Data & Model Loading (cached)
# -------------------------
@st.cache_resource
def get_connection():
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, port=DB_PORT, charset='utf8mb4'
    )

@st.cache_resource
def load_models():
    model_path = Path(__file__).parent.parent / "models"

    forecast_model = joblib.load(f"{model_path}/forecast_model.joblib")
    anomaly_model = joblib.load(f"{model_path}/anomaly_detector.joblib")
    return forecast_model, anomaly_model

@st.cache_data(ttl=600)
def load_city_data():
    conn = get_connection()
    df = dbh.get_city_hourly_consumption(conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

@st.cache_data(ttl=600)
def load_household_data():
    conn = get_connection()
    df = dbh.get_household_consumption_with_meta(conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# -------------------------
# Feature engineering (same logic as Phase 2)
# -------------------------
def build_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    lags = [1, 2, 3, 24, 48, 168]
    for lag in lags:
        df[f'lag_{lag}'] = df['total_consumption'].shift(lag)

    df['rolling_mean_24'] = df['total_consumption'].shift(1).rolling(24).mean()
    df['rolling_std_24'] = df['total_consumption'].shift(1).rolling(24).std()

    df['target'] = df['total_consumption'].shift(-24)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = ['temperature_c', 'humidity_percent', 'hour_sin', 'hour_cos',
                'dow_sin', 'dow_cos', 'is_weekend',
                'lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 'lag_168',
                'rolling_mean_24', 'rolling_std_24']

ANOMALY_FEATURES = ['total_consumption', 'temperature_c', 'humidity_percent', 'hour', 'day_of_week']


# -------------------------
# Load everything
# -------------------------
forecast_model, anomaly_model = load_models()
city_df_raw = load_city_data()
household_df = load_household_data()

featured_df = build_features(city_df_raw)
X_all = featured_df[FEATURE_COLS]
y_all = featured_df['target']
y_pred_all = forecast_model.predict(X_all)
featured_df['predicted'] = y_pred_all

# anomaly detection on raw data
anomaly_input_df = city_df_raw.copy()
anomaly_input_df['hour'] = anomaly_input_df['timestamp'].dt.hour
anomaly_input_df['day_of_week'] = anomaly_input_df['timestamp'].dt.dayofweek
X_anomaly_all = anomaly_input_df[ANOMALY_FEATURES]
anomaly_input_df['anomaly_score'] = anomaly_model.decision_function(X_anomaly_all)
anomaly_input_df['is_anomaly'] = anomaly_model.predict(X_anomaly_all)
anomaly_input_df['is_anomaly'] = anomaly_input_df['is_anomaly'].map({1: 0, -1: 1})


# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Smart City Energy Dashboard")
page = st.sidebar.radio(
    "Select a section",
    ["Overview & Forecasting", "Consumption Heatmap", "Anomaly Alerts", "Model Interpretation (XAI)"]
)


# =========================
# PAGE 1: Time series forecast
# =========================
if page == "Overview & Forecasting":
    st.header("Actual vs Predicted Consumption")

    n_hours = st.slider("Number of hours to display (from end of period)", 24, 24 * 30, 24 * 7)
    plot_df = featured_df.tail(n_hours)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df['timestamp'], plot_df['target'], label='Actual', color='steelblue')
    ax.plot(plot_df['timestamp'], plot_df['predicted'], label='Predicted', color='orange', linestyle='--')
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Consumption (kWh)")
    ax.legend()
    ax.set_title("Actual vs Predicted Energy Consumption (24h ahead)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    col1, col2, col3, col4 = st.columns(4)
    mae = np.mean(np.abs(plot_df['target'] - plot_df['predicted']))
    rmse = np.sqrt(np.mean((plot_df['target'] - plot_df['predicted'])**2))
    mape = np.mean(np.abs((plot_df['target'] - plot_df['predicted']) / plot_df['target'])) * 100
    r2 = 1 - np.sum((plot_df['target'] - plot_df['predicted'])**2) / np.sum((plot_df['target'] - plot_df['target'].mean())**2)

    col1.metric("MAE", f"{mae:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")
    col3.metric("MAPE (%)", f"{mape:.2f}")
    col4.metric("R²", f"{r2:.3f}")


# =========================
# PAGE 2: Heatmap
# =========================
elif page == "Consumption Heatmap":
    st.header("Energy Consumption Heatmap")

    heatmap_source = st.radio("Data source", ["Hour of day × Day of week (City-wide)", "Households"])

    if heatmap_source == "Hour of day × Day of week (City-wide)":
        df = city_df_raw.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        pivot = df.pivot_table(
            values='total_consumption', index='day_of_week',
            columns='hour', aggfunc='mean'
        )
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        pivot.index = [day_labels[i] for i in pivot.index]

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        ax.set_title("Average Consumption Heatmap (Hour x Day)")
        plt.colorbar(im, ax=ax, label="kWh")
        st.pyplot(fig)

    else:
        agg_household = household_df.groupby('household_id')['consumption_kwh'].mean().reset_index()
        agg_household = agg_household.sort_values('consumption_kwh', ascending=False)

        n_grid = int(np.ceil(np.sqrt(len(agg_household))))
        grid = np.zeros((n_grid, n_grid))
        values = agg_household['consumption_kwh'].values
        grid.flat[:len(values)] = values

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(grid, cmap='YlOrRd')
        ax.set_title("Household Energy Consumption Heatmap (simulated grid layout)")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        plt.colorbar(im, ax=ax, label="Avg kWh")
        st.pyplot(fig)

        st.dataframe(agg_household.head(20))


# =========================
# PAGE 3: Anomaly alerts
# =========================
elif page == "Anomaly Alerts":
    st.header("Detected Anomaly Alerts")

    anomalies = anomaly_input_df[anomaly_input_df['is_anomaly'] == 1].copy()
    anomalies = anomalies.sort_values('timestamp', ascending=False)

    def classify_anomaly(row, mean_val, std_val):
        if row['total_consumption'] > mean_val + std_val:
            return "High abnormal consumption (Spike)"
        elif row['total_consumption'] < mean_val - std_val:
            return "Unusual drop (Drop)"
        else:
            return "Abnormal pattern (Pattern)"

    mean_val = city_df_raw['total_consumption'].mean()
    std_val = city_df_raw['total_consumption'].std()
    anomalies['anomaly_type'] = anomalies.apply(
        lambda r: classify_anomaly(r, mean_val, std_val), axis=1
    )

    st.metric("Total number of detected anomalies", len(anomalies))

    st.dataframe(
        anomalies[['timestamp', 'total_consumption', 'temperature_c',
                   'humidity_percent', 'anomaly_score', 'anomaly_type']],
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(city_df_raw['timestamp'], city_df_raw['total_consumption'], alpha=0.6, label='Consumption')
    ax.scatter(anomalies['timestamp'], anomalies['total_consumption'], color='red', label='Anomaly', zorder=5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Consumption (kWh)")
    ax.legend()
    ax.set_title("Detected Anomalies Over Time")
    st.pyplot(fig)


# =========================
# PAGE 4: XAI Panel
# =========================
elif page == "Model Interpretation (XAI)":
    st.header("Model Interpretation Panel (XAI)")

    sample_size = st.slider("Number of samples for SHAP calculation", 50, 500, 200, step=50)

    with st.spinner("Calculating SHAP values..."):
        X_sample = X_all.sample(n=min(sample_size, len(X_all)), random_state=42)
        explainer = shap.TreeExplainer(forecast_model)
        shap_values = explainer.shap_values(X_sample)

    st.subheader("Overall Feature Importance")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_COLS, plot_type='bar', show=False)
    st.pyplot(fig1)

    st.subheader("Feature Impact Distribution (Summary Plot)")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_COLS, show=False)
    st.pyplot(fig2)

    st.subheader("Single Sample Analysis (Waterfall)")
    idx = st.number_input("Sample index", min_value=0, max_value=len(X_sample) - 1, value=0)
    fig3 = plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_sample.iloc[idx],
            feature_names=FEATURE_COLS
        ),
        show=False
    )
    st.pyplot(fig3)