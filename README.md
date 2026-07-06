# Smart City Energy Management System

A comprehensive end‑to‑end AI solution for monitoring, forecasting, and securing energy consumption in a smart city.  
This project covers the full lifecycle of a data‑driven system – from synthetic data generation and database management to machine learning forecasting, anomaly detection, model explainability (XAI), adversarial attack simulation, and an interactive dashboard.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Dashboard Preview](#dashboard-preview)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Setup Instructions](#setup-instructions)
   - [Environment Variables](#environment-variables)
   - [Database Setup](#database-setup)
   - [Virtual Environment & Dependencies](#virtual-environment--dependencies)
   - [Using the CLI](#using-the-cli)
7. [Phase‑by‑Phase Walkthrough](#phasebyphase-walkthrough)
   - [Phase 1 – Data Generation & Database](#phase-1--data-generation--database)
   - [Phase 2 – Forecasting, Anomaly Detection & XAI](#phase-2--forecasting-anomaly-detection--xai)
   - [Phase 3 – Interactive Dashboard](#phase-3--interactive-dashboard)
   - [Phase 4 – Adversarial Attack Simulation](#phase-4--adversarial-attack-simulation)
8. [Results and Interpretations](#results-and-interpretations)
   - [Phase 1 – Exploratory Data Analysis](#phase-1--exploratory-data-analysis)
   - [Phase 2 – Model Performance & Explainability](#phase-2--model-performance--explainability)
   - [Phase 4 – Attack Impact & Robustness](#phase-4--attack-impact--robustness)
9. [Strengths and Weaknesses](#strengths-and-weaknesses)
10. [Ethical Considerations & Privacy](#ethical-considerations--privacy)
11. [Conclusion](#conclusion)
12. [Future Work](#future-work)

---

## Introduction

The goal of this project is to simulate, analyse, and manage energy consumption in a smart city using modern Python libraries.  
It demonstrates the complete cycle of a data science project:

- **Data generation** – realistic hourly consumption patterns with seasonal, daily, and household factors.
- **Data governance** – storing data in a relational database (MySQL) with differential privacy applied to resident counts.
- **Forecasting** – predicting city‑wide consumption 24 hours ahead with a Random Forest model.
- **Anomaly detection** – identifying unusual consumption events using Isolation Forest.
- **Explainability** – interpreting model decisions with SHAP (feature importance, summary, and waterfall plots).
- **Adversarial robustness** – simulating cyber‑attacks on the forecasting model and evaluating performance degradation.
- **Dashboard** – a Streamlit web app for real‑time monitoring and decision support.

---

## Key Features

- **Realistic synthetic data** – 100 households, 8760 hourly records, incorporating weather, weekend, and seasonal effects.
- **SQL database** – well‑structured tables with foreign keys and analytical queries.
- **Forecast model** – Random Forest Regressor with **R² = 0.983**, **MAPE = 3.73%** on the test set.
- **Anomaly detector** – Isolation Forest that flags 176 anomalies in normal data and detects 40% of injected spikes.
- **SHAP explanations** – identifies `lag_168`, `lag_48`, and `dow_cos` as the most influential features.
- **Adversarial attack analysis** – quantifies the model’s vulnerability to lag‑spike attacks (R² drops to ‑1.03) while remaining robust to temperature sensor tampering.
- **Interactive dashboard** – built with Streamlit, offering:
  - Actual vs predicted consumption plots
  - Consumption heatmaps (time‑of‑day × day‑of‑week)
  - Anomaly alerts with classification
  - XAI panel showing SHAP values
- **CLI utility** – `smartcity.py` simplifies environment setup, database creation, notebook execution, and dashboard launch.

---
## Dashboard Preview

Watch the interactive Streamlit dashboard in action, showing real-time forecasts, anomaly alerts, and model explanations.

<video src="https://github.com/user-attachments/assets/dashboard-demo.mp4" controls width="850" autoplay></video>

You can also run the dashboard locally using `python smartcity.py streamlit` **after** completing Phase 1 (data generation and database insertion). The dashboard reads directly from the database, so it requires the generated data to be present.

## Project Structure

```
.
├── .env                     # Environment variables (created from .env.example)
├── .env.example             # Template for database credentials
├── .gitignore
├── requirements.txt         # Python dependencies
├── smartcity.py             # CLI helper script
├── dashboard/
│   └── app.py               # Streamlit dashboard application
├── data/
│   ├── schema.sql           # SQL table definitions
│   └── generated-dataset/   # Exported Excel files (optional)
├── models/
│   ├── forecast_model.joblib
│   └── anomaly_detector.joblib
└── notebooks/
    ├── db_helpers.py        # Database utility functions
    ├── Phase1_Data_Generation_DB.ipynb
    ├── Phase2_Modeling_Forecast_Anomaly.ipynb
    ├── Phase4_Adversarial_Attack.ipynb
    └── results/             # All generated plots (PNG)
        ├── Actual_vs_Predicted.png
        ├── adversarial_robustness.png
        ├── attack_metrics_comparison.png
        ├── consumption_by_activity_pattern.png
        ├── consumption_by_house_type.png
        ├── consumption_by_income.png
        ├── consumption_outliers.png
        ├── correlation_humidity_consumption.png
        ├── correlation_temp_consumption.png
        ├── daily_profile.png
        ├── Detected_Anomalies.png
        ├── monthly_avg_consumption.png
        ├── seasonal_profile.png
        ├── shap_bar_summary.png
        ├── shap_summary.png
        ├── top_consuming_households.png
        ├── top_peak_hours.png
        ├── waterfall_plot.png
        └── weekday_vs_weekend.png
```

---

## Prerequisites

- **Python 3.9+** (recommended: use a virtual environment)
- **MySQL** (local or remote) – the project uses MySQL but can be adapted to SQLite.
- **pip** and basic command‑line familiarity.
- Optional: **Git** to clone the repository.

---

## Setup Instructions

### 1. Environment Variables

Copy the example environment file and adjust the database credentials:

```bash
python smartcity.py env
```

This creates `.env` from `.env.example`. Edit it to set your MySQL host, user, password, database name, and port.

**Example `.env` content:**

```ini
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=smartcity
DB_PORT=3306
EPSILON=1.0          # for differential privacy
RANDOM_SEED=42
```

### 2. Database Setup

Make sure your MySQL server is running. Then create the database:

```bash
python smartcity.py db:create
```

This will create a database named `smartcity` (or the name you set in `.env`) if it does not already exist.

### 3. Virtual Environment & Dependencies

Create a virtual environment and install all required packages:

```bash
python smartcity.py setup
```

This does:
- Creates `.venv`
- Upgrades pip
- Installs all packages listed in `requirements.txt`

**Manual alternative** (if you prefer):

```bash
python -m venv .venv
# Activate it (see below)
pip install -r requirements.txt
```

Activate the virtual environment:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 4. Using the CLI

A dedicated CLI `smartcity.py` is provided to orchestrate the entire project.  
After setting up the environment, you can use the following commands:

| Command | Description |
|---------|-------------|
| `python smartcity.py setup` | Create virtual environment and install dependencies |
| `python smartcity.py env` | Copy `.env.example` to `.env` |
| `python smartcity.py db:create` | Create the database |
| `python smartcity.py notebook <phase>` | Open the Jupyter notebook for a specific phase (1, 2, or 4) |
| `python smartcity.py streamlit` | Launch the Streamlit dashboard |
| `python smartcity.py pipeline` | Run the full pipeline in order: `.env` → setup → DB → Phase1 → Phase2 → Phase4 |

> **Note:** The pipeline command will open each notebook interactively – you need to run the cells and close the notebook to proceed to the next phase.

---

## Phase‑by‑Phase Walkthrough

### Phase 1 – Data Generation & Database

**Notebook:** `Phase1_Data_Generation_DB.ipynb`

**What it does:**
- Generates synthetic data for 100 households over one year (hourly resolution).
- Applies **differential privacy** (Laplace mechanism) to the number of residents before storing – ensuring privacy of residents’ data.
- Creates two MySQL tables: `households` and `energy_consumption`.
- Inserts all data into the database.
- Executes a variety of SQL queries to produce summary statistics and visualisations:
  - Monthly averages
  - Weekday vs weekend consumption
  - Correlation with temperature and humidity
  - Top peak hours, top households, outliers, etc.
- Exports data to Excel for backup.

**How to run:**
```bash
python smartcity.py notebook 1
```
Or manually open the notebook in Jupyter.

---

### Phase 2 – Forecasting, Anomaly Detection & XAI

**Notebook:** `Phase2_Modeling_Forecast_Anomaly.ipynb`

**What it does:**
- Aggregates consumption to city‑wide hourly totals.
- Engineers features: cyclical encoding of hour and day‑of‑week, lag features (1, 2, 3, 24, 48, 168 hours), rolling mean and standard deviation (24‑hour window).
- Splits data chronologically (80% train, 20% test).
- Trains a **Random Forest Regressor** to predict 24‑hour ahead consumption.
- Evaluates the model using MAE, RMSE, R², and MAPE.
- Visualises actual vs predicted for the first week of the test set.
- Trains an **Isolation Forest** detector on consumption, temperature, humidity, hour, and day‑of‑week.
- Injects 20 artificial anomalies (10 spikes, 10 drops) to test detection performance.
- Computes **SHAP** values for the forecast model and produces:
  - Bar summary (average absolute SHAP)
  - Summary plot (feature impact distribution)
  - Waterfall plot for a single prediction
- Exports both models to `models/` as joblib files.

**How to run:**
```bash
python smartcity.py notebook 2
```

---

### Phase 3 – Interactive Dashboard

**File:** `dashboard/app.py`

**What it does:**
- Provides a web‑based interface for exploring the system.
- **Pages:**
  1. **Overview & Forecasting** – shows actual vs predicted plots with adjustable time window; displays key metrics.
  2. **Consumption Heatmap** – allows switching between a time‑of‑day × day‑of‑week heatmap and a household grid heatmap.
  3. **Anomaly Alerts** – lists detected anomalies with type classification (spike/drop/pattern) and visualises them on a timeline.
  4. **Model Interpretation (XAI)** – computes SHAP values on‑the‑fly for a selected sample and shows bar, summary, and waterfall plots.

The dashboard is cached for performance and connects to the same MySQL database and models.

**How to run:**
```bash
python smartcity.py streamlit
```
The dashboard will open in your default browser.

---

### Phase 4 – Adversarial Attack Simulation

**Notebook:** `Phase4_Adversarial_Attack.ipynb`

**What it does:**
- Loads the trained forecast model and the test set.
- Defines three attack scenarios on the test feature matrix:
  1. **Lag Spike** – multiply a random subset of lag features by a large factor (3–6×).
  2. **Temperature Shift** – add a large positive offset (15–30°C) to a random subset of temperature readings.
  3. **Combined** – a mix of both.
- Compares model performance (MAE, RMSE, R², MAPE) under each attack.
- Visualises the predicted vs true consumption under attack, and plots a bar‑line chart of RMSE and MAPE across attacks.
- Also evaluates the anomaly detector on the same attack‑injected data to show its potential as a defence layer.

**How to run:**
```bash
python smartcity.py notebook 4
```

---

## Results and Interpretations

### Phase 1 – Exploratory Data Analysis

The SQL queries and plots revealed the following insights:

#### Monthly Average Consumption
- Consumption peaks in **June** (1.45 kWh) and is lowest in **December** (0.64 kWh), reflecting higher cooling demand in summer and lower heating in winter (temperatures reach 34°C in summer and 5°C in winter).

#### Weekday vs Weekend
- Weekend consumption is **15% higher** (1.15 vs 1.00 kWh), likely due to residents being at home.

#### Correlation with Weather
- Temperature vs consumption: **0.47** – moderate positive correlation (warmer weather drives more cooling).
- Humidity vs consumption: **‑0.22** – weak negative correlation (high humidity may reduce cooling efficiency? Actually negative indicates consumption slightly decreases with higher humidity).

#### Peak Hours
- Top 5 peak hours: **18:00–21:00** (1.64 kWh) and **22:00** (1.17 kWh) – evening hours dominate.

#### Consumption by House Type
- **Villas** consume **56% more** than apartments (1.39 vs 0.89 kWh), as expected.

#### Consumption by Income
- Higher income households consume slightly more (1.08 vs 0.96 for low income) – but the difference is modest.

#### Activity Pattern
- **Mixed** patterns (work from home? or irregular) have the highest consumption (1.10 kWh), followed by Night shift (1.06) and Day shift (1.02).

#### Outliers
- Households **75** and **92** show the highest average consumption and variance, indicating either large families or high‑usage appliances.

All these insights are valuable for urban energy planning and demand response.

---

### Phase 2 – Model Performance & Explainability

**Forecast Model Metrics (on test set):**
- **MAE:** 2.2865 kWh  
- **RMSE:** 2.8232 kWh  
- **R²:** 0.9831  
- **MAPE:** 3.73%

These numbers indicate excellent predictive accuracy – the model captures the complex temporal patterns very well.

#### SHAP Analysis
- **Bar summary** (average absolute SHAP) shows that **`lag_168`** (same hour, 7 days ago) is the most influential feature, followed by **`lag_48`** and **`dow_cos`** (day‑of‑week cosine).  
  This highlights the strong weekly seasonality in consumption.
- The **summary plot** reveals that high values of `lag_168` increase the prediction (positive SHAP) while low values decrease it – consistent with weekly patterns.
- The **waterfall plot** for a single sample demonstrates how each feature pushes the prediction up or down from the base value.

These explanations build trust and help operators understand what drives the forecast.

#### Anomaly Detection
- On normal data, the Isolation Forest flagged **176** anomalies (2% of data, matching the contamination parameter).
- After injecting 20 artificial anomalies (10 spikes and 10 zero drops), the model detected **8** of them – a **40% detection rate**.  
  While not perfect, the detector successfully identifies many extreme events. The false positives can be fine‑tuned.

---

### Phase 4 – Attack Impact & Robustness

The adversarial attack results are summarised in the table and plots below.

| Attack Type       | MAE    | RMSE   | R²      | MAPE (%) | Extreme Errors |
|-------------------|--------|--------|---------|----------|----------------|
| Baseline (no attack) | 2.29   | 2.82   | 0.983   | 3.73     | –              |
| Lag Spike         | 8.95   | 30.97  | **‑1.03**| 14.33    | 115 / 1714     |
| Temperature Shift | 2.33   | 2.90   | 0.982   | 3.85     | 39 / 1714      |
| Combined          | 6.42   | 24.88  | **‑0.31**| 10.34    | 82 / 1714      |

**Key findings:**

1. **Lag Spike is devastating** – R² becomes negative (worse than a mean predictor), RMSE explodes to 31.0 (10× baseline). The model relies heavily on historical consumption (lag features), so corrupting them derails the forecast completely.

2. **Temperature Shift is barely noticeable** – the model is robust to sensor tampering because other features (lags, time patterns) compensate.

3. **Combined attack** – although less severe than pure lag spikes, it still damages performance significantly, showing that even a small percentage of manipulated lag records (3%) can cause major errors.

4. **Anomaly detector as defence** – when the same extreme consumption values were injected into the test set, the Isolation Forest flagged 49 out of 50 injected spikes, demonstrating its potential as a complementary security layer.

**Implication:**  
To secure the system, the highest priority is protecting the integrity of the historical consumption database. Data validation, anomaly filtering before feeding the model, and regular checks for sudden changes are essential.

---

## Strengths and Weaknesses

### Strengths
- **Comprehensive pipeline** – covers data generation, storage, modelling, XAI, security, and visualisation.
- **High forecasting accuracy** (R²=0.98) with a simple, interpretable model.
- **Explainability** – SHAP provides clear insights into model behaviour.
- **Privacy** – differential privacy is applied to resident counts, respecting user privacy.
- **Modular design** – clearly separated phases and reusable helper functions.
- **CLI automation** – makes the project easy to set up and run step by step.

### Weaknesses / Limitations
- **Anomaly detection rate** – 40% detection of injected anomalies suggests room for improvement (e.g., using a more sophisticated method or tuning contamination).
- **Adversarial vulnerability** – the forecasting model is critically exposed to lag‑feature manipulation. In a real system, stronger defences (e.g., input validation, robust regression) would be needed.
- **Synthetic data** – although realistic, the data is not real. Generalisation to actual smart city data may vary.
- **Scalability** – currently handles 100 households and 8760 hours; a larger city would require distributed processing.
- **Weather data** – assumes city‑wide uniform weather; in reality, micro‑climates could affect consumption.

---

## Ethical Considerations & Privacy

- **Differential Privacy** – the number of residents per household is obfuscated by adding Laplace noise (ε = 1.0). This prevents re‑identification while still allowing aggregate analysis.
- **Data minimisation** – only necessary attributes are collected; sensitive fields like exact addresses are avoided.
- **Transparency** – the dashboard’s XAI panel makes model decisions understandable to operators and the public.
- **Security** – the adversarial analysis highlights risks that real‑world systems must address to prevent cyber‑attacks that could disrupt energy distribution.

---

## Conclusion

This project successfully demonstrates an integrated AI‑driven energy management system for a smart city.  
From generating realistic consumption data to deploying an interactive dashboard, we have covered every stage of a data‑centric project.  

The forecasting model achieves excellent accuracy, while SHAP analysis reveals the dominant role of historical weekly patterns. The anomaly detector can flag suspicious events, though it misses a portion of injected attacks. The adversarial attack simulations expose a critical weakness: the model is highly sensitive to manipulation of past consumption records. This underscores the need for robust data integrity measures in production.

Overall, the system provides a solid foundation for real‑world deployment, with clear pathways for improvement in security and anomaly detection.

---

## Future Work

- **Improve anomaly detection** – try algorithms like LSTM‑Autoencoder or ensemble methods to increase detection rate.
- **Defend against lag attacks** – implement robust regression or add adversarial training.
- **Integrate real‑time data** – connect to IoT sensors and update forecasts every hour.
- **Scale up** – use Apache Spark or Dask for larger datasets.
- **Add more features** – e.g., holidays, building characteristics, solar generation.
- **Deploy the dashboard** – containerise with Docker and host on a cloud platform.

---

## Getting Help

For any issues, please check the `.env` configuration and ensure your MySQL server is running.  
You can also run each notebook interactively to debug step by step.  
The CLI commands (`python smartcity.py --help`) provide a quick reference.

---

*This project is part of an academic exercise in smart city analytics and AI security.*