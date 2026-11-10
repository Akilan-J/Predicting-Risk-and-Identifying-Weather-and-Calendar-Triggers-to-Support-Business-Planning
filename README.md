# Early Warning for Severe Air-Pollution Days in Delhi-NCR

A Business Analytics project that forecasts high-pollution days in
Delhi-NCR and identifies weather and calendar conditions associated with
them, to support business planning.

## Project Overview

The project provides 1--3 day PM2.5 forecasts and high-risk alerts. It
also identifies conditions associated with high-pollution days and
presents the results through an interactive dashboard.

The intended users include construction firms, logistics and delivery
teams, schools, and event organisers.

## Objectives

-   Forecast PM2.5 levels 1, 2, and 3 days ahead.
-   Flag high-risk days using the threshold **PM2.5 ≥ 120.5 µg/m³**
    (equivalent to AQI \> 300 in a PM2.5-only calculation).
-   Identify weather and calendar patterns associated with high-risk
    days.
-   Compare false alarms with missed high-pollution days to support
    alert-threshold selection.
-   Present forecasts and insights in an interactive dashboard.

## Data

The project uses:

-   **OpenAQ v3:** Daily PM2.5 and PM10 measurements from monitoring
    stations within 25 km of central Delhi.
-   **Open-Meteo Archive:** Hourly weather observations summarised into
    daily values.
-   **Calendar data:** Season, weekday, and Diwali-related features.

The final dataset contains **3,195 daily records and 31 variables**,
covering **1 January 2018 to 30 September 2026**. The data is split
chronologically: 2018--2024 for training and January 2025--September
2026 for testing.

## Methods

### Predictive Modelling

Logistic Regression, Random Forest, and Gradient Boosting were evaluated
for high-risk classification. A persistence baseline ("tomorrow =
today") was used for comparison.

### Time-Series Forecasting

PM2.5 is forecast 1, 2, and 3 days ahead. The selected models are
Gradient Boosting for Day 1 and SARIMAX for Days 2 and 3.

### Association Rule Mining

FP-Growth is used to identify weather and calendar combinations
associated with a high-risk PM2.5 day on the following day.

## Key Results

-   The selected time-series approach achieved MAE values of **15.7,
    19.5, and 19.6 µg/m³** for Days 1, 2, and 3, respectively.
-   The corresponding persistence baseline MAE values were **18.8, 24.7,
    and 27.5 µg/m³**.
-   The selected forecasting approach achieved approximately **91--92%
    alert recall**.
-   Association Rule Mining produced **113 rules**. These describe
    associations, not causal relationships.

## Dashboard

The **Delhi Air-Risk Planner** is built using Streamlit and Plotly. It
contains six pages:

-   Overview
-   Dataset
-   Forecast
-   Rules
-   Insights
-   My Business

**Dashboard:** https://delhi-air-risk-planner.streamlit.app/

To run the application locally:

``` bash
pip install -r requirements.txt
cd dashboard
streamlit run app.py
```

The dashboard replays historical data; it is not a live data feed. Model
alerts are planning signals and do not declare official GRAP stages.

## Limitations

-   The target is based on one reference monitoring station, with a
    city-average fallback on 87 days.
-   272 days have no PM2.5 target and are not scored.
-   The held-out test period covers 21 months.
-   Association rules indicate relationships, not causation.
-   The dashboard currently uses historical replay rather than live
    data.

## Repository structure

| Path | Contents |
|---|---|
| `Final_Project_Report.pdf` | Final project report |
| `Final_Project_Presentation.pptx` | Final presentation |
| `requirements.txt` | Python dependencies for the notebooks and the dashboard |
| `data/raw/` | Original data: OpenAQ sensor-days and locations, Open-Meteo weather, calendar |
| `data/merged_daily_clean_m2.csv` | Cleaned and merged daily data (output of the cleaning notebook) |
| `data/cleaned_final.csv` | Final dataset: 3,195 days × 31 variables with the target |
| `data/M2_cleaning_log.csv` | Every cleaning step with rows before/after and justification |
| `data/features/` | Engineered features and the train/test split (Unit 2) |
| `data/results/` | Method 1 forecasts and metrics (`m3/`), Method 2 rules (`m4/`) |
| `notebooks/` | Complete implementation of Units 1–3 (order below) |
| `notebooks/scripts/` | Helper scripts used while building the association rules |
| `dashboard/` | Streamlit application (`app.py`, `dash_data.py`, `data/`) |
| `visualizations/` | Project charts by stage: EDA, feature engineering, time series, association rules |

### Notebooks (run in this order)

| # | Notebook | Stage |
|---|---|---|
| 1 | `Dataset_Generation.ipynb` | Data collection from OpenAQ v3, Open-Meteo and the calendar |
| 2 | `M2_cleaning_preprocessing_merging.ipynb` | Cleaning, validation and merging |
| 3 | `M3_EDA.ipynb` | Unit 1: exploratory analysis and visualization |
| 4 | `M4_Feature_Engineering.ipynb` | Unit 2: target definition, feature engineering, train/test split |
| 5 | `M5_Model_Evaluation.ipynb` | Unit 2: predictive modelling and evaluation |
| 6 | `M2_Input_Preparation_P2.ipynb`, `M2_TimeSeries_Prep_Baselines.ipynb` | Unit 3 Method 1: series preparation and baselines |
| 7 | `M3_Time_Series_Modelling_Final.ipynb` | Unit 3 Method 1: time-series forecasting |
| 8 | `M4_Association_Rule_Mining.ipynb` | Unit 3 Method 2: association rule mining |

The notebooks were run in Google Colab; the OpenAQ API key is read from the
`OPENAQ_API_KEY` secret and is not stored in the repository.

## Project management

Tasks were tracked on the GitHub Projects board:
https://github.com/users/sanjanav2102/projects/2

## Team

**Group 11 --- 23CSE452 Business Analytics**
Guide: Rohini S, Assistant Professor, Department of CSE

| Member | Register number | Contribution |
|---|---|---|
| V Sanjana | CB.SC.U4CSE23751 | Data collection; project board; Unit 3 Method 1 (time series) |
| M Nandan Reddy | CB.SC.U4CSE23431 | Data merging and cleaning; GRAP integration |
| Akilan J | CB.SC.U4CSE23002 | Unit 1 EDA; dashboard build and deployment |
| Jei Akash M A | CB.SC.U4CSE23518 | Feature engineering; Unit 3 Method 2 (association rules) |
| Isha Sri Prakash | CB.SC.U4CSE23029 | Model data preparation; Unit 2 predictive modelling |
