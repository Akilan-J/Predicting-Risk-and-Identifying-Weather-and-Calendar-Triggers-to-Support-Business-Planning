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

## Team

**Group 11 --- 23CSE452 Business Analytics**

-   Akilan J
-   Isha Sri Prakash
-   M Nandan Reddy
-   Jei Akash M A
-   V Sanjana
