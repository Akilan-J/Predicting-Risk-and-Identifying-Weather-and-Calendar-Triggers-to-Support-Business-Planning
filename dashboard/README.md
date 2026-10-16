# Delhi Air-Risk Planner (M5 dashboard, Review 2)

Interactive Streamlit + Plotly dashboard for Team 11's case study: predicting days when Delhi's
PM2.5-based AQI exceeds 300, and the weather and calendar conditions behind them.

## Run it

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Pages and the Review 2 rubric

| Page | What it shows | Rubric item |
|---|---|---|
| Overview | 3-day risk cards (replay any test day, quick-jump scenarios), last 60 days + outlook, year × month risk heat map | Results on the final data; live user scenarios |
| Dataset | Records, variables, sources, validation checks, filterable data table, data dictionary | Display the final project dataset |
| Forecast (Method 1) | Actual vs forecast per horizon, intervals, MAE/RMSE, alert recall/precision, model comparison | Method 1 results, compare alternatives |
| Rules (Method 2) | Support/confidence/lift filters, bubble chart, plain-language rules, test-period confidence, Set A vs Set B | Method 2 results, compare alternatives |
| Insights | Alert threshold + cost ratio, model vs "tomorrow = today" baseline, combined findings, what-if planner with sector actions | Interactions that update results; combined business insights |

## Data contract

Put teammates' files in `data/` with these names. Until a file is there, the dashboard computes a
clearly labelled **placeholder** from `cleaned_final.csv`, and switches automatically once it arrives.

| File | Owner | Used by |
|---|---|---|
| `cleaned_final.csv` | Team (Phase 1) | Everything |
| `p2_series.csv` | M2 | Series checks |
| `m3_test_predictions.csv` | M3 | Overview, Forecast (M3 notebook output: origin_date, target_date, horizon, actual_pm25, one column per model, selected) |
| `m3_test_metrics.csv` | M3 | Forecast evaluation table with 95% intervals |
| `p2_rules.csv` | M4 | Rule explorer (antecedent, consequent, support, confidence, lift, test_confidence) |
| `m5_test_predictions.csv` | Phase 1 model | Control room, Alert threshold (date, prob, actual, optional horizon) |

## Notes

- The Control room shows the **last date in the data**, not today (no live feed).
- Rules are **associations, not causes**.
- Placeholders use past-only features and are trained on 2018–2024, tested on Jan 2025 – Sep 2026.
