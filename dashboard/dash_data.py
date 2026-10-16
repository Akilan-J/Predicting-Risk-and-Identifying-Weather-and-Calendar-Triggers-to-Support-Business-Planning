"""Data layer for the M5 dashboard.

Every loader first looks for the file agreed in the Phase 2 data contract (in data/).
If a teammate's file is not there yet, it builds a clearly labelled PLACEHOLDER from
cleaned_final.csv, so every dashboard section works from day one. When the real file
is dropped into data/, the dashboard switches to it automatically.

Contract files:
    p2_series.csv          (M2)  date, pm25_target, target_source, missing, high_risk
    p2_forecasts.csv       (M3)  forecasts per horizon (+ intervals if available)
    p2_ts_metrics.csv      (M3)  MAE / RMSE per horizon
    m4_arm_dashboard.csv   (M4)  rule_text, set, support, confidence, lift (+ CI), test_conf, test_n, status
    m5_test_predictions.csv (Phase 1) test-set probabilities
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
THRESH = 120.5          # shared team setting: PM2.5-based AQI > 300
SEVERE = 250.5          # AQI 401+
TEST_START = pd.Timestamp("2025-01-01")
HORIZONS = (1, 2, 3)

CATEGORY_EDGES = [-np.inf, 30.5, 60.5, 90.5, 120.5, 250.5, np.inf]
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def file_status():
    """Which contract files are present (True) or replaced by placeholders (False)."""
    names = {
        "cleaned_final.csv": "Team (Phase 1)",
        "m3_test_predictions.csv": "M3",
        "m3_test_metrics.csv": "M3",
        "m4_arm_dashboard.csv": "M4",
        "m5_test_predictions.csv": "M5 (Phase 1)",
    }
    return [(n, owner, (DATA / n).exists()) for n, owner in names.items()]


# ---------------------------------------------------------------- base data
@st.cache_data
def load_base() -> pd.DataFrame:
    df = pd.read_csv(DATA / "cleaned_final.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["year"] = df.date.dt.year
    df["category"] = pd.cut(df.pm25_target, CATEGORY_EDGES, labels=CATEGORIES, right=False)
    df["aqi"] = df.pm25_target.apply(pm25_to_aqi)
    return df


@st.cache_data
def load_series() -> tuple[pd.DataFrame, bool]:
    """M2's series if present, else the same columns taken from cleaned_final.csv."""
    p = DATA / "p2_series.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"]), True
    b = load_base()
    s = b[["date", "pm25_target", "target_source", "high_risk"]].copy()
    s["missing"] = s.pm25_target.isna().astype(int)
    return s, False


def pm25_to_aqi(c):
    """CPCB PM2.5 sub-index (concentration rounded to a whole number first)."""
    if pd.isna(c):
        return np.nan
    c = round(c)
    bp = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300),
          (121, 250, 301, 400), (251, 380, 401, 500)]
    for lo, hi, ilo, ihi in bp:
        if lo <= c <= hi:
            return ilo + (ihi - ilo) * (c - lo) / (hi - lo)
    return 500.0


# ---------------------------------------------------------------- features (placeholder models)
def _features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only features available at the end of day t (no future information)."""
    f = pd.DataFrame({"date": df.date})
    pm = df.pm25_target.ffill(limit=2)           # fill at most 2 days, past values only
    f["pm_filled_flag"] = df.pm25_target.isna().astype(int)
    for k in (0, 1, 2):
        f[f"log_pm_lag{k}"] = np.log1p(pm.shift(k))
    f["log_pm_roll3"] = np.log1p(pm.rolling(3, min_periods=2).mean())
    f["log_pm_roll7"] = np.log1p(pm.rolling(7, min_periods=4).mean())
    doy = df.date.dt.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["wind"] = df.wind_speed_mean
    f["rain_day"] = (df.precip_sum >= 1).astype(int)
    f["humidity"] = df.humidity_mean
    f["diwali_near"] = df.days_from_diwali.between(-7, 7).astype(int)
    return f


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, bool]:
    """Test-set alert probabilities per horizon.

    Real file: m5_test_predictions.csv (Phase 1). Accepts common column names.
    Placeholder: logistic regression on past-only features, trained 2018-2024, tested 2025-2026.
    Output columns: date (issue date), horizon, target_date, prob, actual
    """
    p = DATA / "m5_test_predictions.csv"
    if p.exists():
        r = pd.read_csv(p)
        cols = {c.lower(): c for c in r.columns}
        exact = [cols[c] for c in cols if c in ("prob", "proba", "y_prob", "p_high", "probability", "pred_proba")]
        # Phase 1 notebook writes one prob_<model> column per model; its selected model is Gradient Boosting
        prob = (exact or [cols[c] for c in cols if c == "prob_gradient_boosting"]
                or [cols[c] for c in cols if c.startswith("prob_")])[0]
        act = next(cols[c] for c in cols if c in ("actual", "y_true", "high_risk", "target", "y"))
        dcol = next(cols[c] for c in cols if c in ("date", "target_date", "ds"))
        out = pd.DataFrame({"target_date": pd.to_datetime(r[dcol]), "prob": r[prob], "actual": r[act]})
        out["horizon"] = r[cols["horizon"]] if "horizon" in cols else 1
        out["date"] = out.target_date - pd.to_timedelta(out.horizon, unit="D")
        return out.dropna(subset=["actual"]), True

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    df = load_base()
    X = _features(df)
    feats = [c for c in X.columns if c != "date"]
    rows = []
    for h in HORIZONS:
        y = df.high_risk.shift(-h)
        d = X.assign(y=y, target_date=df.date.shift(-h)).dropna(subset=["y", "log_pm_lag0"])
        tr, te = d[d.target_date < TEST_START], d[d.target_date >= TEST_START]
        m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
        m.fit(tr[feats], tr.y)
        rows.append(pd.DataFrame({"date": te.date, "horizon": h, "target_date": te.target_date,
                                  "prob": m.predict_proba(te[feats])[:, 1], "actual": te.y}))
    return pd.concat(rows, ignore_index=True), False


@st.cache_data
def load_forecasts() -> tuple[pd.DataFrame, bool]:
    """PM2.5 forecasts per horizon on the test period.

    Real file: p2_forecasts.csv (M3), long or wide format.
    Placeholder: persistence, seasonal-naive (same day last year) and a ridge model on log PM2.5
    with an 80% interval from training residuals.
    Output (long): date (issue), horizon, target_date, model, forecast, lo, hi, actual
    """
    m3 = DATA / "m3_test_predictions.csv"
    if m3.exists():   # M3's notebook output: one column per model, plus the selected model per horizon
        w = pd.read_csv(m3, parse_dates=["origin_date", "target_date"])
        ids = [c for c in ["origin_date", "target_date", "horizon", "actual_pm25", "is_winter_target", "selected"] if c in w]
        f = w.melt(id_vars=ids, var_name="model", value_name="forecast")
        f = f.rename(columns={"origin_date": "date", "actual_pm25": "actual"})
        if "selected" in f:
            f["is_selected"] = f.model == f.selected.str.replace(" (best config)", "", regex=False)
        f["lo"], f["hi"] = np.nan, np.nan
        return f, True
    p = DATA / "p2_forecasts.csv"
    if p.exists():
        f = pd.read_csv(p)
        for c in ("date", "target_date"):
            if c in f:
                f[c] = pd.to_datetime(f[c])
        return f, True

    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    df = load_base()
    X = _features(df)
    feats = [c for c in X.columns if c != "date"]
    pm = df.pm25_target
    out = []
    for h in HORIZONS:
        tgt = pm.shift(-h)
        d = X.assign(y=tgt, target_date=df.date.shift(-h), persist=pm.ffill(limit=2),
                     snaive=pm.shift(365 - h)).dropna(subset=["y", "log_pm_lag0"])
        tr, te = d[d.target_date < TEST_START], d[d.target_date >= TEST_START]
        m = make_pipeline(SimpleImputer(), Ridge(alpha=1.0)).fit(tr[feats], np.log1p(tr.y))
        resid = np.log1p(tr.y) - m.predict(tr[feats])
        q10, q90 = np.quantile(resid, [0.1, 0.9])
        pred = m.predict(te[feats])
        base = dict(date=te.date.values, horizon=h, target_date=te.target_date.values, actual=te.y.values)
        out.append(pd.DataFrame({**base, "model": "Ridge (placeholder)", "forecast": np.expm1(pred),
                                 "lo": np.expm1(pred + q10), "hi": np.expm1(pred + q90)}))
        out.append(pd.DataFrame({**base, "model": "Persistence", "forecast": te.persist.values, "lo": np.nan, "hi": np.nan}))
        out.append(pd.DataFrame({**base, "model": "Seasonal naive", "forecast": te.snaive.values, "lo": np.nan, "hi": np.nan}))
    return pd.concat(out, ignore_index=True), False


def selected_models(fc: pd.DataFrame) -> dict:
    """Model chosen by M3 for each horizon (placeholder: the ridge model)."""
    if "is_selected" in fc and fc.is_selected.any():
        return fc[fc.is_selected].groupby("horizon").model.first().to_dict()
    return {h: fc.model.iloc[0] for h in HORIZONS}


@st.cache_data
def load_ts_metrics():
    """M3's test metrics with 95% block-bootstrap intervals, if delivered."""
    for name in ("m3_test_metrics.csv", "p2_ts_metrics.csv"):
        if (DATA / name).exists():
            return pd.read_csv(DATA / name), True
    return None, False


def forecast_metrics(fc: pd.DataFrame, winter_only=False) -> pd.DataFrame:
    """MAE / RMSE per model and horizon, plus alert recall / precision at THRESH."""
    d = fc.dropna(subset=["forecast", "actual"]).copy()
    if winter_only:
        d = d[pd.to_datetime(d.target_date).dt.month.isin([11, 12, 1])]
    rows = []
    for (mdl, h), g in d.groupby(["model", "horizon"]):
        e = g.forecast - g.actual
        pa, pp = g.actual >= THRESH, g.forecast >= THRESH
        tp = (pa & pp).sum()
        rows.append(dict(model=mdl, horizon=h, days=len(g), MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()),
                         alert_recall=tp / pa.sum() if pa.sum() else np.nan,
                         alert_precision=tp / pp.sum() if pp.sum() else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- association rules
def bin_conditions(df: pd.DataFrame, fit_mask) -> pd.DataFrame:
    """Turn weather/calendar into categories. Bin edges are fitted on pre-2025 days only."""
    c = pd.DataFrame({"date": df.date})
    for col, name, labels in [("wind_speed_mean", "Wind", ["calm", "moderate", "windy"]),
                              ("humidity_mean", "Humidity", ["low", "medium", "high"]),
                              ("temp_mean", "Temp", ["cool", "mild", "hot"]),
                              ("pressure_mean", "Pressure", ["low", "medium", "high"])]:
        edges = df.loc[fit_mask, col].quantile([1 / 3, 2 / 3]).values
        c[name] = pd.cut(df[col], [-np.inf, *edges, np.inf], labels=labels).astype(str)
    c["Rain"] = np.where(df.precip_sum >= 1, "rain", "dry")
    m = df.date.dt.month
    c["Season"] = np.select([m.isin([11, 12, 1]), m.isin([10, 2]), m.isin([3, 4, 5])],
                            ["peak winter", "shoulder", "spring"], "monsoon")
    c["Diwali"] = np.where(df.is_festival_window == 1, "window", "no")
    c["Weekend"] = np.where(df.is_weekend == 1, "yes", "no")
    c["Yesterday"] = np.where(df.high_risk.shift(1) == 1, "high-risk",
                              np.where(df.high_risk.shift(1) == 0, "not high-risk", "unknown"))
    c["next_high_risk"] = df.high_risk.shift(-1)       # consequent: high_risk on day t+1
    return c


@st.cache_data
def load_rules() -> tuple[pd.DataFrame, bool]:
    """Ranked rules 'conditions on day t -> high-risk on day t+1'.

    Real file: m4_arm_dashboard.csv (M4's notebook output, FP-Growth). Placeholder: mlxtend Apriori on
    pre-2025 days, Set A (weather + calendar) and Set B (+ yesterday's status), re-checked on 2025-2026.
    """
    p = DATA / "m4_arm_dashboard.csv"
    if p.exists():
        r = pd.read_csv(p)
        sets = {"A": "A: weather + calendar", "B": "B: + yesterday's status"}
        out = pd.DataFrame({
            "set": r["set"].map(sets).fillna(r["set"]),
            "antecedent": r["rule_text"].str.replace(" + ", " AND ", regex=False),
            "consequent": "high-risk next day", "n_conditions": r["k"],
            "support": r["support"], "confidence": r["confidence"], "lift": r["lift"],
            "lift_ci_lo": r.get("lift_ci_lo"), "lift_ci_hi": r.get("lift_ci_hi"),
            "test_days": r.get("test_n"), "test_confidence": r.get("test_conf"), "status": r.get("status")})
        return out.sort_values(["lift", "support"], ascending=False).reset_index(drop=True), True

    from mlxtend.frequent_patterns import apriori

    df = load_base()
    train = df.date < TEST_START
    cond = bin_conditions(df, train)
    cond = cond.dropna(subset=["next_high_risk"])
    out = []
    for set_name, cols in [("A: weather + calendar", ["Wind", "Humidity", "Temp", "Pressure", "Rain", "Season", "Diwali", "Weekend"]),
                           ("B: + yesterday's status", ["Wind", "Humidity", "Temp", "Pressure", "Rain", "Season", "Diwali", "Weekend", "Yesterday"])]:
        tr = cond[cond.date < TEST_START]
        onehot = pd.get_dummies(tr[cols].astype(str), prefix_sep="=").astype(bool)
        onehot["HIGH_RISK_NEXT_DAY"] = tr.next_high_risk.astype(bool).values
        freq = apriori(onehot, min_support=0.05, use_colnames=True, max_len=4)
        sup = {frozenset(s): v for s, v in zip(freq.itemsets, freq.support)}
        base_rate = tr.next_high_risk.mean()
        te = cond[cond.date >= TEST_START]
        te_hot = pd.get_dummies(te[cols].astype(str), prefix_sep="=").astype(bool)
        for items, s_ac in sup.items():
            if "HIGH_RISK_NEXT_DAY" not in items or len(items) < 2:
                continue
            ante = items - {"HIGH_RISK_NEXT_DAY"}
            s_a = sup.get(frozenset(ante))
            if not s_a:
                continue
            conf = s_ac / s_a
            lift = conf / base_rate
            if conf < 0.6 or lift < 1.2:
                continue
            mask = np.ones(len(te), dtype=bool)
            for it in ante:
                mask &= te_hot[it].values if it in te_hot else np.zeros(len(te), dtype=bool)
            n_te = int(mask.sum())
            out.append(dict(set=set_name, antecedent=" AND ".join(sorted(ante)), consequent="high-risk next day",
                            n_conditions=len(ante), support=s_ac, confidence=conf, lift=lift,
                            test_days=n_te, test_confidence=te.next_high_risk.values[mask].mean() if n_te else np.nan))
    rules = pd.DataFrame(out).sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
    return rules, False


def rule_check(conds: dict) -> dict:
    """Live rule builder: support / confidence / lift for any user-chosen set of conditions."""
    df = load_base()
    train = df.date < TEST_START
    c = bin_conditions(df, train).dropna(subset=["next_high_risk"])
    res = {}
    for name, part in [("train", c[c.date < TEST_START]), ("test", c[c.date >= TEST_START])]:
        mask = np.ones(len(part), dtype=bool)
        for k, v in conds.items():
            if v:
                mask &= part[k].isin(v).values
        base = part.next_high_risk.mean()
        n = int(mask.sum())
        conf = part.next_high_risk.values[mask].mean() if n else np.nan
        res[name] = dict(days=n, support=n / len(part), confidence=conf,
                         lift=conf / base if n else np.nan, base_rate=base)
    return res
