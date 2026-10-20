"""Delhi Air-Risk Planner: M5 dashboard (Review 2).

Run:  streamlit run app.py
Every section works now; sections waiting for a teammate's file show a PLACEHOLDER badge
and switch to the real file as soon as it is dropped into data/ (see dash_data.py).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import dash_data as D

st.set_page_config(page_title="Delhi Air-Risk Planner", page_icon=":material/air:", layout="wide")

# ---------------------------------------------------------------- look and feel
BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1b2a44", "#52514e", "#898781", "#e6e6e1"
STATUS = {"Low": "#0ca30c", "Watch": "#fab219", "High": "#ec835a", "Severe": "#d03b3b"}
CAT_COLORS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]   # Good -> Severe (ordinal)
SEQ = ["#eef5fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

pio.templates["planner"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[BLUE, ORANGE, AQUA, "#eda100", "#4a3aa7"],
    xaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False),
    hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor="#c3c2b7"),
    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08, x=0)))
pio.templates.default = "planner"

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg,#1b2a44 0%,#1c5cab 60%,#2a78d6 100%); color:#fff;
       padding: 22px 28px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {color:#fff; font-size: 1.9rem; margin: 0 0 4px 0;}
.hero p {color:#dbe7fb; margin:0; font-size: 1.0rem;}
.card {background:#fff; border:1px solid #e3e6ee; border-radius:16px; padding:16px 18px; height:100%;
       box-shadow: 0 1px 2px rgba(27,42,68,.05);}
.card .k {font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; color:#6b7280; font-weight:600;}
.card .v {font-size:2.1rem; font-weight:700; color:#1b2a44; line-height:1.15;}
.card .s {font-size:.9rem; color:#52514e;}
.chip {display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700; font-size:.8rem; color:#1b2a44;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:700;
        background:#fff4e5; color:#8a4b00; border:1px solid #f3c27a;}
.badge.ok {background:#e8f7ec; color:#0b6b23; border-color:#9fdcae;}
.note {font-size:.85rem; color:#6b7280;}
</style>""", unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def source_badge(real: bool, owner: str):
    if real:
        st.markdown(f'<span class="badge ok">✓ Live file from {owner}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge">⏳ PLACEHOLDER: computed stand-in until {owner} delivers the file</span>',
                    unsafe_allow_html=True)


def card(col, k, v, s="", chip=None):
    chip_html = f'<span class="chip" style="background:{STATUS[chip]}33;border:1px solid {STATUS[chip]}">{chip}</span>' if chip else ""
    col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
                 f'<div class="s">{chip_html} {s}</div></div>', unsafe_allow_html=True)


def risk_level(p, t):
    """Status from the alert probability, relative to the chosen alert threshold."""
    if p >= max(t, 0.8):
        return "Severe"
    if p >= t:
        return "High"
    if p >= t * 0.6:
        return "Watch"
    return "Low"


def chart(fig, h=None):
    if h:
        fig.update_layout(height=h)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


base = D.load_base()
preds, preds_real = D.load_predictions()
fcs, fcs_real = D.load_forecasts()
rules, rules_real = D.load_rules()
st.session_state.setdefault("alert_t", 0.5)

ACTIONS = {   # wording follows the team's "PM2.5 Risk Preparation Playbook"; the model does not decide official restrictions
    "Construction": {"Low": "Normal schedule.",
                     "Watch": "Review the forecast trend; prepare dust-suppression materials and covered storage.",
                     "High": "Recheck the latest forecast; inspect dust-control arrangements so they can be applied promptly.",
                     "Severe": "Increase dust suppression, material handling and site monitoring; follow official construction requirements."},
    "Logistics": {"Low": "Normal routing.",
                  "Watch": "Review delivery schedules and vehicle availability; find trips that could be combined.",
                  "High": "Coordinate changes with drivers, customers and dispatch; prepare alternative schedules.",
                  "Severe": "Reduce avoidable trips where feasible; follow official vehicle and traffic requirements."},
    "Schools": {"Low": "Outdoor activities as planned.",
                "Watch": "Prepare indoor alternatives and messages for staff and parents.",
                "High": "Check official air-quality advisories; confirm staff know the health and safety procedures.",
                "Severe": "Use approved procedures for outdoor activities and sensitive students; follow official directions."},
    "Events": {"Low": "Go ahead as planned.",
               "Watch": "Review the forecast before finalising outdoor arrangements; identify backup venues.",
               "High": "Reassess the latest forecast; confirm how you will decide to continue, relocate or postpone.",
               "Severe": "Reassess outdoor activities with current observations and official advice; communicate decisions clearly."},
}

PLAYBOOK = {   # "PM2.5 Risk Preparation Playbook", section 3
    "Construction": {
        "2–3 days ahead": ["Review the forecast trend and check whether the weather/calendar risk patterns are present.",
                           "Prepare dust-suppression materials, site-inspection schedules and covered storage.",
                           "Coordinate equipment, labour and deliveries in case preventive measures need to increase."],
        "1 day ahead": ["Recheck the latest forecast and available air-quality observations.",
                        "Inspect dust-control arrangements and confirm site procedures can be applied promptly."],
        "If risk stays high": ["Increase attention to dust suppression, material handling and site monitoring.",
                               "Follow applicable official construction and demolition requirements."]},
    "Logistics": {
        "2–3 days ahead": ["Review delivery schedules, vehicle availability and operational flexibility.",
                           "Identify deliveries that can be consolidated or rescheduled without disrupting essential services."],
        "1 day ahead": ["Review updated forecasts and coordinate changes with drivers, customers and dispatch teams.",
                        "Prepare alternative schedules if high pollution coincides with operational constraints."],
        "If risk stays high": ["Reduce avoidable trips where feasible.",
                               "Follow applicable vehicle-emission, traffic and regulatory requirements."]},
    "Schools": {
        "2–3 days ahead": ["Review the forecast trend and prepare indoor alternatives for outdoor activities.",
                           "Prepare communication for staff, parents and students."],
        "1 day ahead": ["Check updated forecasts and official air-quality advisories.",
                        "Confirm that staff know the relevant health and safety procedures."],
        "If risk stays high": ["Use the institution's approved procedures for outdoor activities and sensitive students.",
                               "Follow official government directions on school operations or hybrid classes."]},
    "Events": {
        "2–3 days ahead": ["Review the forecast trend before finalising outdoor arrangements.",
                           "Identify backup venues, flexible scheduling options and attendee communication channels."],
        "1 day ahead": ["Reassess the latest forecast and available air-quality observations.",
                        "Confirm the decision process for continuing, modifying, relocating or postponing."],
        "If risk stays high": ["Reassess outdoor activities using current observations, forecast uncertainty and official advice.",
                               "Communicate decisions clearly to participants and staff."]},
}


# ================================================================ 1. OVERVIEW
def page_overview():
    hero("Overview", "3-day risk outlook for Delhi, and when high-risk days happen: will the PM2.5-based AQI exceed 300?")
    t = st.session_state.alert_t
    fc_dates = set(fcs.loc[fcs.horizon == 3, "date"])
    dates = sorted(d_ for d_ in preds.loc[preds.horizon == 1, "date"].unique() if d_ in fc_dates) \
        or sorted(preds.loc[preds.horizon == 1, "date"].unique())
    diwali = base.loc[(base.days_from_diwali == 0) & (base.date >= D.TEST_START), "date"]
    win = preds[(preds.horizon == 1) & pd.to_datetime(preds.target_date).dt.month.isin([11, 12, 1])]
    jumps = {"Latest date": dates[-1],
             "Day before Diwali 2025": (diwali.iloc[0] - pd.Timedelta(days=1)) if len(diwali) else dates[-1],
             "Riskiest winter day": win.loc[win.prob.idxmax(), "date"] if len(win) else dates[-1],
             "A missed alert": preds[(preds.horizon == 1) & (preds.actual == 1)].sort_values("prob").date.iloc[0]}
    st.session_state.setdefault("asof", dates[-1])
    j = st.columns(len(jumps))
    for col, (lbl, dt) in zip(j, jumps.items()):
        if col.button(lbl, width="stretch"):
            st.session_state.asof = pd.Timestamp(dt)
    asof = st.select_slider("Replay any day in the test period (Jan 2025 – Sep 2026)", options=dates,
                            key="asof", format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
    source_badge(preds_real, "the Phase 1 model (m5_test_predictions.csv)")
    st.caption("Shows the **last date in the data** (30 Sep 2026) by default, not today: the dashboard has no live data feed.")

    p = preds[preds.date == asof].set_index("horizon")
    sel = D.selected_models(fcs)
    fc = pd.concat([fcs[(fcs.date == asof) & (fcs.horizon == h) & (fcs.model == m)] for h, m in sel.items()]).set_index("horizon")
    cols = st.columns(3)
    for col, h in zip(cols, D.HORIZONS):
        pm = fc.loc[h, "forecast"] if h in fc.index else np.nan
        if h in p.index:      # alert probability from the Phase 1 classifier
            prob = float(p.loc[h, "prob"])
            tdate = pd.Timestamp(p.loc[h, "target_date"])
            outcome = "✔ was high-risk" if p.loc[h, "actual"] == 1 else "✘ was not high-risk"
            card(col, f"Day +{h} · {tdate:%a %d %b}", f"{prob:.0%}",
                 f"alert probability · PM2.5 ≈ {pm:.0f} µg/m³<br>Actual: {outcome}", chip=risk_level(prob, t))
        elif h in fc.index:   # no classifier for this horizon: alert when M3's PM2.5 forecast ≥ 120.5
            tdate = pd.Timestamp(fc.loc[h, "target_date"])
            act = fc.loc[h, "actual"]
            outcome = "✔ was high-risk" if act >= D.THRESH else "✘ was not high-risk"
            card(col, f"Day +{h} · {tdate:%a %d %b}", f"{pm:.0f} µg/m³",
                 f"PM2.5 forecast (AQI ≈ {D.pm25_to_aqi(pm):.0f})<br>Actual: {outcome}",
                 chip="High" if pm >= D.THRESH else "Low")
        else:
            card(col, f"Day +{h}", "–", "no forecast for this day")
    st.caption(f"Risk labels use your alert threshold of **{t:.0%}** (change it on the *Insights* page).")

    hist = base[(base.date > asof - pd.Timedelta(days=60)) & (base.date <= asof)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist.date, y=hist.pm25_target, name="PM2.5 (observed)",
                         marker_color=np.where(hist.high_risk == 1, BLUE, "#b7d3f6"),
                         hovertemplate="%{x|%d %b}<br>PM2.5 %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fc.target_date, y=fc.forecast, mode="lines+markers", name="Forecast (next 3 days)",
                             line=dict(color=ORANGE, width=3, dash="dot"), marker=dict(size=10)))
    fig.add_hline(y=D.THRESH, line_dash="dash", line_color=STATUS["Severe"],
                  annotation_text="AQI 300 line", annotation_position="top left")
    fig.update_layout(title="Last 60 days and the 3-day outlook (dark bars = high-risk days)", yaxis_title="PM2.5 (µg/m³)")
    chart(fig, 340)

    st.subheader("Seasonal risk calendar")
    c1, c2 = st.columns([2, 3])
    metric = c1.segmented_control("Colour by", ["% high-risk days", "% Severe days", "Median PM2.5"],
                                  default="% high-risk days") or "% high-risk days"
    yrs = c2.slider("Years", int(base.year.min()), int(base.year.max()), (int(base.year.min()), int(base.year.max())))
    d = base[base.year.between(*yrs)]
    f = {"% high-risk days": lambda g: g.high_risk.mean() * 100,
         "% Severe days": lambda g: (g.pm25_target >= D.SEVERE).sum() / g.pm25_target.notna().sum() * 100,
         "Median PM2.5": lambda g: g.pm25_target.median()}[metric]
    grid = d.groupby(["year", d.date.dt.month]).apply(f).unstack().reindex(columns=range(1, 13))
    mn = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(z=grid.values, x=mn, y=grid.index.astype(str), colorscale=SEQ,
                               text=[["" if np.isnan(v) else f"{v:.0f}" for v in row] for row in grid.values],
                               texttemplate="%{text}", xgap=3, ygap=3,
                               hovertemplate="%{y} %{x}<br>" + metric + ": %{z:.0f}<extra></extra>",
                               colorbar=dict(title=metric, thickness=12)))
    fig.update_layout(title=f"{metric} by year and month", yaxis=dict(autorange="reversed"))
    chart(fig, 400)
    st.caption("High-risk = PM2.5-based AQI above 300 (pm25_target ≥ 120.5 µg/m³). Days with no target are excluded.")


# ================================================================ 3. FORECAST (METHOD 1)
def page_forecast():
    hero("PM2.5 Forecast · Method 1: Time-series", "Forecasts for 1, 2 and 3 days ahead on the held-out 2025–2026 test period.")
    source_badge(fcs_real, "M3 (m3_test_predictions.csv)")
    sel = D.selected_models(fcs)
    models = list(dict.fromkeys(fcs.model))
    mcol = {m: [BLUE, ORANGE, AQUA, "#eda100", "#4a3aa7", "#e87ba4", "#008300", MUTED][i % 8] for i, m in enumerate(models)}
    c1, c2, c3, c4 = st.columns([1.2, 2.4, 1, 1])
    h = c1.segmented_control("Horizon", [1, 2, 3], default=1, format_func=lambda x: f"Day +{x}") or 1
    default = [m for m in [sel.get(h), "Persistence"] if m in models] or models[:2]
    chosen = c2.multiselect("Models", models, default=default, key=f"fc_models_{h}")
    winter = c3.toggle("Winter only (Nov–Jan)")
    band = c4.toggle("Show interval", value=True, disabled=fcs.lo.isna().all())
    st.caption(f"Selected model for Day +{h}: **{sel.get(h, '–')}**.")

    d = fcs[(fcs.horizon == h) & fcs.model.isin(chosen)].copy()
    d["target_date"] = pd.to_datetime(d.target_date)
    lo, hi = d.target_date.min().date(), d.target_date.max().date()
    rng = st.slider("Date range", lo, hi, (pd.Timestamp("2025-10-01").date(), pd.Timestamp("2026-02-28").date()))
    d = d[d.target_date.between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]))]
    if winter:
        d = d[d.target_date.dt.month.isin([11, 12, 1])]

    fig = go.Figure()
    act = d.drop_duplicates("target_date").sort_values("target_date")
    fig.add_trace(go.Scatter(x=act.target_date, y=act.actual, name="Actual", mode="lines", line=dict(color=INK, width=2)))
    for m in chosen:
        g = d[d.model == m].sort_values("target_date")
        if band and g.lo.notna().any():
            fig.add_trace(go.Scatter(x=pd.concat([g.target_date, g.target_date[::-1]]), y=pd.concat([g.hi, g.lo[::-1]]),
                                     fill="toself", fillcolor=mcol[m] + "22", line=dict(width=0), name=f"{m} interval",
                                     hoverinfo="skip"))
        name = m + (" ★ selected" if m == sel.get(h) else "")
        fig.add_trace(go.Scatter(x=g.target_date, y=g.forecast, name=name, mode="lines", line=dict(color=mcol[m], width=2)))
    fig.add_hline(y=D.THRESH, line_dash="dash", line_color=STATUS["Severe"], annotation_text="AQI 300 line")
    fig.update_layout(title=f"Actual vs forecast, Day +{h}", yaxis_title="PM2.5 (µg/m³)", hovermode="x unified")
    chart(fig, 430)

    st.subheader("Evaluation on the held-out test set")
    mt, _ = D.load_ts_metrics()
    c1, c2 = st.columns([1.3, 1])
    if mt is not None and "subset" in mt:
        sub = mt[mt.subset.str.startswith("winter")] if winter else mt[mt.subset == "all test days"]
        sub = sub.assign(base=sub.model.str.replace(" [SELECTED]", "", regex=False))
        sub = sub[sub.base.isin(chosen)].sort_values(["horizon", "MAE"])
        ci = lambda r, m, nd: f"{r[m]:.{nd}f} [{r[m + '_lo']:.{nd}f}–{r[m + '_hi']:.{nd}f}]"
        tbl = pd.DataFrame({"Day +": sub.horizon, "Model": sub.model,
                            "MAE [95% CI]": [ci(r, "MAE", 1) for _, r in sub.iterrows()],
                            "RMSE": sub.RMSE.round(1), "Recall": sub.recall, "Precision": sub.precision, "F1": sub.F1})
        with c1:
            st.dataframe(tbl, hide_index=True, width="stretch", column_config={
                "Recall": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Precision": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "F1": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
            st.caption("Intervals: 95% block bootstrap (M3). An alert is a forecast ≥ 120.5 µg/m³.")
        with c2:
            fig = px.bar(sub, x="horizon", y="MAE", color="base", barmode="group", color_discrete_map=mcol,
                         error_y=sub.MAE_hi - sub.MAE, error_y_minus=sub.MAE - sub.MAE_lo,
                         labels={"horizon": "Days ahead", "MAE": "MAE (µg/m³)", "base": "model"})
            fig.update_layout(title="Error by horizon (with 95% CI)")
            fig.update_xaxes(tickvals=[1, 2, 3])
            chart(fig, 340)
    else:
        met = D.forecast_metrics(fcs, winter_only=winter)
        with c1:
            show = met[met.model.isin(chosen)].sort_values(["horizon", "MAE"])
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "MAE": st.column_config.NumberColumn(format="%.1f"), "RMSE": st.column_config.NumberColumn(format="%.1f"),
                "alert_recall": st.column_config.ProgressColumn("Alert recall", min_value=0, max_value=1, format="%.2f"),
                "alert_precision": st.column_config.ProgressColumn("Alert precision", min_value=0, max_value=1, format="%.2f")})
            st.caption("An alert is a forecast ≥ 120.5 µg/m³.")
        with c2:
            fig = px.bar(met[met.model.isin(chosen)], x="horizon", y="MAE", color="model", barmode="group",
                         color_discrete_map=mcol, labels={"horizon": "Days ahead", "MAE": "MAE (µg/m³)"})
            fig.update_layout(title="Error by horizon")
            fig.update_xaxes(tickvals=[1, 2, 3])
            chart(fig, 320)
    st.caption("Persistence ('tomorrow = today') is the baseline to beat. Models marked ORACLE use perfect future weather "
               "and are an upper bound only, not a usable forecast.")


# ================================================================ 4. RULE EXPLORER (METHOD 2)
def page_rules():
    hero("Rule Explorer · Method 2: Association rules",
         "Which weather and calendar conditions on day t are followed by a high-risk day t+1?")
    source_badge(rules_real, "M4 (m4_arm_dashboard.csv)")
    r = rules.copy()
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        sets = sorted(r.set.unique()) if "set" in r else []
        pick = c1.radio("Rule set", sets, index=0) if sets else None
        sup = c2.slider("Min support", 0.0, float(r.support.max()), 0.05, 0.01)
        conf = c3.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)
        lift = c4.slider("Min lift", 1.0, float(np.ceil(r.lift.max() * 10) / 10), 1.2, 0.1)
        items = sorted({i for a in r.antecedent for i in a.split(" AND ")})
        must = st.multiselect("Rule must include", items, placeholder="e.g. Low wind")
    if pick:
        r = r[r.set == pick]
    r = r[(r.support >= sup) & (r.confidence >= conf) & (r.lift >= lift)]
    for m in must:
        r = r[r.antecedent.str.contains(m, regex=False)]
    st.markdown(f"**{len(r)} rules** match.")
    if r.empty:
        st.info("No rules match these filters. Lower the thresholds.")
        return

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.scatter(r, x="support", y="confidence", size="lift", color="lift", color_continuous_scale=SEQ[1:],
                         hover_name="antecedent", hover_data={"lift": ":.2f", "support": ":.3f", "confidence": ":.2f"},
                         labels={"support": "Support (share of days)", "confidence": "Confidence"})
        if "test_confidence" in r:
            base_rate = D.rule_check({})["train"]["base_rate"]
            fig.add_hline(y=base_rate, line_dash="dot", line_color=MUTED, annotation_text=f"base rate {base_rate:.0%}")
        fig.update_layout(title="Each bubble is a rule (bigger = higher lift)")
        chart(fig, 420)
    with c2:
        st.markdown("**Top rules in plain language**")
        for _, x in r.head(5).iterrows():
            cond = x.antecedent.replace("=", " is ").replace(" AND ", ", ")
            test = f" · held at {x.test_confidence:.0%} in 2025–26" if "test_confidence" in r and pd.notna(x.test_confidence) else ""
            if "status" in r and pd.notna(x.status):
                test += f" · <b>{x.status}</b>"
            st.markdown(f'<div class="card" style="margin-bottom:8px"><div class="s">When <b>{cond}</b>, the next day is '
                        f'high-risk <b>{x.confidence:.0%}</b> of the time ({x.lift:.1f}× the usual rate){test}.</div></div>',
                        unsafe_allow_html=True)

    if "lift_ci_lo" in r and r.lift_ci_lo.notna().any():
        r = r.assign(lift_95ci=[f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(r.lift_ci_lo, r.lift_ci_hi)])
    cols = [c for c in ["antecedent", "support", "confidence", "lift", "lift_95ci", "test_days", "test_confidence", "status"] if c in r]
    st.dataframe(r[cols], hide_index=True, width="stretch", column_config={
        "support": st.column_config.ProgressColumn(min_value=0, max_value=float(rules.support.max()), format="%.3f"),
        "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "lift": st.column_config.NumberColumn(format="%.2f"), "lift_95ci": "Lift 95% CI",
        "test_days": "Test days", "status": "Stability",
        "test_confidence": st.column_config.ProgressColumn("Confidence 2025–26", min_value=0, max_value=1, format="%.2f")})

    st.subheader("Compare rule sets")
    st.caption("Same support, confidence and lift filters applied to both sets. Set B adds yesterday's status, "
               "so the gap between them shows how much weather and calendar add beyond persistence.")
    allr = rules[(rules.support >= sup) & (rules.confidence >= conf) & (rules.lift >= lift)]
    if "set" in allr:
        comp = allr.groupby("set").agg(rules=("antecedent", "size"), best_lift=("lift", "max"),
                                       mean_confidence=("confidence", "mean"),
                                       mean_test_confidence=("test_confidence", "mean")).reset_index()
        st.dataframe(comp, hide_index=True, width="stretch", column_config={
            "best_lift": st.column_config.NumberColumn("Best lift", format="%.2f"),
            "mean_confidence": st.column_config.ProgressColumn("Mean confidence (2018–24)", min_value=0, max_value=1, format="%.2f"),
            "mean_test_confidence": st.column_config.ProgressColumn("Mean confidence (2025–26)", min_value=0, max_value=1, format="%.2f")})
    st.caption("Rules show **association, not cause**. Peak-winter rules look strong partly because winter itself is risky.")


# ================================================================ 5. INSIGHTS (ALERTS + COMBINED INSIGHTS)
def page_insights():
    hero("Insights", "Set your alert level, then see what the combined findings mean for planning.")

    st.subheader("1. Alert threshold and cost")
    src = st.radio("Alerts based on", ["M3 PM2.5 forecast (Days +1 to +3)", "Phase 1 alert model (Day +1)"],
                   horizontal=True, key="alert_src")
    ratio_default = 3
    if src.startswith("M3"):
        source_badge(fcs_real, "M3 (m3_test_predictions.csv)")
        sel = D.selected_models(fcs)
        c1, c2, c3 = st.columns(3)
        h = c1.segmented_control("Horizon", [1, 2, 3], default=1, format_func=lambda x: f"Day +{x}", key="pm_h") or 1
        ratio = c2.slider("A missed high-risk day costs … false alarms", 1, 20, ratio_default, key="pm_ratio",
                          help="The team's cost analysis uses 3: missing a high-risk day costs three false alarms.")
        st.session_state.setdefault("pm_t", D.THRESH)
        t = c3.slider("Alert when forecast PM2.5 is at least (µg/m³)", 80.0, 200.0, float(st.session_state.pm_t), 0.5,
                      key="pm_slider")
        st.session_state.pm_t = t
        d = fcs[(fcs.horizon == h)].dropna(subset=["actual"])
        g_sel = d[d.model == sel[h]]
        y, p = (g_sel.actual >= D.THRESH).values, g_sel.forecast.values
        per = d[d.model == "Persistence"]
        b = per.forecast.values >= D.THRESH
        yb = (per.actual >= D.THRESH).values
        b_tp, b_fp, b_fn = int((b & yb).sum()), int((b & ~yb).sum()), int((~b & yb).sum())
        grid = np.round(np.arange(80, 200.5, 0.5), 1)
        fa = np.array([((p >= g) & ~y).sum() for g in grid])
        miss = np.array([((p < g) & y).sum() for g in grid])
        unit, fmt = "µg/m³", "{:.1f}"
        st.caption(f"Model: **{sel[h]}** (M3's selected model for Day +{h}). A high-risk day is PM2.5 ≥ 120.5 µg/m³.")
    else:
        source_badge(preds_real, "the Phase 1 model (m5_test_predictions.csv)")
        c1, c2, c3 = st.columns(3)
        hs = sorted(int(x) for x in preds.horizon.unique())
        h = c1.segmented_control("Horizon", hs, default=hs[0], format_func=lambda x: f"Day +{x}", key="th_h") or hs[0]
        ratio = c2.slider("A missed high-risk day costs … false alarms", 1, 20, ratio_default, key="p1_ratio",
                          help="The team's cost analysis uses 3: missing a high-risk day costs three false alarms.")
        t = c3.slider("Alert threshold (probability)", 0.01, 0.99, float(st.session_state.alert_t), 0.01, key="alert_slider")
        st.session_state.alert_t = t
        d = preds[preds.horizon == h]
        y, p = d.actual.values.astype(int) == 1, d.prob.values
        today = base.set_index("date").high_risk.reindex(pd.to_datetime(d.date)).values   # baseline: tomorrow = today
        ok = ~np.isnan(today)
        b_tp = int(((today == 1) & y & ok).sum()); b_fp = int(((today == 1) & ~y & ok).sum())
        b_fn = int(((today == 0) & y & ok).sum())
        grid = np.round(np.arange(0.01, 0.995, 0.01), 2)
        fa = np.array([((p >= g) & ~y).sum() for g in grid])
        miss = np.array([((p < g) & y).sum() for g in grid])
        unit, fmt = "probability", "{:.2f}"

    cost = fa + ratio * miss
    best = grid[cost.argmin()]
    tp, fp, fn = int(((p >= t) & y).sum()), int(((p >= t) & ~y).sum()), int(((p < t) & y).sum())
    k = st.columns(5)
    card(k[0], "Caught", tp, "high-risk days alerted")
    card(k[1], "Missed", fn, "high-risk days without alert")
    card(k[2], "False alarms", fp, "alerts on safe days")
    card(k[3], "Recall · Precision", f"{tp / max(tp + fn, 1):.0%} · {tp / max(tp + fp, 1):.0%}")
    card(k[4], "Total cost", f"{fp + ratio * fn}", f"{fp} + {ratio} × {fn}")
    st.markdown("")
    if st.button(f"Use the lowest-cost threshold for this cost ratio ({fmt.format(best)})", type="primary"):
        if src.startswith("M3"):
            st.session_state.pm_t = float(best); del st.session_state["pm_slider"]
        else:
            st.session_state.alert_t = float(best); del st.session_state["alert_slider"]
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=fa, y=miss, mode="lines", line=dict(color=BLUE, width=3), customdata=grid, name="model",
                                   hovertemplate="threshold %{customdata}<br>false alarms %{x}<br>missed %{y}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[fp], y=[fn], mode="markers", marker=dict(size=14, color=ORANGE), name="your threshold"))
        fig.add_trace(go.Scatter(x=[b_fp], y=[b_fn], mode="markers+text", marker=dict(size=12, color=INK, symbol="diamond"),
                                 text=["baseline"], textposition="top right", name="tomorrow = today"))
        fig.update_layout(title="Model vs baseline: false alarms vs missed days", xaxis_title="False alarms",
                          yaxis_title="Missed high-risk days")
        chart(fig, 340)
    with c2:
        fig = go.Figure(go.Scatter(x=grid, y=cost, mode="lines", line=dict(color=INK, width=3)))
        fig.add_vline(x=best, line_dash="dash", line_color=AQUA, annotation_text=f"lowest cost {fmt.format(best)}")
        fig.add_vline(x=t, line_color=ORANGE, annotation_text="yours", annotation_position="bottom right")
        fig.update_layout(title=f"Total cost (missed = {ratio}× false alarm)", xaxis_title=f"Threshold ({unit})",
                          yaxis_title="Cost (false-alarm units)")
        chart(fig, 340)
    st.caption(f"Held-out test period: {len(y)} days (Jan 2025 – Sep 2026), {int(y.sum())} of them high-risk. "
               f"Baseline 'tomorrow = today': caught {b_tp}, missed {b_fn}, false alarms {b_fp}.")

    if src.startswith("M3"):
        st.markdown(f"**Cost analysis: threshold 120.5 vs 140 µg/m³** (missed day = {ratio}× a false alarm)")
        rows = []
        for hh in (1, 2, 3):
            g = fcs[(fcs.horizon == hh) & (fcs.model == sel[hh])].dropna(subset=["actual"])
            yy = g.actual >= D.THRESH
            r = {"Day": f"+{hh}", "Model": sel[hh]}
            for tt in (120.5, 140):
                aa = g.forecast >= tt
                f_, m_ = int((aa & ~yy).sum()), int((~aa & yy).sum())
                r[f"{tt:g}: FP / FN"] = f"{f_} / {m_}"; r[f"{tt:g}: cost"] = f_ + ratio * m_
            r["Lower cost"] = "120.5" if r["120.5: cost"] <= r["140: cost"] else "140"
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Costs are illustrative units, not rupees: the ratio is an assumption that a missed high-risk day "
                   "is worse than an unnecessary alert.")

    st.subheader("2. Combined findings")
    k = st.columns(4)
    card(k[0], "Peak season", "Nov–Jan", "83–86% of days above AQI 300")
    top_lift = rules.lift.max() if len(rules) else float("nan")
    card(k[1], "Weather trigger", "Calm + dry", f"strongest rules: {top_lift:.1f}× the usual risk")
    card(k[2], "Calendar trigger", "Diwali week", "median PM2.5 +74%")
    card(k[3], "Persistence", "93%", "of days keep yesterday's status")
    st.markdown("")

    st.subheader("3. What-if planner")
    st.caption("Describe a day using the same categories as M4's rules; see how often days like it were followed by a "
               "high-risk day (2018–2024, checked on 2025–26), and what each sector should do.")
    c = st.columns(4)
    season = c[0].selectbox("Season", ["Post-monsoon", "Winter", "Pre-monsoon", "Monsoon"],
                            help="Post-monsoon = Oct–Dec, Winter = Jan–Feb, Pre-monsoon = Mar–May, Monsoon = Jun–Sep")
    wind = c[1].radio("Wind", ["Low wind", "Mid wind", "High wind"], horizontal=True)
    rain = c[2].radio("Rain", ["No rain", "Rain"], horizontal=True)
    festival = c[3].toggle("Diwali window (±5 days)")
    res = D.rule_check({"Season": [season], "Wind": [wind], "Rain": [rain],
                        "Festival": ["Diwali window" if festival else "No Diwali window"]})
    tr = res["train"]
    if tr["days"] < 5:
        st.warning(f"Only {tr['days']} similar days in the data: too few for a reliable answer.")
    else:
        conf = tr["confidence"]
        lvl = risk_level(conf, st.session_state.alert_t)
        g1, g2 = st.columns([1, 2])
        card(g1, "Next day high-risk", f"{conf:.0%}",
             f"{tr['days']} similar days · {tr['lift']:.1f}× the usual {tr['base_rate']:.0%}"
             + (f"<br>2025–26: {res['test']['confidence']:.0%} ({res['test']['days']} days)" if res["test"]["days"] else ""),
             chip=lvl)
        g2.markdown("\n".join(f"- **{s}:** {ACTIONS[s][lvl]}" for s in ACTIONS))
    st.caption("Recommendations are planning suggestions based on past patterns in the data.")


# ================================================================ 6. MY BUSINESS (plain-language view)
CAUTION = {
    "Cautious: warn me early": dict(p=0.2, pm=100.0, note="You'll get more warnings, including some that turn out fine."),
    "Balanced": dict(p=0.5, pm=120.5, note="Warnings when a high-pollution day is more likely than not."),
    "Only when very likely": dict(p=0.8, pm=150.0, note="Fewer warnings, but you may miss some bad days."),
}
LEVEL_WORDS = {"Good": "Good air", "Satisfactory": "Satisfactory air", "Moderate": "Moderate air",
               "Poor": "Poor air", "Very Poor": "Very poor air", "Severe": "Severe air"}
def page_business():
    hero("My Business", "What the next 3 days mean for your business, in plain words.")
    c1, c2 = st.columns(2)
    sector = c1.selectbox("Your business", list(ACTIONS))
    caution = c2.radio("When should we warn you?", list(CAUTION), index=1)
    cfg = CAUTION[caution]
    st.caption(cfg["note"] + " We warn when the PM2.5 forecast, or (for tomorrow) the alert model, crosses your level.")

    dates = sorted(preds.loc[preds.horizon == 1, "date"].unique())
    fc_dates = set(fcs.loc[fcs.horizon == 3, "date"])
    dates = [x for x in dates if x in fc_dates] or dates
    st.session_state.setdefault("asof", dates[-1])
    diwali = base.loc[(base.days_from_diwali == 0) & (base.date >= D.TEST_START), "date"]
    win = preds[(preds.horizon == 1) & pd.to_datetime(preds.target_date).dt.month.isin([11, 12, 1])]
    jumps = {"Latest date": dates[-1],
             "Day before Diwali 2025": (diwali.iloc[0] - pd.Timedelta(days=1)) if len(diwali) else dates[-1],
             "A winter week": win.loc[win.prob.idxmax(), "date"] if len(win) else dates[-1]}
    for col, (lbl, dt) in zip(st.columns(len(jumps)), jumps.items()):
        if col.button(lbl, width="stretch", key=f"biz_{lbl}"):
            st.session_state.asof = pd.Timestamp(dt)
    asof = st.select_slider("Day you are planning from", options=dates, key="asof",
                            format_func=lambda x: pd.Timestamp(x).strftime("%a %d %b %Y"))
    st.caption("By default this is the last day in the data (30 Sep 2026); the dashboard has no live feed.")

    sel = D.selected_models(fcs)
    p1 = preds[(preds.date == asof) & (preds.horizon == 1)]
    cols = st.columns(3)
    alerts = {}
    for col, h in zip(cols, D.HORIZONS):
        f = fcs[(fcs.date == asof) & (fcs.horizon == h) & (fcs.model == sel.get(h))]
        if f.empty:
            card(col, f"Day +{h}", "–", "no forecast for this day"); continue
        f = f.iloc[0]
        pm = float(f.forecast)
        cat = pd.cut([pm], D.CATEGORY_EDGES, labels=D.CATEGORIES, right=False)[0]
        if h == 1 and len(p1):
            prob = float(p1.prob.iloc[0]); alert = prob >= cfg["p"] or pm >= cfg["pm"]   # either signal warns
            chance = f"Chance of a high-pollution day: <b>{prob:.0%}</b><br>"
        else:
            alert = pm >= cfg["pm"]; chance = ""
        alerts[h] = alert
        key = ("Severe" if cat == "Severe" else "High") if alert else ("Watch" if cat in ("Poor", "Very Poor") else "Low")
        head = "Prepare" if alert else ("Keep an eye" if key == "Watch" else "No action needed")
        card(col, f"{pd.Timestamp(f.target_date):%A %d %b}", head,
             f"{LEVEL_WORDS[cat]} expected (AQI ≈ {D.pm25_to_aqi(pm):.0f})<br>{chance}<b>What to do:</b> {ACTIONS[sector][key]}",
             chip=key)

    st.subheader("Why we think so")
    items = D.day_items(asof)
    match = rules[[set(x.split(" AND ")) <= items for x in rules.antecedent]] if items else rules.iloc[0:0]
    match = match.sort_values(["lift", "confidence"], ascending=False)
    if match.empty:
        st.markdown(f"None of the association rules match the conditions on {pd.Timestamp(asof):%d %b}, so there is "
                    "no strong weather or calendar warning sign for tomorrow.")
    else:
        for _, x in match.head(2).iterrows():
            cond = x.antecedent.replace(" AND ", ", ").lower()
            test = f", and {x.test_confidence:.0%} of the time in 2025–26" if pd.notna(x.get("test_confidence")) else ""
            st.markdown(f'<div class="card" style="margin-bottom:8px"><div class="s">Days like {pd.Timestamp(asof):%d %b} '
                        f'(<b>{cond}</b>) were followed by a high-pollution day <b>{x.confidence:.0%}</b> of the time in '
                        f'2018–24{test}. That is {x.lift:.1f}× the usual rate.</div></div>', unsafe_allow_html=True)
    st.caption("Based on M4's association rules. They describe what usually happens together, not what causes it.")

    st.subheader(f"Preparation checklist: {sector.lower()}")
    st.caption("From the team's PM2.5 Risk Preparation Playbook. Highlighted steps match the warnings above.")
    now = {"2–3 days ahead": any(alerts.get(x) for x in (2, 3)), "1 day ahead": bool(alerts.get(1)),
           "If risk stays high": sum(bool(v) for v in alerts.values()) >= 2}
    for col, (phase, steps) in zip(st.columns(3), PLAYBOOK[sector].items()):
        tag = '<span class="chip" style="background:#ec835a33;border:1px solid #ec835a">Now</span> ' if now[phase] else ""
        col.markdown(f'<div class="card"><div class="k">{tag}{phase}</div><div class="s">'
                     + "".join(f"• {s}<br>" for s in steps) + "</div></div>", unsafe_allow_html=True)

    grap = D.load_grap()
    if grap is not None:
        st.subheader("Official restrictions (GRAP, CAQM)")
        g = grap[grap["sector"].str.lower() == sector.lower()] if "sector" in grap else grap
        st.dataframe(g if len(g) else grap, hide_index=True, width="stretch")


# ================================================================ 7. DATASET
DICTIONARY = [
    ("date", "Day (one row per day)", "date"),
    ("pm25_target", "PM2.5 value behind the label: reference station, or city mean on days with 10+ stations", "µg/m³"),
    ("target_source", "reference / city_fallback / missing", "text"),
    ("high_risk", "Target: 1 if PM2.5-based AQI > 300 (pm25_target ≥ 120.5)", "0/1"),
    ("pm25_mean, pm25_median", "City mean / median PM2.5 across Delhi stations", "µg/m³"),
    ("pm25_n_stations", "Stations reporting PM2.5 that day", "count"),
    ("pm25_ref_station", "PM2.5 at the single long-running reference station", "µg/m³"),
    ("pm10_mean, pm10_median", "City mean / median PM10", "µg/m³"),
    ("temp_mean, temp_min, temp_max", "Daily temperature", "°C"),
    ("humidity_mean", "Relative humidity", "%"),
    ("dewpoint_mean", "Dew point", "°C"),
    ("wind_speed_mean, wind_gust_max", "Mean wind speed, strongest gust", "m/s"),
    ("wind_dir_deg", "Direction the wind comes from", "degrees"),
    ("pressure_mean", "Surface pressure", "hPa"),
    ("precip_sum", "Total rain", "mm"),
    ("cloud_mean", "Cloud cover", "%"),
    ("month, day_of_week, is_weekend, year", "Calendar fields (day_of_week 0 = Monday)", "—"),
    ("season", "winter (Jan–Feb), pre_monsoon (Mar–May), monsoon (Jun–Sep), post_monsoon (Oct–Dec)", "text"),
    ("days_from_diwali, is_festival_window", "Days to the nearest Diwali; 1 if within ±5 days", "days / 0-1"),
]


def page_dataset():
    hero("Dataset", "The final validated project dataset behind every page of this dashboard.")
    k = st.columns(4)
    card(k[0], "Records", f"{len(base):,}", f"days, {base.date.min():%d %b %Y} – {base.date.max():%d %b %Y}")
    card(k[1], "Variables", f"{base.shape[1] - 3}", "pollution, weather, calendar, target")
    card(k[2], "Days with a target", f"{base.high_risk.notna().sum():,}", f"{base.high_risk.isna().sum()} excluded, never imputed")
    card(k[3], "High-risk days", f"{base.high_risk.mean():.1%}", "PM2.5-based AQI > 300")
    st.markdown("")
    st.markdown("**Sources:** OpenAQ v3 API (PM2.5 and PM10 from Delhi stations) and the Open-Meteo archive API "
                "(hourly weather, aggregated to days); calendar and Diwali dates built by the team. "
                "**Train** 2018–2024, **test** Jan 2025 – Sep 2026.")

    st.subheader("Validation checks")
    t = base.dropna(subset=["pm25_target"])
    full_days = pd.date_range(base.date.min(), base.date.max())
    checks = [
        ("One row per day, no duplicate dates", base.date.duplicated().sum() == 0),
        ("Every calendar day present", len(full_days) == len(base)),
        ("high_risk matches pm25_target ≥ 120.5 on every day", ((t.pm25_target >= D.THRESH).astype(float) == t.high_risk).all()),
        ("Future-leaking columns removed (next_pm25, prev_pm25, high_risk_city)",
         not any(c in base for c in ["next_pm25", "prev_pm25", "high_risk_city"])),
        ("Target missing only where target_source = missing", (base.high_risk.isna() == (base.target_source == "missing")).all()),
    ]
    st.dataframe(pd.DataFrame([(n, "✅ pass" if ok else "❌ fail") for n, ok in checks], columns=["Check", "Result"]),
                 hide_index=True, width="stretch")

    st.subheader("Explore the data")
    c1, c2, c3 = st.columns(3)
    yrs = c1.slider("Years", int(base.year.min()), int(base.year.max()), (2024, 2026), key="ds_years")
    months = c2.multiselect("Months", list(range(1, 13)), format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%b"),
                            placeholder="All months")
    cats = c3.multiselect("Category", D.CATEGORIES, placeholder="All categories")
    sub = base[base.year.between(*yrs)]
    if months:
        sub = sub[sub.date.dt.month.isin(months)]
    if cats:
        sub = sub[sub.category.isin(cats)]
    m = st.columns(4)
    card(m[0], "Days selected", f"{len(sub):,}")
    card(m[1], "High-risk share", f"{sub.high_risk.mean():.0%}" if sub.high_risk.notna().any() else "–")
    card(m[2], "Median PM2.5", f"{sub.pm25_target.median():.0f}" if sub.pm25_target.notna().any() else "–", "µg/m³")
    card(m[3], "Mean wind", f"{sub.wind_speed_mean.mean():.1f}" if len(sub) else "–", "m/s")
    st.markdown("")
    show = ["date", "pm25_target", "category", "high_risk", "target_source", "pm25_mean", "pm10_mean", "wind_speed_mean",
            "humidity_mean", "temp_mean", "precip_sum", "season", "is_festival_window"]
    st.dataframe(sub[show], hide_index=True, width="stretch", height=320,
                 column_config={"date": st.column_config.DateColumn(format="DD MMM YYYY"),
                                **{c: st.column_config.NumberColumn(format="%.1f") for c in
                                   ["pm25_target", "pm25_mean", "pm10_mean", "wind_speed_mean", "humidity_mean",
                                    "temp_mean", "precip_sum"]}})

    st.subheader("Data dictionary")
    st.dataframe(pd.DataFrame(DICTIONARY, columns=["Column", "Meaning", "Unit"]), hide_index=True, width="stretch")
    st.markdown("**Limitations:** official AQI also uses PM10; on 22 days PM10 alone could have pushed it above 300 "
                "(21 in Mar–Jul). There is no live feed, so the Control room shows the last date in the data. "
                "Rules are associations, not causes.")
    with st.expander("Team result files used by this dashboard"):
        rows = [(n, o, "✅ loaded" if ok else "⏳ placeholder in use") for n, o, ok in D.file_status()]
        st.dataframe(pd.DataFrame(rows, columns=["File", "Owner", "Status"]), hide_index=True, width="stretch")


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon=":material/monitor_heart:", default=True),
    st.Page(page_dataset, title="Dataset", icon=":material/database:"),
    st.Page(page_forecast, title="Forecast (Method 1)", icon=":material/show_chart:"),
    st.Page(page_rules, title="Rules (Method 2)", icon=":material/account_tree:"),
    st.Page(page_insights, title="Insights", icon=":material/lightbulb:"),
    st.Page(page_business, title="My business", icon=":material/storefront:"),
])
with st.sidebar:
    st.markdown("### Delhi Air-Risk Planner")
    st.caption("Team 11 · 23CSE452 Business Analytics · Review 2")
    n_ph = sum(not ok for _, _, ok in D.file_status())
    st.markdown(f'<span class="badge">{n_ph} placeholder file(s) in use</span>' if n_ph else
                '<span class="badge ok">All team files loaded</span>', unsafe_allow_html=True)
pg.run()
