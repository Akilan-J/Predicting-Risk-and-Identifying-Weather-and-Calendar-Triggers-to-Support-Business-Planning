import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Cell 0: Title
add_md("""# M4 — Association Rule Mining: High-Risk PM2.5 Days
### Identifying Weather, Season, and Persistence Regimes for Operational Alerts

This notebook implements the finalized **M4 Association Rule Mining (ARM)** pipeline using `mlxtend` FP-Growth, addressing all operational constraints and feasibility criteria:

1. **Temporal Day-Pairing**: Weather features at day $t$ predict the alert status $\\text{high\\_risk}_{t+1}$ at day $t+1$. Missing labels are dropped strictly without zero-filling.
2. **Frozen Quantile Binning**: Tercile bin edges (wind, humidity, temperature, pressure) and categorical items (rain, season, Diwali window, weekend, yesterday's status) are fitted exclusively on pre-2025 training data and locked for test evaluation.
3. **IMD Season Standard**: Uses the authoritative Indian Meteorological Department (IMD) season definition (Winter: Jan–Feb; Pre-monsoon: Mar–May; Monsoon: Jun–Sep; Post-monsoon: Oct–Dec).
4. **mlxtend Mining with Starting & Tuned Filters**: Initial filters of support $\\ge 5\\%$, confidence $\\ge 0.60$, and lift $\\ge 1.20$. Explores support tuning down to $2\\%$ to capture episodic events like the Diwali festival window.
5. **Season Base-Rate & Headroom Analysis**: Accounts for extreme base-rate disparities (0.4% in monsoon vs. 70.4% in winter) and computes **within-season lift**.
6. **Autoregressive Persistence Baseline (Set B)**: Evaluates Set B rules by their **incremental gain** over the "Yesterday was high" parent rule ($\\text{Conf}=0.869$, $\\text{Lift}=2.95$).
7. **Robust Statistical Inference**: Attaches one-sided Chi-square/Fisher exact $p$-values with **Benjamini–Hochberg False Discovery Rate (FDR)** control ($\\alpha=0.05$) and **vectorized moving-block bootstrap** 95% confidence intervals ($b=7$ days).
8. **Redundancy Pruning**: Drops any complex rule whose lift does not strictly exceed all of its simpler sub-rules.
9. **Out-of-Sample Verification**: Evaluates all rules against the frozen 2025–26 test set ($N=633$) to distinguish stable rules from base-rate-driven confidence drift.
10. **Descriptive Dashboard Output**: Generates an actionable, ranked dashboard table for public health advisories and pollution control authorities.""")

# Cell 1: Setup & Imports
add_code(r"""import os, sys, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from mlxtend.frequent_patterns import fpgrowth, association_rules

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---- Paths and Constants ----
DATA_CANDIDATES = [
    Path(r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study\m4\m4_full_engineered.csv"),
    Path(".") / "m4_full_engineered.csv",
    Path("../group study/m4/m4_full_engineered.csv")
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
assert DATA_PATH is not None, "m4_full_engineered.csv not found in candidates"

OUT_DIR = Path("m4_arm_outputs")
OUT_DIR.mkdir(exist_ok=True)

THRESH = 120.5                      # PM2.5 threshold for high-risk (AQI > 300)
SPLIT_DATE = pd.Timestamp("2025-01-01")
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.20
BOOT_B = 1000
BOOT_BLOCK = 7                       # 7-day blocks for weekly dependence

# Styling
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10.5, "axes.titlesize": 12.0, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 120,
})

def save_fig(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.show()

print(f"Loaded configuration. Data source: {DATA_PATH}")""")

# Cell 2: Data Loading & Day Pairing
add_md("""## 1. Load Data & Calendar Day-Pairing

### Temporal Alignment Rules:
- **Antecedents**: Features observed on Day $t$ (weather, season, calendar, and Day $t$ pollution status).
- **Consequent**: Next-day label $\\text{high\\_risk}_{t+1}$ on Day $t+1$.
- **Integrity**: Missing labels are never imputed with $0$. If either Day $t$ or Day $t+1$ target is missing, the pair is dropped.""")

add_code("""raw = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")

# Verify daily continuity
date_diff = raw.index.to_series().diff().dropna()
assert (date_diff == pd.Timedelta(days=1)).all(), "Gaps in daily calendar sequence"

# Calendar-aligned pairs
pairs = pd.DataFrame(index=raw.index)
pairs["target_today"] = raw["high_risk"]
pairs["target_next"]  = raw["high_risk"].shift(-1, freq="D")
pairs["season_today"] = raw["season"]
pairs["season_next"]  = raw["season"].shift(-1, freq="D")
pairs["is_weekend"]   = raw["is_weekend"]
pairs["is_festival_window"] = raw["is_festival_window"]

WX_COLS = ["temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean", "precip_sum"]
for col in WX_COLS:
    pairs[col] = raw[col]

# Drop incomplete pairs
clean_pairs = pairs[pairs["target_today"].notna() & pairs["target_next"].notna()].copy()
clean_pairs["target_today"] = clean_pairs["target_today"].astype(int)
clean_pairs["target_next"]  = clean_pairs["target_next"].astype(int)

train = clean_pairs[clean_pairs.index < SPLIT_DATE].copy()
test  = clean_pairs[clean_pairs.index >= SPLIT_DATE].copy()

train_base = train["target_next"].mean()
test_base  = test["target_next"].mean()

print(f"Total days in dataset: {len(raw)} ({raw.index.min().date()} -> {raw.index.max().date()})")
print(f"Valid paired days   : {len(clean_pairs)} pairs")
print(f"  Training (pre-2025): {len(train)} day-pairs | Base rate = {train_base:.4f} ({train_base*100:.1f}%)")
print(f"  Test (2025-26)     : {len(test)} day-pairs  | Base rate = {test_base:.4f} ({test_base*100:.1f}%)")""")

# Cell 3: Season Analysis
add_md("""## 2. Season Dominance & Headroom Analysis

### Season Definition:
The dataset uses the official **India Meteorological Department (IMD)** meteorological definition:
- **Winter**: January & February
- **Pre-monsoon**: March, April, May
- **Monsoon**: June, July, August, September
- **Post-monsoon**: October, November, December

### The Season Headroom Problem:
In Winter, the training base rate is **70.4%**. The maximum possible lift achievable by any rule is:
$$\\text{Max Lift} = \\frac{1.0}{\\text{Base Rate}} = \\frac{1.0}{0.704} \\approx 1.42$$
Hence, an overall lift threshold of $1.20$ inside winter leaves almost no headroom ($1.20$ to $1.42$). Therefore, we evaluate both **overall lift** and **within-season lift**.""")

add_code("""SEASON_DISPLAY = {
    "winter": "Winter (Jan-Feb)",
    "pre_monsoon": "Pre-monsoon (Mar-May)",
    "monsoon": "Monsoon (Jun-Sep)",
    "post_monsoon": "Post-monsoon (Oct-Dec)"
}

season_stats = []
for s_code, s_name in SEASON_DISPLAY.items():
    tr_sub = train[train["season_next"] == s_code]
    te_sub = test[test["season_next"] == s_code]
    tr_rate = tr_sub["target_next"].mean() if len(tr_sub) else 0.0
    te_rate = te_sub["target_next"].mean() if len(te_sub) else 0.0
    max_lift = 1.0 / tr_rate if tr_rate > 0 else np.inf
    season_stats.append({
        "Season": s_name,
        "Train Pairs": len(tr_sub),
        "Train Base Rate": f"{tr_rate:.3f}",
        "Max Possible Lift": f"{max_lift:.2f}",
        "Test Pairs": len(te_sub),
        "Test Base Rate": f"{te_rate:.3f}"
    })

pd.DataFrame(season_stats)""")

# Cell 4: Binning
add_md("""## 3. Discretization & Binning (Frozen on Train)

Continuous weather features are discretized into **terciles** (quantile cuts at $0, 33.3\\%, 66.7\\%, 100\\%$).
- The cut points are **strictly computed on pre-2025 training data** and frozen.
- The test set (2025–26) is binned using these exact frozen thresholds to prevent temporal data leakage.
- Rain is binary: No rain ($=0$) vs Rain ($>0$).
- Calendar items include IMD seasons, Diwali festival window (`is_festival_window`), and Weekend (`is_weekend`).""")

add_code("""# Quantile cuts on pre-2025 train data
bin_edges = {}
for col in ["temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean"]:
    q = np.quantile(train[col], [0, 1/3, 2/3, 1])
    bin_edges[col] = q

def encode_items(data):
    cols = {}
    # Wind speed terciles
    q = bin_edges["wind_speed_mean"]
    cols["low_wind"] = data["wind_speed_mean"] <= q[1]
    cols["mid_wind"] = (data["wind_speed_mean"] > q[1]) & (data["wind_speed_mean"] <= q[2])
    cols["high_wind"] = data["wind_speed_mean"] > q[2]

    # Humidity terciles
    q = bin_edges["humidity_mean"]
    cols["low_humidity"] = data["humidity_mean"] <= q[1]
    cols["mid_humidity"] = (data["humidity_mean"] > q[1]) & (data["humidity_mean"] <= q[2])
    cols["high_humidity"] = data["humidity_mean"] > q[2]

    # Temperature terciles
    q = bin_edges["temp_mean"]
    cols["cold"] = data["temp_mean"] <= q[1]
    cols["mild_temp"] = (data["temp_mean"] > q[1]) & (data["temp_mean"] <= q[2])
    cols["hot"] = data["temp_mean"] > q[2]

    # Pressure terciles
    q = bin_edges["pressure_mean"]
    cols["low_pressure"] = data["pressure_mean"] <= q[1]
    cols["mid_pressure"] = (data["pressure_mean"] > q[1]) & (data["pressure_mean"] <= q[2])
    cols["high_pressure"] = data["pressure_mean"] > q[2]

    # Rain
    cols["no_rain"] = data["precip_sum"] == 0
    cols["rain"] = data["precip_sum"] > 0

    # Seasons
    for s_code in ["winter", "post_monsoon", "pre_monsoon", "monsoon"]:
        cols[f"season_{s_code}"] = data["season_today"] == s_code

    # Diwali window & Weekend
    cols["diwali_window"] = data["is_festival_window"] == 1
    cols["weekend"] = data["is_weekend"] == 1

    # Autoregressive persistence
    cols["yesterday_high"] = data["target_today"] == 1

    # Consequent
    cols["high_risk_next"] = data["target_next"] == 1
    return pd.DataFrame(cols, index=data.index)

train_enc = encode_items(train)
test_enc  = encode_items(test)

ITEM_LABELS = {
    "low_wind": "Low wind", "mid_wind": "Mid wind", "high_wind": "High wind",
    "low_humidity": "Low humidity", "mid_humidity": "Mid humidity", "high_humidity": "High humidity",
    "cold": "Cold", "mild_temp": "Mild temp", "hot": "Hot",
    "low_pressure": "Low pressure", "mid_pressure": "Mid pressure", "high_pressure": "High pressure",
    "no_rain": "No rain", "rain": "Rain",
    "season_winter": "Winter", "season_post_monsoon": "Post-monsoon",
    "season_pre_monsoon": "Pre-monsoon", "season_monsoon": "Monsoon",
    "diwali_window": "Diwali window", "weekend": "Weekend",
    "yesterday_high": "Yesterday was high-risk"
}

ITEM_GROUPS = {
    "low_wind": "wind", "mid_wind": "wind", "high_wind": "wind",
    "low_humidity": "hum", "mid_humidity": "hum", "high_humidity": "hum",
    "cold": "temp", "mild_temp": "temp", "hot": "temp",
    "low_pressure": "pres", "mid_pressure": "pres", "high_pressure": "pres",
    "no_rain": "precip", "rain": "precip",
    "season_winter": "season", "season_post_monsoon": "season",
    "season_pre_monsoon": "season", "season_monsoon": "season",
    "diwali_window": "diwali", "weekend": "weekend", "yesterday_high": "persist"
}

print("Discretization complete. Frozen train thresholds:")
for col, q in bin_edges.items():
    print(f"  {col:16s}: terciles = [{q[1]:.2f}, {q[2]:.2f}]")""")

# Cell 5: Rule Mining with mlxtend FP-Growth
add_md("""## 4. Mining Association Rules with mlxtend FP-Growth

- **FP-Growth Algorithm**: Efficient prefix-tree representation of frequent itemsets.
- **Starting Filters**:
  - Minimum Support $\\ge 5\\%$ of days ($N \\ge 108$ training days)
  - Minimum Confidence $\\ge 0.60$
  - Minimum Lift $\\ge 1.20$
- **Target Consequent**: $\\text{high\\_risk}_{t+1}$.
- **Sets**:
  - **Set A**: Weather, Season, and Calendar items $\\to \\text{high\\_risk}_{t+1}$ (no persistence).
  - **Set B**: Weather, Season, Calendar, **plus** Persistence (`yesterday_high`) $\\to \\text{high\\_risk}_{t+1}$.""")

add_code("""# Mine frequent itemsets
frequent_itemsets = fpgrowth(train_enc, min_support=MIN_SUPPORT, use_colnames=True)
print(f"Frequent itemsets mined: {len(frequent_itemsets)}")

raw_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
print(f"Association rules with confidence >= {MIN_CONFIDENCE}: {len(raw_rules)}")

# Filter to target consequent: high_risk_next
target_consequent = frozenset(["high_risk_next"])
filtered_rules = raw_rules[
    (raw_rules["consequents"] == target_consequent) & 
    (raw_rules["lift"] >= MIN_LIFT)
].copy()

# Remove conflicting items within antecedents (e.g., cold + hot)
def has_conflicts(itemset):
    groups = [ITEM_GROUPS[it] for it in itemset]
    return len(groups) != len(set(groups))

filtered_rules["conflict"] = filtered_rules["antecedents"].apply(has_conflicts)
filtered_rules = filtered_rules[~filtered_rules["conflict"]].copy()

# Split into Set A (no yesterday_high) vs Set B (contains yesterday_high)
filtered_rules["has_yesterday"] = filtered_rules["antecedents"].apply(lambda s: "yesterday_high" in s)
filtered_rules["set"] = np.where(filtered_rules["has_yesterday"], "B", "A")
filtered_rules["k"] = filtered_rules["antecedents"].apply(len)

print(f"Target rules predicting next_high_risk: Total={len(filtered_rules)} | Set A={sum(filtered_rules['set']=='A')} | Set B={sum(filtered_rules['set']=='B')}")""")

# Cell 6: De-duplication
add_md("""## 5. De-duplication: Pruning Redundant Sub-Rules

A rule $X \\to Y$ is considered **redundant** if there exists a simpler sub-rule $X' \\subset X$ such that:
$$\\text{Lift}(X) \\le \\text{Lift}(X')$$
In other words, adding extra conditions must provide an incremental lift benefit. If adding an item does not improve lift over the parent rule, the more complex rule is pruned.""")

add_code("""def prune_redundant(df_rules):
    lift_map = {frozenset(r["antecedents"]): r["lift"] for _, r in df_rules.iterrows()}
    keep = []
    for _, r in df_rules.iterrows():
        items = list(r["antecedents"])
        if len(items) == 1:
            keep.append(True)
            continue
        dominated = False
        for m in range(1, len(items)):
            for sub in itertools.combinations(items, m):
                sub_key = frozenset(sub)
                if sub_key in lift_map and lift_map[sub_key] >= r["lift"]:
                    dominated = True
                    break
            if dominated:
                break
        keep.append(not dominated)
    return df_rules[keep].reset_index(drop=True)

before_pruning = len(filtered_rules)
rules_pruned = prune_redundant(filtered_rules)
print(f"De-duplication pruning: {before_pruning} -> {len(rules_pruned)} rules (pruned {before_pruning - len(rules_pruned)})")
print(f"  Final Set A rules: {sum(rules_pruned['set']=='A')}")
print(f"  Final Set B rules: {sum(rules_pruned['set']=='B')}")""")

# Cell 7: Significance Testing
add_md("""## 6. Statistical Significance & FDR Control

- **Contingency Testing**: One-sided Chi-square test (with Yates correction) for cells $\\ge 5$, or Fisher exact test for small cells.
- **Benjamini-Hochberg False Discovery Rate (FDR)**: Controls family-wise error across simultaneous hypotheses at $\\alpha = 0.05$.""")

add_code("""def compute_pvalue(antecedent, items_df):
    mask = items_df[list(antecedent)].all(axis=1).values
    a = int((mask & (items_df["high_risk_next"].values == 1)).sum())
    b = int((mask & (items_df["high_risk_next"].values == 0)).sum())
    c = int((~mask & (items_df["high_risk_next"].values == 1)).sum())
    d = int((~mask & (items_df["high_risk_next"].values == 0)).sum())

    table = np.array([[a, b], [c, d]])
    if table.min() < 5:
        _, p = stats.fisher_exact(table, alternative="greater")
    else:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
        if a / max(mask.sum(), 1) < items_df["high_risk_next"].mean():
            p = 1.0
        else:
            p = p / 2.0  # one-sided p-value
    return p

raw_pvals = [compute_pvalue(r["antecedents"], train_enc) for _, r in rules_pruned.iterrows()]
rules_pruned["p_value_raw"] = raw_pvals

def multipletests_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    cummin = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_p = np.clip(cummin, 0, 1)
    adjusted = np.empty_like(adj_p)
    adjusted[order] = adj_p
    reject = adjusted <= alpha
    return reject, adjusted

reject, adj_pvals = multipletests_bh(raw_pvals, alpha=0.05)
rules_pruned["p_value_bh"] = adj_pvals
rules_pruned["significant_bh05"] = reject
print(f"Significant rules at BH FDR alpha=0.05: {reject.sum()} / {len(rules_pruned)}")""")

# Cell 8: Bootstrap CIs
add_md("""## 7. Moving-Block Bootstrap 95% Confidence Intervals

To account for temporal dependence and weekly autocorrelation, we perform a **moving-block bootstrap** with block size $b = 7$ days ($B = 1000$ resamples) to estimate robust $95\\%$ confidence intervals for lift.""")

add_code("""n_train = len(train_enc)
n_blocks = int(np.ceil(n_train / BOOT_BLOCK))
starts = np.arange(0, n_train - BOOT_BLOCK + 1)
boot_indices = np.empty((BOOT_B, n_train), dtype=int)
for b in range(BOOT_B):
    chosen = np.random.choice(starts, size=n_blocks, replace=True)
    idx = np.concatenate([np.arange(s, min(s + BOOT_BLOCK, n_train)) for s in chosen])[:n_train]
    boot_indices[b] = idx

y_all = train_enc["high_risk_next"].values.astype(float)
y_boot = y_all[boot_indices]       # shape (BOOT_B, n_train)
base_boot = y_boot.mean(axis=1)     # shape (BOOT_B,)

def fast_bootstrap_lift(antecedent):
    item_cols = list(antecedent)
    mask_full = train_enc[item_cols].all(axis=1).values
    mask_boot = mask_full[boot_indices]
    counts = mask_boot.sum(axis=1)
    hits = (mask_boot & (y_boot == 1.0)).sum(axis=1)
    valid = (counts > 0) & (base_boot > 0)
    if valid.sum() < 50:
        return np.nan, np.nan
    conf = hits[valid] / counts[valid]
    lifts = conf / base_boot[valid]
    return np.percentile(lifts, 2.5), np.percentile(lifts, 97.5)

print("Computing bootstrap CIs for all mined rules ...")
boot_cis = [fast_bootstrap_lift(r["antecedents"]) for _, r in rules_pruned.iterrows()]
rules_pruned["lift_ci_lo"] = [c[0] for c in boot_cis]
rules_pruned["lift_ci_hi"] = [c[1] for c in boot_cis]
print("Bootstrap CIs complete.")""")

# Cell 9: Out-of-sample test and metrics
add_md("""## 8. Out-of-Sample Verification & Autoregressive Gain

- **Frozen 2025–26 Test Evaluation**: Rules are evaluated on unseen data to test whether lift remains $>1.0$.
- **Set B Baseline Comparison**: Compares against the univariate "Yesterday was high" parent rule ($\\text{Conf} = 0.869$, $\\text{Lift} = 2.95$).""")

add_code("""def evaluate_test(antecedent, test_items_df):
    item_cols = list(antecedent)
    mask = test_items_df[item_cols].all(axis=1).values
    n_match = int(mask.sum())
    if n_match == 0:
        return np.nan, np.nan, 0
    conf = test_items_df.loc[mask, "high_risk_next"].mean()
    base = test_items_df["high_risk_next"].mean()
    lift = conf / base if base > 0 else np.nan
    return conf, lift, n_match

test_evals = [evaluate_test(r["antecedents"], test_enc) for _, r in rules_pruned.iterrows()]
rules_pruned["test_conf"] = [e[0] for e in test_evals]
rules_pruned["test_lift"] = [e[1] for e in test_evals]
rules_pruned["test_n"] = [e[2] for e in test_evals]

# Within-Season Lift
def compute_within_season_lift(antecedent, train_df, train_items_df):
    season_items = [it for it in antecedent if it.startswith("season_")]
    if not season_items:
        return np.nan
    s_code = season_items[0].replace("season_", "")
    s_mask = (train_df["season_today"] == s_code).values
    s_base = train_items_df.loc[s_mask, "high_risk_next"].mean()
    if s_base == 0:
        return np.nan
    rule_mask = train_items_df[list(antecedent)].all(axis=1).values
    rule_conf = train_items_df.loc[rule_mask, "high_risk_next"].mean()
    return rule_conf / s_base

rules_pruned["lift_within_season"] = [
    compute_within_season_lift(r["antecedents"], train, train_enc)
    for _, r in rules_pruned.iterrows()
]

# Set B Gain over Yesterday Baseline
yesterday_mask = train_enc["yesterday_high"] == 1
yesterday_conf = train_enc.loc[yesterday_mask, "high_risk_next"].mean()
yesterday_lift = yesterday_conf / train_base

rules_pruned["gain_conf_vs_yesterday"] = np.nan
rules_pruned["gain_lift_vs_yesterday"] = np.nan
set_b_mask = rules_pruned["set"] == "B"
rules_pruned.loc[set_b_mask, "gain_conf_vs_yesterday"] = rules_pruned.loc[set_b_mask, "confidence"] - yesterday_conf
rules_pruned.loc[set_b_mask, "gain_lift_vs_yesterday"] = rules_pruned.loc[set_b_mask, "lift"] - yesterday_lift

# Plain Language Text
def make_rule_text(antecedent):
    items = sorted(list(antecedent), key=lambda x: (x == "yesterday_high", x))
    return " + ".join([ITEM_LABELS[c] for c in items])

rules_pruned["rule_text"] = rules_pruned["antecedents"].apply(make_rule_text)

rules_pruned["status"] = "Drifting"
stable_mask = (
    (rules_pruned["test_lift"] >= 1.0) & 
    (rules_pruned["test_conf"] >= 0.50) & 
    (rules_pruned["test_n"] >= 5)
)
rules_pruned.loc[stable_mask, "status"] = "Stable in 2025-26"
rules_pruned.loc[rules_pruned["test_n"] < 5, "status"] = "Insufficient Test Days (<5)"

rules_pruned = rules_pruned.sort_values(["set", "lift"], ascending=[True, False]).reset_index(drop=True)""")

# Cell 10: Dashboard Table Export
add_md("""## 9. Ranked Dashboard Rule Table

The table below is formatted for operational deployment in warning dashboards and public health portals.""")

add_code("""dashboard_cols = [
    "rule_text", "set", "k",
    "support", "confidence", "lift",
    "lift_ci_lo", "lift_ci_hi",
    "lift_within_season",
    "test_conf", "test_lift", "test_n",
    "p_value_bh", "significant_bh05",
    "gain_conf_vs_yesterday", "gain_lift_vs_yesterday",
    "status"
]
dashboard = rules_pruned[dashboard_cols].copy()

# Save CSVs
dashboard.to_csv(OUT_DIR / "m4_arm_dashboard.csv", index=False)
rules_pruned.to_csv(OUT_DIR / "m4_arm_rules_full.csv", index=False)

print("Top 10 Set A Rules (Weather & Season):")
display(dashboard[dashboard["set"] == "A"].head(10)[["rule_text", "support", "confidence", "lift", "test_conf", "test_lift", "test_n", "status"]])

print("\\nTop 10 Set B Rules (Weather + Persistence):")
display(dashboard[dashboard["set"] == "B"].head(10)[["rule_text", "support", "confidence", "lift", "gain_conf_vs_yesterday", "gain_lift_vs_yesterday", "test_conf", "test_lift", "status"]])""")

# Cell 11: Support Tuning
add_md("""## 10. Filter Tuning: Uncovering Episodic Festival (Diwali) Rules

When the support threshold is relaxed from the starting $5\\%$ filter down to $2\\%$, episodic regimes such as the **Diwali festival window** emerge.""")

add_code("""tuned_supports = [0.05, 0.03, 0.02]
tuning_results = []
for sup in tuned_supports:
    f_tuned = fpgrowth(train_enc, min_support=sup, use_colnames=True)
    r_tuned = association_rules(f_tuned, metric="confidence", min_threshold=MIN_CONFIDENCE)
    tgt_tuned = r_tuned[(r_tuned["consequents"] == target_consequent) & (r_tuned["lift"] >= MIN_LIFT)]
    diwali_cnt = tgt_tuned["antecedents"].apply(lambda s: "diwali_window" in s).sum()
    weekend_cnt = tgt_tuned["antecedents"].apply(lambda s: "weekend" in s).sum()
    tuning_results.append({
        "min_support": sup,
        "frequent_itemsets": len(f_tuned),
        "total_rules": len(tgt_tuned),
        "diwali_rules": diwali_cnt,
        "weekend_rules": weekend_cnt
    })
df_tuning = pd.DataFrame(tuning_results)
display(df_tuning)""")

# Cell 12: Visualisations
add_md("""## 11. Diagnostic Visualisations

1. **Train Lift vs. 2025–26 Test Lift**: Evaluates transferability and out-of-sample stability.
2. **Confidence Drift**: Shows how seasonal base-rate shifts impact apparent confidence.
3. **Season Base Rates & Headroom**: Illustrates the mathematical ceiling on lift by season.
4. **Set B Incremental Gains**: Highlights weather items that genuinely enhance persistence.
5. **Support Tuning & Episodic Discovery**: Compares rule discovery across support thresholds.""")

add_code("""# Fig 1: Train Lift vs 2025-26 Test Lift Scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, s_code, col, s_title in [(axes[0], "A", BLUE, "Set A: Weather & Season Rules"), (axes[1], "B", ORANGE, "Set B: Persistence Rules")]:
    sub = dashboard[(dashboard["set"] == s_code) & dashboard["test_lift"].notna()]
    ax.scatter(sub["lift"], sub["test_lift"], c=col, s=45, alpha=0.6, edgecolors="none")
    max_val = max(sub["lift"].max(), sub["test_lift"].max()) * 1.05
    ax.plot([0.8, max_val], [0.8, max_val], "--", color=MUTED, lw=1, label="y = x (Perfect transfer)")
    ax.axhline(1.0, color=RED, ls=":", lw=1, label="Lift = 1.0")
    ax.axvline(1.0, color=RED, ls=":", lw=1)
    ax.set_xlabel("Training Lift (pre-2025)")
    ax.set_ylabel("Test Lift (2025-26)")
    ax.set_title(s_title)
    ax.legend(loc="lower right")
fig.suptitle("Rule Stability: Train Lift vs Frozen 2025–26 Test Lift", fontsize=13, fontweight="bold", x=0.01, ha="left")
save_fig(fig, "01_train_vs_test_lift")

# Fig 2: Confidence Comparison: Train vs Test for Key Rules
key_rules = dashboard[dashboard["set"] == "A"].sort_values("test_n", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(key_rules))[::-1]
ax.barh(y_pos + 0.15, key_rules["confidence"], 0.3, color=BLUE, label="Train Confidence (pre-2025)")
ax.barh(y_pos - 0.15, key_rules["test_conf"], 0.3, color=ORANGE, label="Test Confidence (2025-26)")
ax.set_yticks(y_pos)
ax.set_yticklabels(key_rules["rule_text"], fontsize=9)
ax.set_xlabel("Confidence P(Next Day High Risk | Rule)")
ax.set_xlim(0, 1.05)
ax.set_title("Set A Rules: Confidence Drift Between Training and 2025–26 Test")
ax.legend(loc="lower right")
save_fig(fig, "02_confidence_drift")

# Fig 3: Season Base Rates and Headroom
season_df = pd.DataFrame(season_stats)
season_df["Train Rate Float"] = season_df["Train Base Rate"].astype(float)
season_df["Test Rate Float"] = season_df["Test Base Rate"].astype(float)
season_df["Max Lift Float"] = season_df["Max Possible Lift"].astype(float)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(season_df))
ax.bar(x - 0.15, season_df["Train Rate Float"], 0.3, color=BLUE, label="Train Base Rate (pre-2025)")
ax.bar(x + 0.15, season_df["Test Rate Float"], 0.3, color=ORANGE, label="Test Base Rate (2025-26)")
ax.set_xticks(x)
ax.set_xticklabels(season_df["Season"], fontsize=9)
ax.set_ylabel("P(Next Day High Risk)")
ax.set_title("Season Base Rates: Base Rate Shifts Explain Confidence Drift")
ax.legend()
for i, r in season_df.iterrows():
    ax.text(i, max(r["Train Rate Float"], r["Test Rate Float"]) + 0.03, f"Max Lift\\n<= {r['Max Lift Float']:.2f}", ha="center", fontsize=8, color=INK2)
ax.set_ylim(0, 0.85)
save_fig(fig, "03_season_base_rates")

# Fig 4: Set B Incremental Gain over Yesterday-Only Rule
top_b_gain = dashboard[dashboard["set"] == "B"].sort_values("gain_conf_vs_yesterday", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9, 4.5))
y_pos = np.arange(len(top_b_gain))[::-1]
ax.barh(y_pos, top_b_gain["gain_conf_vs_yesterday"], color=AQUA, height=0.5)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_b_gain["rule_text"], fontsize=8.5)
ax.set_xlabel("Incremental Confidence Gain over 'Yesterday was high' (+0.869)")
ax.set_title("Set B Weather Synergy: Value-Add Beyond Autoregressive Persistence")
save_fig(fig, "04_set_b_gain_over_yesterday")

# Fig 5: Support Tuning & Diwali Window Rules
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.bar(df_tuning.index - 0.15, df_tuning["total_rules"], width=0.3, color=BLUE, label="Total High-Risk Rules")
ax1.bar(df_tuning.index + 0.15, df_tuning["diwali_rules"] * 10, width=0.3, color=ORANGE, label="Diwali Rules (×10)")
ax1.set_xticks(df_tuning.index)
ax1.set_xticklabels([f"Sup ≥ {s*100:.0f}%" for s in df_tuning["min_support"]])
ax1.set_ylabel("Rule Count")
ax1.set_title("Rule Count Sensitivity to Support Filter")
ax1.legend()

f_02 = fpgrowth(train_enc, min_support=0.02, use_colnames=True)
r_02 = association_rules(f_02, metric="confidence", min_threshold=MIN_CONFIDENCE)
diwali_top = r_02[(r_02["consequents"] == target_consequent) & r_02["antecedents"].apply(lambda s: "diwali_window" in s)].sort_values("lift", ascending=False).head(5)
diwali_labels = [make_rule_text(a) for a in diwali_top["antecedents"]]
y_pos = np.arange(len(diwali_top))[::-1]
ax2.barh(y_pos, diwali_top["confidence"], color=ORANGE, height=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(diwali_labels, fontsize=8)
ax2.set_xlabel("Confidence P(Next High Risk | Rule)")
ax2.set_xlim(0, 1.05)
ax2.set_title("Top Tuned Diwali Rules (min_support = 2%)")
for idx, val in enumerate(diwali_top["confidence"].values[::-1]):
    lift_val = diwali_top["lift"].values[::-1][idx]
    ax2.text(val + 0.02, idx, f"Conf={val:.2f}, Lift={lift_val:.2f}", va="center", fontsize=8, color=INK)

fig.suptitle("Filter Tuning: Uncovering Episodic Festival Regimes", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
save_fig(fig, "05_support_tuning_and_diwali")""")

# Save notebook
out_nb_path = Path("M4_Association_Rule_Mining.ipynb")
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Notebook successfully created at: {out_nb_path}")
