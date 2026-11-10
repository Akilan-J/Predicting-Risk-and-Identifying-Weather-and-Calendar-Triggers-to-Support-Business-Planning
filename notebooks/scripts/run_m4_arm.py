"""
M4: Association Rule Mining for High-Risk PM2.5 Days (mlxtend FP-Growth / Apriori)
==================================================================================
Comprehensive implementation covering all requirements:
1. Preparation:
   - Binned features: wind, humidity, temperature, pressure (frozen terciles fitted pre-2025)
   - Rain: binary (no rain vs rain)
   - Season: IMD standard (Winter, Pre-monsoon, Monsoon, Post-monsoon)
   - Diwali window: festival window indicator (is_festival_window)
   - Weekend: is_weekend indicator
   - Yesterday status (Set B): day t high-risk label
   - Correct calendar day-pairing: Day t weather/calendar -> Day t+1 high_risk
   - Clean pairs: drop any pair where either label is missing (never zero-fill)
2. Mining with mlxtend FP-Growth:
   - Starting filters: support >= 0.05, confidence >= 0.60, lift >= 1.20
   - Filter tuning: support tuning (0.05 down to 0.02 for episodic Diwali window)
   - De-duplication: Redundant rule pruning (rule dropped if lift <= any sub-rule)
3. Statistical Evaluation & Inference:
   - One-sided Chi-square / Fisher exact tests with Benjamini-Hochberg FDR correction
   - Moving-block bootstrap 95% confidence intervals on lift (B=1000, block=7 days)
4. Out-of-Sample Verification:
   - Evaluated on frozen 2025-26 test set (N=633 day-pairs)
   - Base-rate comparison (overall, seasonal headroom, and within-season lift)
   - Set B incremental gain over "yesterday was high" baseline (Conf=0.869, Lift=2.95)
5. Ranked Dashboard Table & Diagnostic Figures:
   - CSV outputs: m4_arm_dashboard.csv, m4_arm_rules_full.csv
   - Visualisations: lift transfer, confidence drift, seasonal base rates, Set B gain, support tuning
"""

import os
import sys
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from mlxtend.frequent_patterns import fpgrowth, association_rules

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

np.random.seed(42)

# ---- Settings & Directories ----
DATA_CANDIDATES = [
    Path(r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study\m4\m4_full_engineered.csv"),
    Path(".") / "m4_full_engineered.csv",
    Path("../group study/m4/m4_full_engineered.csv")
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
assert DATA_PATH is not None, "m4_full_engineered.csv not found!"

OUT_DIR = Path("m4_arm_outputs")
OUT_DIR.mkdir(exist_ok=True)

THRESH = 120.5                      # PM2.5 threshold for high_risk (AQI > 300)
SPLIT_DATE = pd.Timestamp("2025-01-01")
MIN_SUPPORT = 0.05                  # Starting filter: 5% of days
MIN_CONFIDENCE = 0.60               # Starting filter: 60% confidence
MIN_LIFT = 1.20                     # Starting filter: 1.2x base rate
BOOT_B = 1000
BOOT_BLOCK = 7                       # 7-day blocks for temporal autocorrelation

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
    "legend.frameon": False, "figure.dpi": 130,
})

def save_fig(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

# ── 1. Load Data & Build Day-Pairs ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# Build day-pairs: Day t weather & calendar -> Day t+1 high_risk
pairs = pd.DataFrame()
pairs["date"] = df["date"]
pairs["target_today"] = df["high_risk"]
pairs["target_next"] = df["high_risk"].shift(-1)
pairs["season_today"] = df["season"]
pairs["season_next"] = df["season"].shift(-1)
pairs["is_weekend"] = df["is_weekend"]
pairs["is_festival_window"] = df["is_festival_window"]

# Weather features of day t
WX_COLS = ["temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean", "precip_sum"]
for col in WX_COLS:
    pairs[col] = df[col]

# Clean pairs: drop any pair where either today's or next-day's label is missing
clean_pairs = pairs.dropna(subset=["target_today", "target_next"]).copy()
clean_pairs["target_today"] = clean_pairs["target_today"].astype(int)
clean_pairs["target_next"] = clean_pairs["target_next"].astype(int)

train = clean_pairs[clean_pairs["date"] < SPLIT_DATE].copy().reset_index(drop=True)
test  = clean_pairs[clean_pairs["date"] >= SPLIT_DATE].copy().reset_index(drop=True)

train_base_rate = train["target_next"].mean()
test_base_rate  = test["target_next"].mean()

print(f"Total clean day-pairs: {len(clean_pairs)}")
print(f"Train day-pairs (pre-2025): {len(train)} | Base rate: {train_base_rate:.4f} ({train_base_rate*100:.1f}%)")
print(f"Test day-pairs (2025-26)  : {len(test)}  | Base rate: {test_base_rate:.4f} ({test_base_rate*100:.1f}%)")

# Seasonal Base Rates
SEASON_DISPLAY = {
    "winter": "Winter (Jan-Feb)",
    "pre_monsoon": "Pre-monsoon (Mar-May)",
    "monsoon": "Monsoon (Jun-Sep)",
    "post_monsoon": "Post-monsoon (Oct-Dec)"
}
season_summary = []
for s_code, s_name in SEASON_DISPLAY.items():
    tr_sub = train[train["season_next"] == s_code]
    te_sub = test[test["season_next"] == s_code]
    tr_rate = tr_sub["target_next"].mean() if len(tr_sub) > 0 else 0
    te_rate = te_sub["target_next"].mean() if len(te_sub) > 0 else 0
    max_lift = (1.0 / tr_rate) if tr_rate > 0 else np.inf
    season_summary.append({
        "season_code": s_code,
        "season_name": s_name,
        "train_n": len(tr_sub),
        "train_rate": tr_rate,
        "max_possible_lift": max_lift,
        "test_n": len(te_sub),
        "test_rate": te_rate
    })
df_seasons = pd.DataFrame(season_summary)
print("\nSeason Base Rates:")
print(df_seasons.to_string(index=False))

# ── 2. Bin Weather & Categorical Features on Train Only ────────────────────────
bin_edges = {}
for col in ["temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean"]:
    q = np.quantile(train[col], [0, 1/3, 2/3, 1])
    bin_edges[col] = q

print("\nFrozen Tercile Bin Edges (fitted on pre-2025 train):")
for col, q in bin_edges.items():
    print(f"  {col:<16}: [Low: <= {q[1]:.2f}, Mid: ({q[1]:.2f}, {q[2]:.2f}], High: > {q[2]:.2f}]")

def encode_items(data_df):
    cols = {}
    # 1. Wind speed
    q = bin_edges["wind_speed_mean"]
    cols["low_wind"] = data_df["wind_speed_mean"] <= q[1]
    cols["mid_wind"] = (data_df["wind_speed_mean"] > q[1]) & (data_df["wind_speed_mean"] <= q[2])
    cols["high_wind"] = data_df["wind_speed_mean"] > q[2]

    # 2. Humidity
    q = bin_edges["humidity_mean"]
    cols["low_humidity"] = data_df["humidity_mean"] <= q[1]
    cols["mid_humidity"] = (data_df["humidity_mean"] > q[1]) & (data_df["humidity_mean"] <= q[2])
    cols["high_humidity"] = data_df["humidity_mean"] > q[2]

    # 3. Temperature
    q = bin_edges["temp_mean"]
    cols["cold"] = data_df["temp_mean"] <= q[1]
    cols["mild_temp"] = (data_df["temp_mean"] > q[1]) & (data_df["temp_mean"] <= q[2])
    cols["hot"] = data_df["temp_mean"] > q[2]

    # 4. Pressure
    q = bin_edges["pressure_mean"]
    cols["low_pressure"] = data_df["pressure_mean"] <= q[1]
    cols["mid_pressure"] = (data_df["pressure_mean"] > q[1]) & (data_df["pressure_mean"] <= q[2])
    cols["high_pressure"] = data_df["pressure_mean"] > q[2]

    # 5. Rain
    cols["no_rain"] = data_df["precip_sum"] == 0
    cols["rain"] = data_df["precip_sum"] > 0

    # 6. Season (Day t)
    for s_code in ["winter", "post_monsoon", "pre_monsoon", "monsoon"]:
        cols[f"season_{s_code}"] = data_df["season_today"] == s_code

    # 7. Diwali / Festival Window
    cols["diwali_window"] = data_df["is_festival_window"] == 1

    # 8. Weekend
    cols["weekend"] = data_df["is_weekend"] == 1

    # 9. Persistence (Day t status for Set B)
    cols["yesterday_high"] = data_df["target_today"] == 1

    # Target Consequent (Day t+1 high_risk)
    cols["high_risk_next"] = data_df["target_next"] == 1

    return pd.DataFrame(cols, index=data_df.index)

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
    "diwali_window": "Diwali window",
    "weekend": "Weekend",
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
    "diwali_window": "diwali",
    "weekend": "weekend",
    "yesterday_high": "persist"
}

# ── 3. Mining with mlxtend FP-Growth ───────────────────────────────────────────
print("\n" + "="*80)
print(f"MINING FREQUENT ITEMSETS WITH MLXTEND FP-GROWTH (min_support={MIN_SUPPORT})")
print("="*80)

frequent_itemsets = fpgrowth(train_enc, min_support=MIN_SUPPORT, use_colnames=True)
print(f"Frequent itemsets mined: {len(frequent_itemsets)}")

raw_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
print(f"Association rules with confidence >= {MIN_CONFIDENCE}: {len(raw_rules)}")

# Filter to target consequent: Day t+1 high risk
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

# Split into Set A (weather + season + calendar) and Set B (includes persistence)
filtered_rules["has_yesterday"] = filtered_rules["antecedents"].apply(lambda s: "yesterday_high" in s)
filtered_rules["set"] = np.where(filtered_rules["has_yesterday"], "B", "A")
filtered_rules["k"] = filtered_rules["antecedents"].apply(len)

print(f"Filtered rules predicting next_high_risk: Total={len(filtered_rules)} | Set A={sum(filtered_rules['set']=='A')} | Set B={sum(filtered_rules['set']=='B')}")

# ── 4. De-duplication: Redundant Rule Pruning ──────────────────────────────────
# A rule is redundant if its lift does NOT strictly exceed all of its simpler sub-rules
def prune_redundant(df_rules):
    # Lookup: frozenset(antecedent) -> lift
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
print(f"  Final Set B rules: {sum(rules_pruned['set']=='B')}")

# ── 5. Significance Testing: Chi-square / Fisher Exact with Benjamini-Hochberg FDR
def compute_pvalue(antecedent, items_df):
    mask = np.ones(len(items_df), dtype=bool)
    for it in antecedent:
        mask &= (items_df[it].values == 1)
    
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
print(f"Significant rules at BH FDR alpha=0.05: {reject.sum()} / {len(rules_pruned)}")

# ── 6. Moving-Block Bootstrap 95% Confidence Intervals for Lift (block=7 days) ──
print(f"Precomputing block-bootstrap index matrix (B={BOOT_B}, block={BOOT_BLOCK} days) ...")
n_train = len(train_enc)
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
print("Bootstrap CIs complete.")

# ── 7. Evaluate on 2025-26 Frozen Out-of-Sample Test Set ───────────────────────
def evaluate_test(antecedent, test_items_df):
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

# ── 8. Within-Season Lift ──────────────────────────────────────────────────────
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

# ── 9. Set B: Incremental Gain over Yesterday-Only Baseline ───────────────────
yesterday_mask = train_enc["yesterday_high"] == 1
yesterday_conf = train_enc.loc[yesterday_mask, "high_risk_next"].mean()
yesterday_lift = yesterday_conf / train_base_rate
print(f"\nYesterday-only baseline: Confidence = {yesterday_conf:.4f}, Lift = {yesterday_lift:.4f}")

rules_pruned["gain_conf_vs_yesterday"] = np.nan
rules_pruned["gain_lift_vs_yesterday"] = np.nan

set_b_mask = rules_pruned["set"] == "B"
rules_pruned.loc[set_b_mask, "gain_conf_vs_yesterday"] = (
    rules_pruned.loc[set_b_mask, "confidence"] - yesterday_conf
)
rules_pruned.loc[set_b_mask, "gain_lift_vs_yesterday"] = (
    rules_pruned.loc[set_b_mask, "lift"] - yesterday_lift
)

# ── 10. Plain Language Descriptions & Stability Status ────────────────────────
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

# Sort rules
rules_pruned = rules_pruned.sort_values(["set", "lift"], ascending=[True, False]).reset_index(drop=True)

# ── 11. Create and Export Dashboard Tables ─────────────────────────────────────
dashboard_cols = [
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

round_cols = ["support", "confidence", "lift", "lift_ci_lo", "lift_ci_hi",
              "lift_within_season", "test_conf", "test_lift", "p_value_bh",
              "gain_conf_vs_yesterday", "gain_lift_vs_yesterday"]
for c in round_cols:
    dashboard[c] = dashboard[c].round(4)

dashboard.to_csv(OUT_DIR / "m4_arm_dashboard.csv", index=False)
rules_pruned.to_csv(OUT_DIR / "m4_arm_rules_full.csv", index=False)

print(f"\nSaved dashboard table -> {OUT_DIR / 'm4_arm_dashboard.csv'} ({len(dashboard)} rules)")
print(f"Saved full rules table -> {OUT_DIR / 'm4_arm_rules_full.csv'}")

# Print Top 10 Set A
print("\n" + "="*85)
print("TOP 10 SET A RULES (Weather + Season + Calendar -> Next-Day High Risk)")
print("="*85)
for _, r in dashboard[dashboard["set"] == "A"].head(10).iterrows():
    print(f"Rule: {r['rule_text']}")
    print(f"  Train: Sup={r['support']:.3f}, Conf={r['confidence']:.3f}, Lift={r['lift']:.2f} [{r['lift_ci_lo']:.2f}, {r['lift_ci_hi']:.2f}], BH-p={r['p_value_bh']:.3e}")
    print(f"  Test : Conf={r['test_conf']:.3f}, Lift={r['test_lift']:.2f}, N={r['test_n']} ({r['status']})")
    if not np.isnan(r['lift_within_season']):
        print(f"  Season Lift: {r['lift_within_season']:.2f}")
    print()

# Print Top 10 Set B
print("="*85)
print("TOP 10 SET B RULES (Weather + Season + Persistence -> Next-Day High Risk)")
print(f"  Yesterday-only baseline: Confidence = {yesterday_conf:.3f}, Lift = {yesterday_lift:.2f}")
print("="*85)
for _, r in dashboard[dashboard["set"] == "B"].head(10).iterrows():
    print(f"Rule: {r['rule_text']}")
    print(f"  Train: Sup={r['support']:.3f}, Conf={r['confidence']:.3f}, Lift={r['lift']:.2f} [{r['lift_ci_lo']:.2f}, {r['lift_ci_hi']:.2f}]")
    print(f"  Gain vs Yesterday: ΔConf={r['gain_conf_vs_yesterday']:+.3f}, ΔLift={r['gain_lift_vs_yesterday']:+.2f}")
    print(f"  Test : Conf={r['test_conf']:.3f}, Lift={r['test_lift']:.2f}, N={r['test_n']} ({r['status']})")
    print()

# ── 12. Support Tuning Analysis (Episodic Diwali Window Discovery) ─────────────
print("="*85)
print("SUPPORT TUNING ANALYSIS: EPISODIC FESTIVAL / DIWALI RULES")
print("="*85)
tuned_supports = [0.05, 0.03, 0.02]
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
print(df_tuning.to_string(index=False))

# ── 13. Generate Publication-Quality Visualisations ────────────────────────────
print("\nGenerating diagnostic figures...")

# Fig 1: Train Lift vs 2025-26 Test Lift Scatter
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
fig.tight_layout()
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
fig.tight_layout()
save_fig(fig, "02_confidence_drift")

# Fig 3: Season Base Rates and Headroom
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(df_seasons))
ax.bar(x - 0.15, df_seasons["train_rate"], 0.3, color=BLUE, label="Train Base Rate (pre-2025)")
ax.bar(x + 0.15, df_seasons["test_rate"], 0.3, color=ORANGE, label="Test Base Rate (2025-26)")
ax.set_xticks(x)
ax.set_xticklabels(df_seasons["season_name"], fontsize=9)
ax.set_ylabel("P(Next Day High Risk)")
ax.set_title("Season Base Rates: Base Rate Shifts Explain Confidence Drift")
ax.legend()
for i, r in df_seasons.iterrows():
    ax.text(i, max(r["train_rate"], r["test_rate"]) + 0.03, f"Max Lift\n≤ {r['max_possible_lift']:.2f}", ha="center", fontsize=8, color=INK2)
ax.set_ylim(0, 0.85)
fig.tight_layout()
save_fig(fig, "03_season_base_rates")

# Fig 4: Set B Incremental Gain over Yesterday-Only Rule
fig, ax = plt.subplots(figsize=(9, 4.5))
top_b_gain = dashboard[dashboard["set"] == "B"].sort_values("gain_conf_vs_yesterday", ascending=False).head(10)
y_pos = np.arange(len(top_b_gain))[::-1]
ax.barh(y_pos, top_b_gain["gain_conf_vs_yesterday"], color=AQUA, height=0.5)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_b_gain["rule_text"], fontsize=8.5)
ax.set_xlabel(f"Incremental Confidence Gain over 'Yesterday was high' (+{yesterday_conf:.3f})")
ax.set_title("Set B Weather Synergy: Value-Add Beyond Autoregressive Persistence")
fig.tight_layout()
save_fig(fig, "04_set_b_gain_over_yesterday")

# Fig 5: Support Tuning & Diwali Window Rules
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
# Bar chart of rule discovery vs support threshold
ax1.bar(df_tuning.index - 0.15, df_tuning["total_rules"], width=0.3, color=BLUE, label="Total High-Risk Rules")
ax1.bar(df_tuning.index + 0.15, df_tuning["diwali_rules"] * 10, width=0.3, color=ORANGE, label="Diwali Rules (×10)")
ax1.set_xticks(df_tuning.index)
ax1.set_xticklabels([f"Sup ≥ {s*100:.0f}%" for s in df_tuning["min_support"]])
ax1.set_ylabel("Rule Count")
ax1.set_title("Rule Count Sensitivity to Support Filter")
ax1.legend()

# Extract top tuned Diwali rules
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
fig.tight_layout()
save_fig(fig, "05_support_tuning_and_diwali")

print("All diagnostic figures successfully created in m4_arm_outputs/.")
