# ── M4 Association‑Rule Mining ──────────────────────────────────────────────────
# Full implementation with all checks from the feasibility review.
# Reads: m4_full_engineered.csv  (produced by the M4 Feature Engineering notebook)
# Writes: m4_arm_rules.csv, m4_arm_dashboard.csv, plots to m4_arm_outputs/
# ────────────────────────────────────────────────────────────────────────────────

import warnings, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 0.  Paths & constants ──────────────────────────────────────────────────────

DATA_DIR = Path(r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study")
OUT      = Path("m4_arm_outputs"); OUT.mkdir(exist_ok=True)
THRESH   = 120.5                                   # PM2.5 threshold for high_risk
TRAIN_END  = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
BOOT_B, BOOT_BLOCK = 2000, 7                       # block‑bootstrap parameters
MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = 0.05, 0.60, 1.20

# ── Chart style (consistent with M3) ───────────────────────────────────────────

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110,
})
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("Setup complete.")

# ── 1.  Load data and build day‑pairs ──────────────────────────────────────────

raw = pd.read_csv(DATA_DIR / "m4" / "m4_full_engineered.csv", parse_dates=["date"])
raw = raw.sort_values("date").reset_index(drop=True)
print(f"Loaded m4_full_engineered.csv: {len(raw)} rows, {raw['date'].min().date()} → {raw['date'].max().date()}")

# ── Season definition ──────────────────────────────────────────────────────────
# The `season` column in the dataset uses the India Meteorological Department (IMD)
# definition:
#   Winter       = Jan, Feb
#   Pre‑monsoon  = Mar, Apr, May
#   Monsoon      = Jun, Jul, Aug, Sep
#   Post‑monsoon = Oct, Nov, Dec
#
# M3's is_winter_nov_jan (Nov–Jan) and M4‑FE's is_winter (Nov–Feb) both differ.
# M4‑ARM uses the IMD season column throughout for consistency with the raw data.
# The column `season` is authoritative; we do NOT use is_winter or is_post_monsoon.

SEASON_MAP = {
    "winter":       "Winter (Jan–Feb)",
    "pre_monsoon":  "Pre-monsoon (Mar–May)",
    "monsoon":      "Monsoon (Jun–Sep)",
    "post_monsoon": "Post-monsoon (Oct–Dec)",
}
raw["season_label"] = raw["season"].map(SEASON_MAP)
assert raw["season_label"].notna().all(), "Unmapped season values"

# ── Day‑pair construction ──────────────────────────────────────────────────────
# Day t's weather features → day (t+1)'s high_risk label.
# Drop any pair where EITHER label is missing (never fill with 0).

weather_t = raw.copy()
weather_t["next_high_risk"] = raw["high_risk"].shift(-1)
weather_t["next_date"]      = raw["date"].shift(-1)
weather_t["next_season"]    = raw["season_label"].shift(-1)

# Drop last row (no next day), and drop where either today's or next day's label is NaN
pairs = weather_t.iloc[:-1].copy()
pairs = pairs.dropna(subset=["high_risk", "next_high_risk"])
pairs["next_high_risk"] = pairs["next_high_risk"].astype(int)
pairs["today_high_risk"] = pairs["high_risk"].astype(int)

train = pairs[pairs["date"] <= TRAIN_END].copy()
test  = pairs[pairs["date"] >= TEST_START].copy()

print(f"Day‑pairs total: {len(pairs)}  |  train (pre‑2025): {len(train)}  |  test (2025–26): {len(test)}")
print(f"Train next‑day high‑risk rate: {train['next_high_risk'].mean():.3f}")
print(f"Test  next‑day high‑risk rate: {test['next_high_risk'].mean():.3f}")

# ── Season base‑rate table ─────────────────────────────────────────────────────

print("\nSeason base rates (train, next‑day high‑risk):")
for s in ["Monsoon (Jun–Sep)", "Pre-monsoon (Mar–May)", "Post-monsoon (Oct–Dec)", "Winter (Jan–Feb)"]:
    sub = train[train["next_season"] == s]
    if len(sub):
        rate = sub["next_high_risk"].mean()
        max_lift = 1.0 / rate if rate > 0 else np.inf
        print(f"  {s:30s}  rate={rate:.3f}  max‑lift={max_lift:.2f}  n={len(sub)}")

# ── 2.  Bin weather features on TRAIN only, freeze edges ───────────────────────

WEATHER_COLS = ["temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean",
                "precip_sum", "cloud_mean", "dewpoint_mean", "wind_gust_max"]

BIN_LABELS = {
    "temp_mean":       {"bins": 3, "labels": ["Cold", "Mild", "Hot"]},
    "humidity_mean":   {"bins": 3, "labels": ["Low humidity", "Mid humidity", "High humidity"]},
    "wind_speed_mean": {"bins": 3, "labels": ["Low wind", "Mid wind", "High wind"]},
    "pressure_mean":   {"bins": 3, "labels": ["Low pressure", "Mid pressure", "High pressure"]},
    "precip_sum":      {"custom": True},   # binary: rain vs no rain
    "cloud_mean":      {"bins": 3, "labels": ["Clear sky", "Partly cloudy", "Overcast"]},
    "dewpoint_mean":   {"bins": 3, "labels": ["Low dewpoint", "Mid dewpoint", "High dewpoint"]},
    "wind_gust_max":   {"bins": 3, "labels": ["Low gust", "Mid gust", "High gust"]},
}

bin_edges = {}   # frozen on train
item_cols = []   # names of all binary item columns

for col in WEATHER_COLS:
    spec = BIN_LABELS[col]
    if spec.get("custom"):
        # Precipitation: binary (no rain / rain)
        pairs[f"item_{col}_no_rain"]  = (pairs[col] == 0).astype(int)
        pairs[f"item_{col}_rain"]     = (pairs[col] > 0).astype(int)
        item_cols += [f"item_{col}_no_rain", f"item_{col}_rain"]
        bin_edges[col] = "binary"
    else:
        # Quantile‑based tercile bins, edges fitted on TRAIN
        edges = np.quantile(train[col].dropna(), [0, 1/3, 2/3, 1])
        edges[0]  -= 1e-6   # include min
        edges[-1] += 1e-6   # include max
        bin_edges[col] = edges.copy()
        for i, lab in enumerate(spec["labels"]):
            lo, hi = edges[i], edges[i+1]
            col_name = f"item_{col}_{lab.replace(' ', '_').lower()}"
            pairs[col_name] = ((pairs[col] > lo) & (pairs[col] <= hi)).astype(int)
            item_cols.append(col_name)

# Season items (from the NEXT day's season, since the consequent is next‑day's label)
for s_raw, s_label in SEASON_MAP.items():
    col_name = f"item_season_{s_raw}"
    pairs[col_name] = (pairs["next_season"] == s_label).astype(int)
    item_cols.append(col_name)

# Persistence item: today was high‑risk (Set B only)
pairs["item_yesterday_high"] = pairs["today_high_risk"]
item_cols.append("item_yesterday_high")

# Festival/Diwali items (next day)
pairs["item_festival_window"] = pairs["is_festival_window"].astype(int)
item_cols.append("item_festival_window")

# Weekend item (next day)
pairs["item_weekend"] = pairs["is_weekend"].astype(int)
item_cols.append("item_weekend")

# Re‑split after binning
train = pairs[pairs["date"] <= TRAIN_END].copy()
test  = pairs[pairs["date"] >= TEST_START].copy()

print(f"\nTotal item columns: {len(item_cols)}")
print(f"Bin edges frozen on {len(train)} train pairs")
print("Items:", [c.replace('item_', '') for c in item_cols])

# ── 3.  Mine association rules ─────────────────────────────────────────────────

def mine_rules(df, antecedent_items, consequent_col, max_k=4, min_sup=MIN_SUPPORT,
               min_conf=MIN_CONFIDENCE, min_lift=MIN_LIFT, rule_set="A"):
    """
    Apriori‑style mining.  Enumerates itemsets of size 1..max_k, computes
    support(antecedent ∪ consequent=1), confidence and lift.
    """
    n = len(df)
    base_rate = df[consequent_col].mean()
    rules = []

    for k in range(1, max_k + 1):
        for combo in itertools.combinations(antecedent_items, k):
            mask = np.ones(n, dtype=bool)
            for item in combo:
                mask &= (df[item].values == 1)
            sup_ant = mask.sum()
            if sup_ant / n < min_sup:
                continue
            hits = (df.loc[mask, consequent_col] == 1).sum()
            support   = sup_ant / n
            confidence = hits / sup_ant if sup_ant > 0 else 0
            lift       = confidence / base_rate if base_rate > 0 else np.inf
            if confidence >= min_conf and lift >= min_lift:
                rules.append({
                    "items":      combo,
                    "set":        rule_set,
                    "k":          k,
                    "support":    support,
                    "sup_count":  sup_ant,
                    "hits":       hits,
                    "confidence": confidence,
                    "lift":       lift,
                    "base_rate":  base_rate,
                })
    return pd.DataFrame(rules)

# Set A items: weather + season + calendar (no persistence)
set_a_items = [c for c in item_cols if c != "item_yesterday_high"]
# Set B items: all items including persistence
set_b_items = item_cols.copy()

print("Mining Set A rules (weather + season + calendar → next‑day high‑risk) ...")
rules_a = mine_rules(train, set_a_items, "next_high_risk", max_k=4, rule_set="A")
print(f"  Set A: {len(rules_a)} rules")

print("Mining Set B rules (weather + season + calendar + persistence → next‑day high‑risk) ...")
rules_b = mine_rules(train, set_b_items, "next_high_risk", max_k=4, rule_set="B")
print(f"  Set B: {len(rules_b)} rules")

all_rules = pd.concat([rules_a, rules_b], ignore_index=True)

# ── 4.  De‑duplication: prune redundant rules ─────────────────────────────────
#  Drop any rule whose lift is NOT higher than that of ALL its simpler sub‑rules.

def prune_redundant(df_rules):
    """Remove rules that don't improve lift over any sub‑rule."""
    # Build a lookup: frozenset(items) → lift
    lift_map = {}
    for _, r in df_rules.iterrows():
        lift_map[frozenset(r["items"])] = r["lift"]

    keep = []
    for _, r in df_rules.iterrows():
        items = r["items"]
        if len(items) == 1:
            keep.append(True)
            continue
        # Check all sub‑rules of size k-1
        dominated = False
        for sub in itertools.combinations(items, len(items) - 1):
            sub_key = frozenset(sub)
            if sub_key in lift_map and lift_map[sub_key] >= r["lift"]:
                dominated = True
                break
        keep.append(not dominated)
    return df_rules[keep].reset_index(drop=True)

before = len(all_rules)
all_rules = prune_redundant(all_rules)
print(f"\nDe‑duplication: {before} → {len(all_rules)} rules (removed {before - len(all_rules)} redundant)")

# ── 5.  Significance testing ──────────────────────────────────────────────────
#  Chi‑square (or Fisher exact) test for each rule, with BH false‑discovery correction.
#  Also block‑bootstrap confidence intervals on lift (block size = 7 days).

def rule_pvalue(row, df):
    """One‑sided chi‑square (or Fisher) p‑value for the rule."""
    mask = np.ones(len(df), dtype=bool)
    for item in row["items"]:
        mask &= (df[item].values == 1)
    # 2×2 contingency table
    a = int((df.loc[mask, "next_high_risk"] == 1).sum())   # rule AND high_risk
    b = int(mask.sum() - a)                                 # rule AND NOT high_risk
    c = int((df.loc[~mask, "next_high_risk"] == 1).sum())  # NOT rule AND high_risk
    d = int((~mask).sum() - c)                              # NOT rule AND NOT high_risk

    table = np.array([[a, b], [c, d]])
    if table.min() < 5:
        _, p = stats.fisher_exact(table, alternative="greater")
    else:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
        # One‑sided: keep p if lift > 1 (the rule enriches high‑risk)
        if a / max(mask.sum(), 1) < df["next_high_risk"].mean():
            p = 1.0  # wrong direction
    return p

print("Computing p‑values (chi‑square / Fisher) ...")
pvals = all_rules.apply(lambda r: rule_pvalue(r, train), axis=1)
all_rules["p_value_raw"] = pvals

# BH correction
reject, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
all_rules["p_value_bh"] = pvals_adj
all_rules["significant_bh05"] = reject

print(f"  Significant at BH‑adjusted α=0.05: {reject.sum()} / {len(all_rules)}")

# ── Block‑bootstrap confidence interval on lift ───────────────────────────────

def block_bootstrap_lift(row, df, B=BOOT_B, block=BOOT_BLOCK):
    """Moving‑block bootstrap: resample blocks of `block` consecutive days."""
    n = len(df)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    lifts = []
    for _ in range(B):
        chosen = np.random.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in chosen])[:n]
        boot = df.iloc[idx]
        mask = np.ones(len(boot), dtype=bool)
        for item in row["items"]:
            mask &= (boot[item].values == 1)
        base = boot["next_high_risk"].mean()
        if mask.sum() == 0 or base == 0:
            continue
        conf = boot.loc[mask, "next_high_risk"].mean()
        lifts.append(conf / base)
    if len(lifts) < 100:
        return np.nan, np.nan
    return np.percentile(lifts, 2.5), np.percentile(lifts, 97.5)

print(f"Block‑bootstrap ({BOOT_B} resamples, block={BOOT_BLOCK}) for lift CIs ...")
boot_results = all_rules.apply(lambda r: block_bootstrap_lift(r, train), axis=1)
all_rules["lift_ci_lo"] = boot_results.apply(lambda x: x[0])
all_rules["lift_ci_hi"] = boot_results.apply(lambda x: x[1])
print("  Done.")

# ── 6.  Evaluate on 2025–26 test set ──────────────────────────────────────────

test_base_rate = test["next_high_risk"].mean()

def eval_on_test(row, df_test, test_base):
    mask = np.ones(len(df_test), dtype=bool)
    for item in row["items"]:
        mask &= (df_test[item].values == 1)
    n_match = mask.sum()
    if n_match == 0:
        return pd.Series({"test_conf": np.nan, "test_lift": np.nan, "test_n": 0})
    conf = df_test.loc[mask, "next_high_risk"].mean()
    lift = conf / test_base if test_base > 0 else np.inf
    return pd.Series({"test_conf": conf, "test_lift": lift, "test_n": n_match})

print("Evaluating rules on 2025–26 test set ...")
test_eval = all_rules.apply(lambda r: eval_on_test(r, test, test_base_rate), axis=1)
all_rules = pd.concat([all_rules, test_eval], axis=1)
all_rules["test_n"] = all_rules["test_n"].astype(int)

# ── 7.  Set B: gain over yesterday‑only rule ──────────────────────────────────
# The "yesterday was high" baseline for Set B

yesterday_mask = (train["item_yesterday_high"] == 1)
yesterday_conf = train.loc[yesterday_mask, "next_high_risk"].mean()
yesterday_lift = yesterday_conf / train["next_high_risk"].mean()

print(f"\nYesterday‑only baseline: confidence={yesterday_conf:.3f}, lift={yesterday_lift:.2f}")

all_rules["gain_conf_vs_yesterday"] = np.nan
all_rules["gain_lift_vs_yesterday"] = np.nan
set_b_mask = all_rules["set"] == "B"
all_rules.loc[set_b_mask, "gain_conf_vs_yesterday"] = all_rules.loc[set_b_mask, "confidence"] - yesterday_conf
all_rules.loc[set_b_mask, "gain_lift_vs_yesterday"] = all_rules.loc[set_b_mask, "lift"] - yesterday_lift

# ── 8.  Within‑season lift ─────────────────────────────────────────────────────

def within_season_lift(row, df):
    """Compute lift relative to the base rate of the rule's dominant season."""
    mask = np.ones(len(df), dtype=bool)
    for item in row["items"]:
        mask &= (df[item].values == 1)

    # Identify which season items are in the antecedent
    season_items = [it for it in row["items"] if it.startswith("item_season_")]
    if not season_items:
        return np.nan  # no season in antecedent
    # Use the first season item to define the season subset
    season_mask = (df[season_items[0]] == 1)
    season_base = df.loc[season_mask, "next_high_risk"].mean()
    if season_base == 0:
        return np.inf
    conf = df.loc[mask, "next_high_risk"].mean() if mask.sum() > 0 else 0
    return conf / season_base

all_rules["lift_within_season"] = all_rules.apply(lambda r: within_season_lift(r, train), axis=1)

# ── 9.  Build plain‑language rule descriptions ────────────────────────────────

ITEM_LABELS = {}
for col in WEATHER_COLS:
    spec = BIN_LABELS[col]
    if spec.get("custom"):
        ITEM_LABELS[f"item_{col}_no_rain"]  = "No rain"
        ITEM_LABELS[f"item_{col}_rain"]     = "Rain"
    else:
        for lab in spec["labels"]:
            key = f"item_{col}_{lab.replace(' ', '_').lower()}"
            ITEM_LABELS[key] = lab

for s_raw, s_label in SEASON_MAP.items():
    ITEM_LABELS[f"item_season_{s_raw}"] = s_label

ITEM_LABELS["item_yesterday_high"]   = "Yesterday was high-risk"
ITEM_LABELS["item_festival_window"]  = "Festival window"
ITEM_LABELS["item_weekend"]          = "Weekend"

def rule_text(items):
    parts = [ITEM_LABELS.get(it, it.replace("item_", "")) for it in items]
    return " + ".join(parts)

all_rules["rule_text"] = all_rules["items"].apply(rule_text)

# ── 10.  Sort and export ──────────────────────────────────────────────────────

all_rules = all_rules.sort_values(["set", "lift"], ascending=[True, False]).reset_index(drop=True)

# Dashboard columns
dashboard_cols = [
    "rule_text", "set", "k",
    "support", "confidence", "lift",
    "lift_ci_lo", "lift_ci_hi",
    "lift_within_season",
    "test_conf", "test_lift", "test_n",
    "p_value_bh", "significant_bh05",
    "gain_conf_vs_yesterday", "gain_lift_vs_yesterday",
]
dashboard = all_rules[dashboard_cols].copy()

# Round for readability
for c in ["support", "confidence", "lift", "lift_ci_lo", "lift_ci_hi",
           "lift_within_season", "test_conf", "test_lift", "p_value_bh",
           "gain_conf_vs_yesterday", "gain_lift_vs_yesterday"]:
    if c in dashboard.columns:
        dashboard[c] = dashboard[c].round(4)

dashboard.to_csv(OUT / "m4_arm_dashboard.csv", index=False)
all_rules.to_csv(OUT / "m4_arm_rules_full.csv", index=False)
print(f"\nDashboard saved: {OUT / 'm4_arm_dashboard.csv'}")
print(f"Full rules saved: {OUT / 'm4_arm_rules_full.csv'}")

# ── 11.  Summary tables ───────────────────────────────────────────────────────

print("\n" + "="*90)
print("TOP 15 SET A RULES (weather + season + calendar → next‑day high‑risk)")
print("="*90)
top_a = dashboard[dashboard["set"] == "A"].head(15)
print(top_a.to_string(index=False))

print("\n" + "="*90)
print("TOP 15 SET B RULES (includes persistence; gain over yesterday‑only baseline shown)")
print(f"  Yesterday‑only baseline: confidence={yesterday_conf:.3f}, lift={yesterday_lift:.2f}")
print("="*90)
top_b = dashboard[dashboard["set"] == "B"].head(15)
print(top_b.to_string(index=False))

# ── Significance summary ──────────────────────────────────────────────────────

n_sig = all_rules["significant_bh05"].sum()
n_tot = len(all_rules)
print(f"\nSignificance: {n_sig}/{n_tot} rules survive BH‑corrected α=0.05")
print(f"  Set A significant: {all_rules.loc[all_rules['set']=='A', 'significant_bh05'].sum()}/{(all_rules['set']=='A').sum()}")
print(f"  Set B significant: {all_rules.loc[all_rules['set']=='B', 'significant_bh05'].sum()}/{(all_rules['set']=='B').sum()}")

# ── 12.  Visualisations ───────────────────────────────────────────────────────

# 12a. Train lift vs Test lift scatter (Set A)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, s, color, title in [
    (axes[0], "A", BLUE, "Set A: Weather/season rules"),
    (axes[1], "B", ORANGE, "Set B: Persistence rules"),
]:
    sub = all_rules[all_rules["set"] == s].copy()
    sub = sub.dropna(subset=["test_lift"])
    ax.scatter(sub["lift"], sub["test_lift"], c=color, alpha=0.5, edgecolors="white", s=50)
    lims = [0.8, max(sub["lift"].max(), sub["test_lift"].max()) * 1.1]
    ax.plot(lims, lims, "--", color=MUTED, lw=1, label="y = x")
    ax.axhline(1, color=RED, ls=":", lw=0.8, alpha=0.6)
    ax.axvline(1, color=RED, ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Train lift"); ax.set_ylabel("Test lift (2025–26)")
    ax.set_title(title)
    ax.legend(loc="lower right")
fig.suptitle("Train lift vs 2025–26 test lift", fontsize=13, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(); save(fig, "01_train_vs_test_lift")

# 12b. Top Set A rules — confidence bars (train vs test)
top_rules = all_rules[(all_rules["set"] == "A") & (all_rules["test_n"] >= 5)].head(12)
if len(top_rules):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_rules))[::-1]
    ax.barh(y_pos + 0.15, top_rules["confidence"], height=0.3, color=BLUE, label="Train", alpha=0.85)
    ax.barh(y_pos - 0.15, top_rules["test_conf"], height=0.3, color=ORANGE, label="Test 2025–26", alpha=0.85)
    ax.set_yticks(y_pos)
    labels = [t[:55] + "…" if len(t) > 55 else t for t in top_rules["rule_text"]]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Confidence (P[next‑day high‑risk | rule])")
    ax.set_title("Top Set A rules: train vs test confidence")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)
    fig.tight_layout(); save(fig, "02_set_a_confidence_bars")

# 12c. Season base‑rate chart
season_data = []
for s in ["Monsoon (Jun–Sep)", "Pre-monsoon (Mar–May)", "Post-monsoon (Oct–Dec)", "Winter (Jan–Feb)"]:
    sub_tr = train[train["next_season"] == s]
    sub_te = test[test["next_season"] == s]
    season_data.append({
        "season": s.split(" (")[0],
        "train_rate": sub_tr["next_high_risk"].mean() if len(sub_tr) else 0,
        "test_rate":  sub_te["next_high_risk"].mean() if len(sub_te) else 0,
    })
sd = pd.DataFrame(season_data)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(sd))
ax.bar(x - 0.15, sd["train_rate"], 0.3, color=BLUE, label="Train (pre‑2025)")
ax.bar(x + 0.15, sd["test_rate"],  0.3, color=ORANGE, label="Test (2025–26)")
ax.set_xticks(x); ax.set_xticklabels(sd["season"], fontsize=9)
ax.set_ylabel("P(next‑day high‑risk)")
ax.set_title("High‑risk base rate by season")
ax.legend(); fig.tight_layout(); save(fig, "03_season_base_rates")

# 12d. Bootstrap CI on top rules
top_ci = all_rules[(all_rules["set"] == "A") & (all_rules["lift_ci_lo"].notna())].head(10)
if len(top_ci):
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(top_ci))[::-1]
    ax.barh(y_pos, top_ci["lift"], height=0.5, color=BLUE, alpha=0.7)
    ax.errorbar(top_ci["lift"].values, y_pos,
                xerr=[top_ci["lift"].values - top_ci["lift_ci_lo"].values,
                      top_ci["lift_ci_hi"].values - top_ci["lift"].values],
                fmt="none", color=INK2, capsize=3)
    ax.axvline(1, color=RED, ls=":", lw=1)
    ax.set_yticks(y_pos)
    labels = [t[:55] + "…" if len(t) > 55 else t for t in top_ci["rule_text"]]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Lift (with 95% block‑bootstrap CI)")
    ax.set_title("Top Set A rules: lift with bootstrap uncertainty")
    fig.tight_layout(); save(fig, "04_lift_bootstrap_ci")

# ── 13.  Stability analysis ───────────────────────────────────────────────────
# Flag rules where test lift drops below 1.0 (rule not useful out‑of‑sample)

all_rules["stable"] = (all_rules["test_lift"] >= 1.0) & (all_rules["test_n"] >= 5)

stable_a = all_rules[(all_rules["set"] == "A") & all_rules["stable"]].copy()
unstable_a = all_rules[(all_rules["set"] == "A") & ~all_rules["stable"] & (all_rules["test_n"] >= 5)].copy()

print(f"\nStability analysis (Set A):")
print(f"  Stable rules (test lift ≥ 1.0, n ≥ 5): {len(stable_a)}")
print(f"  Unstable rules (test lift < 1.0, n ≥ 5): {len(unstable_a)}")

if len(stable_a):
    print("\n  Top stable Set A rules:")
    for _, r in stable_a.head(8).iterrows():
        print(f"    {r['rule_text']:55s}  train_conf={r['confidence']:.2f}  test_conf={r['test_conf']:.2f}"
              f"  train_lift={r['lift']:.2f}  test_lift={r['test_lift']:.2f}")

if len(unstable_a):
    print("\n  Examples of unstable Set A rules:")
    for _, r in unstable_a.head(5).iterrows():
        print(f"    {r['rule_text']:55s}  train_conf={r['confidence']:.2f}  test_conf={r['test_conf']:.2f}"
              f"  train_lift={r['lift']:.2f}  test_lift={r['test_lift']:.2f}")

# ── Final summary ─────────────────────────────────────────────────────────────

print("\n" + "="*90)
print("M4 ASSOCIATION RULE MINING — SUMMARY")
print("="*90)
print(f"""
Data:
  Source file:       m4_full_engineered.csv
  Day‑pairs:         {len(pairs)} total ({len(train)} train, {len(test)} test)
  Target:            next‑day high_risk (PM2.5 ≥ {THRESH})
  Pair logic:        day t weather → day t+1 label; both labels must exist

Season definition:   IMD standard (season column)
  Winter = Jan–Feb,  Pre‑monsoon = Mar–May,  Monsoon = Jun–Sep,  Post‑monsoon = Oct–Dec

Binning:             Tercile quantile bins fitted on pre‑2025 train data, frozen for test

Thresholds:          support ≥ {MIN_SUPPORT}, confidence ≥ {MIN_CONFIDENCE}, lift ≥ {MIN_LIFT}

Rules mined:
  Set A (weather + season + calendar): {(all_rules['set']=='A').sum()} rules after pruning
  Set B (+ persistence):               {(all_rules['set']=='B').sum()} rules after pruning

Significance:        Chi‑square/Fisher with Benjamini–Hochberg FDR correction
                     {all_rules['significant_bh05'].sum()}/{len(all_rules)} rules significant at α=0.05
                     Block‑bootstrap CIs (B={BOOT_B}, block={BOOT_BLOCK} days)

De‑duplication:      Rules pruned if lift ≤ any sub‑rule's lift

Set B baseline:      "Yesterday was high" alone → confidence={yesterday_conf:.3f}, lift={yesterday_lift:.2f}
                     Set B gain columns show improvement over this baseline

Interpretation:      Rules are DESCRIPTIVE — they identify conditions associated with
                     high‑pollution days and feed a ranked dashboard list.
                     They are NOT a forecasting model (M3 shows alerts are mostly season).
""")

print("Outputs:")
print(f"  {OUT / 'm4_arm_dashboard.csv'}")
print(f"  {OUT / 'm4_arm_rules_full.csv'}")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f}")
print("\nDone.")
