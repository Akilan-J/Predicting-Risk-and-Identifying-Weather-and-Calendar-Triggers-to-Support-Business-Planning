import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from pathlib import Path

raw = pd.read_csv(Path(r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study\m4\m4_full_engineered.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
raw["next_high_risk"] = raw["high_risk"].shift(-1)
pairs = raw.iloc[:-1].dropna(subset=["high_risk", "next_high_risk"]).copy()
pairs["next_high_risk"] = pairs["next_high_risk"].astype(int)
pairs["today_high_risk"] = pairs["high_risk"].astype(int)

train = pairs[pairs["date"] < "2025-01-01"].copy()
test  = pairs[pairs["date"] >= "2025-01-01"].copy()

bin_edges = {}
for col in ["temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean"]:
    bin_edges[col] = np.quantile(train[col], [0, 1/3, 2/3, 1])

def encode_df(df_sub):
    cols = {}
    q = bin_edges["wind_speed_mean"]
    cols["low_wind"] = df_sub["wind_speed_mean"] <= q[1]
    cols["mid_wind"] = (df_sub["wind_speed_mean"] > q[1]) & (df_sub["wind_speed_mean"] <= q[2])
    cols["high_wind"] = df_sub["wind_speed_mean"] > q[2]
    q = bin_edges["humidity_mean"]
    cols["low_humidity"] = df_sub["humidity_mean"] <= q[1]
    cols["mid_humidity"] = (df_sub["humidity_mean"] > q[1]) & (df_sub["humidity_mean"] <= q[2])
    cols["high_humidity"] = df_sub["humidity_mean"] > q[2]
    q = bin_edges["temp_mean"]
    cols["cold"] = df_sub["temp_mean"] <= q[1]
    cols["mild_temp"] = (df_sub["temp_mean"] > q[1]) & (df_sub["temp_mean"] <= q[2])
    cols["hot"] = df_sub["temp_mean"] > q[2]
    q = bin_edges["pressure_mean"]
    cols["low_pressure"] = df_sub["pressure_mean"] <= q[1]
    cols["mid_pressure"] = (df_sub["pressure_mean"] > q[1]) & (df_sub["pressure_mean"] <= q[2])
    cols["high_pressure"] = df_sub["pressure_mean"] > q[2]
    cols["no_rain"] = df_sub["precip_sum"] == 0
    cols["rain"] = df_sub["precip_sum"] > 0
    for s in ["winter", "pre_monsoon", "monsoon", "post_monsoon"]:
        cols[f"season_{s}"] = df_sub["season"] == s
    cols["diwali_window"] = df_sub["is_festival_window"] == 1
    cols["weekend"] = df_sub["is_weekend"] == 1
    cols["yesterday_high"] = df_sub["today_high_risk"] == 1
    cols["next_high_risk"] = df_sub["next_high_risk"] == 1
    return pd.DataFrame(cols, index=df_sub.index)

train_enc = encode_df(train)
test_enc = encode_df(test)

frequent_all = fpgrowth(train_enc, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_all, metric="confidence", min_threshold=0.60)
target_consequent = frozenset(["next_high_risk"])
target_rules = rules[(rules["consequents"] == target_consequent) & (rules["lift"] >= 1.20)].copy()

target_rules["has_yesterday"] = target_rules["antecedents"].apply(lambda s: "yesterday_high" in s)
target_rules["k"] = target_rules["antecedents"].apply(len)

set_a = target_rules[~target_rules["has_yesterday"]].copy()
set_b = target_rules[target_rules["has_yesterday"]].copy()

print(f"Total target rules: {len(target_rules)}")
print(f"Set A target rules: {len(set_a)}")
print(f"Set B target rules: {len(set_b)}")

print("\n--- Top 10 Set A Rules by Lift ---")
for _, r in set_a.sort_values("lift", ascending=False).head(10).iterrows():
    items = sorted(list(r["antecedents"]))
    print(f"{' + '.join(items):<55} | Sup={r['support']:.3f} Conf={r['confidence']:.3f} Lift={r['lift']:.2f}")

print("\n--- Top 10 Set B Rules by Lift ---")
for _, r in set_b.sort_values("lift", ascending=False).head(10).iterrows():
    items = sorted(list(r["antecedents"]))
    print(f"{' + '.join(items):<55} | Sup={r['support']:.3f} Conf={r['confidence']:.3f} Lift={r['lift']:.2f}")
