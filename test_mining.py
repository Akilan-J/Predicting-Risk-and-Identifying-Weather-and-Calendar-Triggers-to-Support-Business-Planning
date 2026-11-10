import pandas as pd
import numpy as np
import itertools
import os

base = r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study"
df = pd.read_csv(os.path.join(base, "m4", "m4_full_engineered.csv"), parse_dates=["date"]).set_index("date")

# Day t weather + Day t+1 high_risk
df["next_high_risk"] = df["high_risk"].shift(-1, freq="D")
df["target_today"] = df["high_risk"]

# Pre-2025 train:
# Let's test both train row sets:
# A) All rows before 2025 where next_high_risk.notna() (2287/2288 rows)
# B) Rows where both today and next are notna (2159 rows)

for name, tr in [
    ("2288_next_only", df[(df.index < "2025-01-01") & df["next_high_risk"].notna()].copy()),
    ("2159_both_notna", df[(df.index < "2025-01-01") & df["high_risk"].notna() & df["next_high_risk"].notna()].copy())
]:
    print(f"\n=================== Dataset: {name} (N={len(tr)}) ===================")
    base_rate = tr["next_high_risk"].mean()
    print(f"Base rate: {base_rate:.4f}")

    # Tercile binning on train:
    items = pd.DataFrame(index=tr.index)

    # 1. Weather terciles:
    for var, vname in [
        ("temp_mean", "temp"),
        ("humidity_mean", "hum"),
        ("wind_speed_mean", "wind"),
        ("pressure_mean", "pres")
    ]:
        q = np.quantile(tr[var].dropna(), [0, 1/3, 2/3, 1])
        items[f"{vname}_low"] = (tr[var] <= q[1]).astype(int)
        items[f"{vname}_mid"] = ((tr[var] > q[1]) & (tr[var] <= q[2])).astype(int)
        items[f"{vname}_high"] = (tr[var] > q[2]).astype(int)

    # Precipitation: no rain vs rain
    items["no_rain"] = (tr["precip_sum"] == 0).astype(int)
    items["rain"] = (tr["precip_sum"] > 0).astype(int)

    # Season items (day t or day t+1):
    for s in ["monsoon", "pre_monsoon", "post_monsoon", "winter"]:
        items[f"season_{s}"] = (tr["season"] == s).astype(int)

    # Persistence:
    items["yesterday_high"] = (tr["high_risk"] == 1).astype(int)

    # Now let's count rules for Set A (without persistence) and Set B (with persistence)
    set_a_cols = [c for c in items.columns if c != "yesterday_high"]
    set_b_cols = list(items.columns)

    def count_rules(item_cols, max_k=3):
        rules = []
        n = len(tr)
        y = tr["next_high_risk"].values
        for k in range(1, max_k + 1):
            for combo in itertools.combinations(item_cols, k):
                # Filter out incompatible combinations (e.g. temp_low and temp_high)
                prefixes = [c.split("_")[0] for c in combo]
                if len(prefixes) != len(set(prefixes)):
                    continue
                mask = np.ones(n, dtype=bool)
                for c in combo:
                    mask &= (items[c].values == 1)
                sup = mask.sum()
                if sup / n < 0.05:
                    continue
                conf = y[mask].mean()
                if conf < 0.60:
                    continue
                lift = conf / base_rate
                if lift < 1.20:
                    continue
                rules.append((combo, sup/n, conf, lift))
        return rules

    rules_a = count_rules(set_a_cols, max_k=3)
    rules_b = count_rules(set_b_cols, max_k=3)
    print(f"Max k=3 -> Set A: {len(rules_a)}, Set B: {len(rules_b)}")
    
    rules_a4 = count_rules(set_a_cols, max_k=4)
    rules_b4 = count_rules(set_b_cols, max_k=4)
    print(f"Max k=4 -> Set A: {len(rules_a4)}, Set B: {len(rules_b4)}")
