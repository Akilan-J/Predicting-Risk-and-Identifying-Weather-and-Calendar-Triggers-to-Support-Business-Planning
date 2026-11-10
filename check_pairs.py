import pandas as pd
import numpy as np
import os
import sys

base = r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study"
df = pd.read_csv(os.path.join(base, "m4", "m4_full_engineered.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)

print(f"Total rows in m4_full_engineered: {len(df)}")
print(f"Min date: {df['date'].min()}, Max date: {df['date'].max()}")

# Look at date gaps if any
date_diff = df["date"].diff()
print(f"Any non-1-day diffs? {(date_diff[1:] != pd.Timedelta(days=1)).sum()}")

# High risk missingness
print(f"high_risk missing: {df['high_risk'].isna().sum()}")

# Construct day-pairs: Day t weather/season/persistence -> Day t+1 high_risk
# Let's check:
df["target_next"] = df["high_risk"].shift(-1)
df["target_today"] = df["high_risk"]

# Check drop condition:
# "Drop any pair where either label is missing, and never fill missing labels with 0."
valid_pairs = df[df["target_today"].notna() & df["target_next"].notna()].copy()
print(f"Total valid pairs: {len(valid_pairs)}")

# Train vs test split: "mining on pre-2025 data"
# Does "pre-2025" mean date < 2025-01-01 for day t or day t+1?
# Let's check both:
train_by_t = valid_pairs[valid_pairs["date"] < "2025-01-01"]
train_by_next = valid_pairs[valid_pairs["date"] < "2024-12-31"] # so next_date < 2025-01-01
print(f"Train pairs with date < 2025-01-01: {len(train_by_t)}")
print(f"Train pairs with next_date < 2025-01-01: {len(train_by_next)}")

# Let's check base rates in train_by_t:
print("\nBase rates with date < 2025-01-01:")
print(f"Overall next-day high_risk mean: {train_by_t['target_next'].mean():.4f}")
print("By season column of day t:")
print(train_by_t.groupby("season")["target_next"].agg(["count", "mean"]))
print("By season column of day t+1:")
train_by_t["next_season"] = df["season"].shift(-1).loc[train_by_t.index]
print(train_by_t.groupby("next_season")["target_next"].agg(["count", "mean"]))

# Let's also check test set (2025+):
test = valid_pairs[valid_pairs["date"] >= "2025-01-01"]
print(f"\nTest pairs count: {len(test)}")
print(f"Test next-day high_risk mean: {test['target_next'].mean():.4f}")

# Check yesterday-only rule:
# If today is high risk (target_today == 1), what is P(target_next == 1)?
yest_high = train_by_t[train_by_t["target_today"] == 1]
conf_yest = yest_high["target_next"].mean()
base_rate = train_by_t["target_next"].mean()
lift_yest = conf_yest / base_rate
print(f"\n'Yesterday was high' rule on train_by_t:")
print(f"Count: {len(yest_high)} / {len(train_by_t)} (support = {len(yest_high)/len(train_by_t):.4f})")
print(f"Confidence = {conf_yest:.4f}")
print(f"Lift = {lift_yest:.4f}")
