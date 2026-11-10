import pandas as pd
import numpy as np
import os

base = r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study"
df = pd.read_csv(os.path.join(base, "m4", "m4_full_engineered.csv"), parse_dates=["date"]).set_index("date")
df["next_high_risk"] = df["high_risk"].shift(-1, freq="D")

# What if day t's label was also checked?
# In the prompt:
# "With support ≥ 5%, confidence ≥ 0.6 and lift ≥ 1.2, you get 64 rules for set A and 103 for set B (mining on pre-2025 data, 2,287 day-pairs)."
# "Pair days correctly. Day t’s weather goes with day t+1’s high_risk. Drop any pair where either label is missing, and never fill missing labels with 0."

# Let's see how many pairs if:
# 1) only next_high_risk is notna:
p1 = df[(df.index < "2025-01-01") & df["next_high_risk"].notna()]
print("p1 (next_high_risk notna):", len(p1))

# 2) df was df.dropna(subset=['high_risk']), and shifted:
df_sub = df.dropna(subset=["high_risk"]).copy()
df_sub["next_hr"] = df_sub["high_risk"].shift(-1)
# how many before 2025?
p2 = df_sub[df_sub.index < "2025-01-01"]
print("p2 (df_sub where date < 2025-01-01):", len(p2))
# if dropna on next_hr:
print("p2 with next_hr notna:", p2["next_hr"].notna().sum())

# 3) calendar-aligned where both are notna:
p3 = df[(df.index < "2025-01-01") & df["high_risk"].notna() & df["next_high_risk"].notna()]
print("p3 (both notna):", len(p3))
