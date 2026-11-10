import pandas as pd
import numpy as np
import os

base = r"C:\Users\jxiak\Documents\7th Sem\business analytics\group study"
df = pd.read_csv(os.path.join(base, "m4", "m4_full_engineered.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# Build calendar-aligned day-pairs:
# Day t: weather, date, high_risk
# Day t+1: date+1, next_high_risk
df_indexed = df.set_index("date")
df_pairs = df_indexed.copy()
df_pairs["next_high_risk"] = df_indexed["high_risk"].shift(-1, freq="D")
df_pairs["target_today"] = df_indexed["high_risk"]

# Filter pre-2025:
# Let's see: what if train is where date < 2025-01-01 and next_high_risk is notna?
train_all = df_pairs[df_pairs.index < "2025-01-01"].copy()
print("Train rows (date < 2025-01-01):", len(train_all))
print("Train rows where next_high_risk.notna():", train_all["next_high_risk"].notna().sum())
print("Train rows where both target_today and next_high_risk are notna:", (train_all["target_today"].notna() & train_all["next_high_risk"].notna()).sum())

# Let's inspect test set as well:
test_all = df_pairs[df_pairs.index >= "2025-01-01"].copy()
print("\nTest rows (date >= 2025-01-01):", len(test_all))
print("Test rows where both are notna:", (test_all["target_today"].notna() & test_all["next_high_risk"].notna()).sum())

# Now let's explore tercile bins of temp_mean, humidity_mean, wind_speed_mean on train:
tr = train_all[(train_all["target_today"].notna()) & (train_all["next_high_risk"].notna())].copy()
te = test_all[(test_all["target_today"].notna()) & (test_all["next_high_risk"].notna())].copy()

# Tercile bins on train:
q_temp = np.quantile(tr["temp_mean"], [0, 1/3, 2/3, 1])
q_hum  = np.quantile(tr["humidity_mean"], [0, 1/3, 2/3, 1])
q_wind = np.quantile(tr["wind_speed_mean"], [0, 1/3, 2/3, 1])

print(f"\nTemp quantiles (train): {q_temp}")
print(f"Humidity quantiles (train): {q_hum}")
print(f"Wind quantiles (train): {q_wind}")

# Check "no rain, cold, low wind"
# If cold = temp_mean <= q_temp[1]
# low wind = wind_speed_mean <= q_wind[1]
# no rain = precip_sum == 0
tr_cold = tr["temp_mean"] <= q_temp[1]
tr_low_wind = tr["wind_speed_mean"] <= q_wind[1]
tr_no_rain = tr["precip_sum"] == 0
tr_mid_hum = (tr["humidity_mean"] > q_hum[1]) & (tr["humidity_mean"] <= q_hum[2])

rule1_tr = tr_no_rain & tr_cold & tr_low_wind
rule2_tr = tr_mid_hum & tr_cold & tr_low_wind

print(f"\nRule 1: no rain, cold, low wind (Train)")
print(f"  Count: {rule1_tr.sum()}  Conf: {tr.loc[rule1_tr, 'next_high_risk'].mean():.4f}  Lift: {tr.loc[rule1_tr, 'next_high_risk'].mean() / tr['next_high_risk'].mean():.4f}")

print(f"\nRule 2: mid humidity, cold, low wind (Train)")
print(f"  Count: {rule2_tr.sum()}  Conf: {tr.loc[rule2_tr, 'next_high_risk'].mean():.4f}  Lift: {tr.loc[rule2_tr, 'next_high_risk'].mean() / tr['next_high_risk'].mean():.4f}")

# Now on test (using frozen train edges!):
te_cold = te["temp_mean"] <= q_temp[1]
te_low_wind = te["wind_speed_mean"] <= q_wind[1]
te_no_rain = te["precip_sum"] == 0
te_mid_hum = (te["humidity_mean"] > q_hum[1]) & (te["humidity_mean"] <= q_hum[2])

rule1_te = te_no_rain & te_cold & te_low_wind
rule2_te = te_mid_hum & te_cold & te_low_wind

print(f"\nRule 1 on Test (2025-26):")
print(f"  Count: {rule1_te.sum()}  Conf: {te.loc[rule1_te, 'next_high_risk'].mean():.4f}  Lift: {te.loc[rule1_te, 'next_high_risk'].mean() / te['next_high_risk'].mean():.4f}")

print(f"\nRule 2 on Test (2025-26):")
print(f"  Count: {rule2_te.sum()}  Conf: {te.loc[rule2_te, 'next_high_risk'].mean():.4f}  Lift: {te.loc[rule2_te, 'next_high_risk'].mean() / te['next_high_risk'].mean():.4f}")
