import pandas as pd

df = pd.read_csv("horizon_scanning_combined.csv")

# 1) Look for any row that says it’s “Recent ≤ 7 days” but is actually older.
bad1 = df[(df["recency_label"] == "Recent (≤ 7 days)") & (df["days_old"] > 7)]

# 2) Look for any row marked “Older” but with days_old ≤ 90 (should be impossible now).
bad2 = df[(df["recency_label"] == "Older") & (df["days_old"] <= 90)]

print("Mismatch rows:", len(bad1) + len(bad2))
