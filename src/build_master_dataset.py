"""Merge application_train with aggregated auxiliary features + engineer ratio features."""
import gc
import numpy as np
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading application_train.csv ...")
app = pd.read_csv(RAW_DIR / "application_train.csv")
print(app.shape)

# --- anomaly handling ---
app["DAYS_EMPLOYED_ANOM"] = (app["DAYS_EMPLOYED"] == 365243).astype("int8")
app.loc[app["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

# --- application-level engineered ratios ---
app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / (app["AMT_INCOME_TOTAL"] + 1)
app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"] / (app["AMT_INCOME_TOTAL"] + 1)
app["CREDIT_TERM"] = app["AMT_ANNUITY"] / (app["AMT_CREDIT"] + 1)
app["GOODS_PRICE_CREDIT_RATIO"] = app["AMT_GOODS_PRICE"] / (app["AMT_CREDIT"] + 1)
app["DAYS_EMPLOYED_PERC"] = app["DAYS_EMPLOYED"] / (app["DAYS_BIRTH"] + 1)
app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365
app["EMPLOYED_YEARS"] = -app["DAYS_EMPLOYED"] / 365
app["INCOME_PER_FAM_MEMBER"] = app["AMT_INCOME_TOTAL"] / (app["CNT_FAM_MEMBERS"] + 1)

ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
app["EXT_SOURCE_MEAN"] = app[ext_cols].mean(axis=1)
app["EXT_SOURCE_STD"] = app[ext_cols].std(axis=1)
app["EXT_SOURCE_MIN"] = app[ext_cols].min(axis=1)
app["EXT_SOURCE_MAX"] = app[ext_cols].max(axis=1)
app["EXT_SOURCE_NAN_CNT"] = app[ext_cols].isna().sum(axis=1)

# --- merge auxiliary aggregated tables ---
for fname in ["bureau_agg", "prev_agg", "pos_agg", "cc_agg", "inst_agg"]:
    aux = pd.read_parquet(PROCESSED_DIR / f"{fname}.parquet")
    app = app.merge(aux, on="SK_ID_CURR", how="left")
    del aux
    gc.collect()
    print(f"merged {fname}, shape now {app.shape}")

# count-type features: fill missing (no history) with 0
count_like_cols = [c for c in app.columns if c.endswith("_CNT") or "NUNIQUE" in c]
for c in count_like_cols:
    app[c] = app[c].fillna(0)

# has-history flags (useful, interpretable signal for "thin-file" clients)
app["HAS_BUREAU_HISTORY"] = (app["BUREAU_CNT_CREDITS"] > 0).astype("int8")
app["HAS_PREV_HISTORY"] = (app["PREV_CNT"] > 0).astype("int8")

out_path = PROCESSED_DIR / "master_train.parquet"
app.to_parquet(out_path, index=False)
print("Saved:", out_path, app.shape)
print("Missing value share (top 15):")
print((app.isna().mean().sort_values(ascending=False) * 100).head(15))
