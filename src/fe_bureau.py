"""Aggregate bureau.csv + bureau_balance.csv to one row per SK_ID_CURR."""
import gc
import numpy as np
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading bureau_balance.csv ...")
bb = pd.read_csv(
    RAW_DIR / "bureau_balance.csv",
    dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"},
)
print(bb.shape)

# STATUS: C=closed, X=unknown, 0=no DPD, 1..5 = increasing DPD buckets
bb["DPD_FLAG"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")

bb_agg = bb.groupby("SK_ID_BUREAU").agg(
    BB_MONTHS_COUNT=("MONTHS_BALANCE", "size"),
    BB_DPD_MONTHS=("DPD_FLAG", "sum"),
).reset_index()
bb_agg["BB_DPD_RATIO"] = bb_agg["BB_DPD_MONTHS"] / bb_agg["BB_MONTHS_COUNT"]

del bb
gc.collect()
print("bureau_balance aggregated:", bb_agg.shape)

print("Loading bureau.csv ...")
bureau = pd.read_csv(
    RAW_DIR / "bureau.csv",
    dtype={
        "SK_ID_CURR": "int32",
        "SK_ID_BUREAU": "int32",
        "CREDIT_ACTIVE": "category",
        "CREDIT_CURRENCY": "category",
        "DAYS_CREDIT": "int32",
        "CREDIT_DAY_OVERDUE": "int32",
        "DAYS_CREDIT_ENDDATE": "float32",
        "DAYS_ENDDATE_FACT": "float32",
        "AMT_CREDIT_MAX_OVERDUE": "float32",
        "CNT_CREDIT_PROLONG": "int32",
        "AMT_CREDIT_SUM": "float32",
        "AMT_CREDIT_SUM_DEBT": "float32",
        "AMT_CREDIT_SUM_LIMIT": "float32",
        "AMT_CREDIT_SUM_OVERDUE": "float32",
        "CREDIT_TYPE": "category",
        "DAYS_CREDIT_UPDATE": "int32",
        "AMT_ANNUITY": "float32",
    },
)
print(bureau.shape)

bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
del bb_agg
gc.collect()

bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype("int8")
bureau["IS_OVERDUE"] = (bureau["AMT_CREDIT_SUM_OVERDUE"] > 0).astype("int8")
bureau["CREDIT_SUM_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1)

agg = bureau.groupby("SK_ID_CURR").agg(
    BUREAU_CNT_CREDITS=("SK_ID_BUREAU", "count"),
    BUREAU_CNT_ACTIVE=("IS_ACTIVE", "sum"),
    BUREAU_CNT_OVERDUE=("IS_OVERDUE", "sum"),
    BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
    BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
    BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
    BUREAU_AMT_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
    BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
    BUREAU_AMT_CREDIT_SUM_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
    BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    BUREAU_CREDIT_SUM_DEBT_RATIO_MEAN=("CREDIT_SUM_DEBT_RATIO", "mean"),
    BUREAU_CNT_PROLONGED=("CNT_CREDIT_PROLONG", "sum"),
    BUREAU_BB_DPD_RATIO_MEAN=("BB_DPD_RATIO", "mean"),
    BUREAU_BB_MONTHS_COUNT_MEAN=("BB_MONTHS_COUNT", "mean"),
).reset_index()

agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_CNT_ACTIVE"] / agg["BUREAU_CNT_CREDITS"]
agg["BUREAU_OVERDUE_RATIO"] = agg["BUREAU_CNT_OVERDUE"] / agg["BUREAU_CNT_CREDITS"]

del bureau
gc.collect()

out_path = PROCESSED_DIR / "bureau_agg.parquet"
agg.to_parquet(out_path, index=False)
print("Saved:", out_path, agg.shape)
