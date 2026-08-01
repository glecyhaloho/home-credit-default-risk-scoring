"""Aggregate previous_application.csv to one row per SK_ID_CURR."""
import gc
import numpy as np
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading previous_application.csv ...")
prev = pd.read_csv(
    RAW_DIR / "previous_application.csv",
    dtype={
        "SK_ID_PREV": "int32",
        "SK_ID_CURR": "int32",
        "NAME_CONTRACT_TYPE": "category",
        "AMT_ANNUITY": "float32",
        "AMT_APPLICATION": "float32",
        "AMT_CREDIT": "float32",
        "AMT_DOWN_PAYMENT": "float32",
        "AMT_GOODS_PRICE": "float32",
        "RATE_DOWN_PAYMENT": "float32",
        "NAME_CONTRACT_STATUS": "category",
        "DAYS_DECISION": "int32",
        "CNT_PAYMENT": "float32",
        "NFLAG_INSURED_ON_APPROVAL": "float32",
    },
    usecols=[
        "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_ANNUITY",
        "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT", "NAME_CONTRACT_STATUS", "DAYS_DECISION",
        "CNT_PAYMENT", "NFLAG_INSURED_ON_APPROVAL",
    ],
)
print(prev.shape)

prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype("int8")
prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")
prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1)

agg = prev.groupby("SK_ID_CURR").agg(
    PREV_CNT=("SK_ID_PREV", "count"),
    PREV_APPROVED_CNT=("IS_APPROVED", "sum"),
    PREV_REFUSED_CNT=("IS_REFUSED", "sum"),
    PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
    PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
    PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
    PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
    PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
    PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
    PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
    PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    PREV_INSURED_RATE=("NFLAG_INSURED_ON_APPROVAL", "mean"),
).reset_index()

agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_CNT"] / agg["PREV_CNT"]
agg["PREV_REFUSAL_RATE"] = agg["PREV_REFUSED_CNT"] / agg["PREV_CNT"]

del prev
gc.collect()

out_path = PROCESSED_DIR / "prev_agg.parquet"
agg.to_parquet(out_path, index=False)
print("Saved:", out_path, agg.shape)
