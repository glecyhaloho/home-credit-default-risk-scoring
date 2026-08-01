"""Aggregate credit_card_balance.csv to one row per SK_ID_CURR."""
import gc
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading credit_card_balance.csv ...")
cc = pd.read_csv(
    RAW_DIR / "credit_card_balance.csv",
    dtype={
        "SK_ID_PREV": "int32",
        "SK_ID_CURR": "int32",
        "MONTHS_BALANCE": "int16",
        "AMT_BALANCE": "float32",
        "AMT_CREDIT_LIMIT_ACTUAL": "float32",
        "AMT_PAYMENT_TOTAL_CURRENT": "float32",
        "AMT_INST_MIN_REGULARITY": "float32",
        "AMT_DRAWINGS_CURRENT": "float32",
        "CNT_DRAWINGS_CURRENT": "float32",
        "SK_DPD": "int32",
        "SK_DPD_DEF": "int32",
    },
    usecols=[
        "SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL", "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_INST_MIN_REGULARITY", "AMT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_CURRENT", "SK_DPD", "SK_DPD_DEF",
    ],
)
print(cc.shape)

cc["UTILIZATION"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
cc["IS_LATE"] = (cc["SK_DPD"] > 0).astype("int8")
cc["UNDERPAYMENT"] = (cc["AMT_PAYMENT_TOTAL_CURRENT"] < cc["AMT_INST_MIN_REGULARITY"]).astype("int8")

agg = cc.groupby("SK_ID_CURR").agg(
    CC_CNT=("SK_ID_PREV", "count"),
    CC_NUNIQUE_PREV=("SK_ID_PREV", "nunique"),
    CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
    CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
    CC_UTILIZATION_MAX=("UTILIZATION", "max"),
    CC_SK_DPD_MEAN=("SK_DPD", "mean"),
    CC_SK_DPD_MAX=("SK_DPD", "max"),
    CC_LATE_RATIO=("IS_LATE", "mean"),
    CC_UNDERPAYMENT_RATIO=("UNDERPAYMENT", "mean"),
    CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
    CC_CNT_DRAWINGS_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
).reset_index()

del cc
gc.collect()

out_path = PROCESSED_DIR / "cc_agg.parquet"
agg.to_parquet(out_path, index=False)
print("Saved:", out_path, agg.shape)
