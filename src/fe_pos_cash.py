"""Aggregate POS_CASH_balance.csv to one row per SK_ID_CURR."""
import gc
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading POS_CASH_balance.csv ...")
pos = pd.read_csv(
    RAW_DIR / "POS_CASH_balance.csv",
    dtype={
        "SK_ID_PREV": "int32",
        "SK_ID_CURR": "int32",
        "MONTHS_BALANCE": "int16",
        "CNT_INSTALMENT": "float32",
        "CNT_INSTALMENT_FUTURE": "float32",
        "NAME_CONTRACT_STATUS": "category",
        "SK_DPD": "int32",
        "SK_DPD_DEF": "int32",
    },
)
print(pos.shape)

pos["IS_LATE"] = (pos["SK_DPD"] > 0).astype("int8")

agg = pos.groupby("SK_ID_CURR").agg(
    POS_CNT=("SK_ID_PREV", "count"),
    POS_NUNIQUE_PREV=("SK_ID_PREV", "nunique"),
    POS_SK_DPD_MEAN=("SK_DPD", "mean"),
    POS_SK_DPD_MAX=("SK_DPD", "max"),
    POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
    POS_LATE_RATIO=("IS_LATE", "mean"),
    POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
).reset_index()

del pos
gc.collect()

out_path = PROCESSED_DIR / "pos_agg.parquet"
agg.to_parquet(out_path, index=False)
print("Saved:", out_path, agg.shape)
