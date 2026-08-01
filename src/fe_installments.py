"""Aggregate installments_payments.csv to one row per SK_ID_CURR."""
import gc
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR

print("Loading installments_payments.csv ...")
inst = pd.read_csv(
    RAW_DIR / "installments_payments.csv",
    dtype={
        "SK_ID_PREV": "int32",
        "SK_ID_CURR": "int32",
        "NUM_INSTALMENT_NUMBER": "int32",
        "DAYS_INSTALMENT": "float32",
        "DAYS_ENTRY_PAYMENT": "float32",
        "AMT_INSTALMENT": "float32",
        "AMT_PAYMENT": "float32",
    },
    usecols=[
        "SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
    ],
)
print(inst.shape)

inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype("int8")
inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 1)
inst["IS_UNDERPAID"] = (inst["PAYMENT_RATIO"] < 0.99).astype("int8")
inst["MISSED"] = inst["DAYS_ENTRY_PAYMENT"].isna().astype("int8")

agg = inst.groupby("SK_ID_CURR").agg(
    INST_CNT=("NUM_INSTALMENT_NUMBER", "count"),
    INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
    INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
    INST_LATE_RATIO=("IS_LATE", "mean"),
    INST_MISSED_RATIO=("MISSED", "mean"),
    INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
    INST_UNDERPAID_RATIO=("IS_UNDERPAID", "mean"),
    INST_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
    INST_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
).reset_index()

del inst
gc.collect()

out_path = PROCESSED_DIR / "inst_agg.parquet"
agg.to_parquet(out_path, index=False)
print("Saved:", out_path, agg.shape)
