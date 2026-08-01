"""Shared helpers to load the master dataset and produce a consistent train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split
from config import PROCESSED_DIR

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

DROP_COLS = [ID_COL, TARGET_COL]


def load_master():
    df = pd.read_parquet(PROCESSED_DIR / "master_train.parquet")
    return df


def get_feature_columns(df):
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if c not in cat_cols and c not in DROP_COLS]
    return num_cols, cat_cols


def split(df, test_size=0.2, random_state=42):
    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
