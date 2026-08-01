"""Data-driven business insights: segment default rates, population share, and a
model-based decile lift table to quantify the impact of using the score for
accept/reject decisions."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR
from data_utils import load_master

pd.set_option("display.width", 140)

df = load_master()
overall_default_rate = df["TARGET"].mean()
print(f"Overall default rate: {overall_default_rate:.4f}")


def segment_table(col, min_count=500, top_n=12):
    g = df.groupby(col, observed=True).agg(
        n=("TARGET", "size"),
        default_rate=("TARGET", "mean"),
    ).reset_index()
    g["share_pct"] = g["n"] / g["n"].sum() * 100
    g = g[g["n"] >= min_count].sort_values("default_rate")
    return g.head(top_n) if len(g) > top_n else g


segments = {}
for col in [
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE", "CODE_GENDER", "NAME_HOUSING_TYPE",
    "NAME_FAMILY_STATUS",
]:
    tab = segment_table(col)
    segments[col] = tab
    print(f"\n=== {col} (overall default rate {overall_default_rate:.2%}) ===")
    print(tab.to_string(index=False))

# quartile-based numeric cuts
for col in ["CREDIT_TERM", "EXT_SOURCE_MEAN", "ANNUITY_INCOME_RATIO", "PREV_REFUSAL_RATE"]:
    tmp = df[[col, "TARGET"]].dropna()
    tmp["bin"] = pd.qcut(tmp[col], 4, duplicates="drop")
    g = tmp.groupby("bin", observed=True).agg(n=("TARGET", "size"), default_rate=("TARGET", "mean")).reset_index()
    segments[col] = g
    print(f"\n=== {col} quartiles ===")
    print(g.to_string(index=False))

# thin-file vs bureau history
g = df.groupby("HAS_BUREAU_HISTORY").agg(n=("TARGET", "size"), default_rate=("TARGET", "mean")).reset_index()
print("\n=== HAS_BUREAU_HISTORY ===")
print(g.to_string(index=False))
segments["HAS_BUREAU_HISTORY"] = g

# save all segment tables to one json-friendly dict
out = {k: v.astype(str).to_dict(orient="records") if False else json.loads(v.to_json(orient="records")) for k, v in segments.items()}
with open(MODELS_DIR / "segment_insights.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nSaved segment_insights.json")

# ---- decile lift table using LightGBM (best model) predictions ----
y_test = np.load(MODELS_DIR / "lightgbm_y_test.npy")
y_prob = np.load(MODELS_DIR / "lightgbm_y_prob.npy")

lift = pd.DataFrame({"y": y_test, "prob": y_prob})
lift["decile"] = pd.qcut(lift["prob"], 10, labels=False, duplicates="drop")
lift["decile_display"] = 10 - lift["decile"]  # 1 = highest risk decile

decile_tab = lift.groupby("decile_display").agg(
    n=("y", "size"),
    bad_count=("y", "sum"),
    default_rate=("y", "mean"),
    avg_score=("prob", "mean"),
).reset_index().sort_values("decile_display")

decile_tab["cum_bad_captured_pct"] = decile_tab["bad_count"].cumsum() / decile_tab["bad_count"].sum() * 100
decile_tab["cum_applicants_pct"] = decile_tab["n"].cumsum() / decile_tab["n"].sum() * 100
decile_tab["cum_good_affected"] = (decile_tab["n"] - decile_tab["bad_count"]).cumsum()
decile_tab["cum_good_affected_pct"] = decile_tab["cum_good_affected"] / (decile_tab["n"].sum() - decile_tab["bad_count"].sum()) * 100

decile_tab.to_csv(MODELS_DIR / "decile_lift_table.csv", index=False)
print("\n=== Decile lift table (decile 1 = highest risk) ===")
print(decile_tab.to_string(index=False))

# plot cumulative bad capture vs applicants declined
plt.figure(figsize=(6, 5))
plt.plot(decile_tab["cum_applicants_pct"], decile_tab["cum_bad_captured_pct"], marker="o", label="Defaults captured")
plt.plot(decile_tab["cum_applicants_pct"], decile_tab["cum_good_affected_pct"], marker="o", label="Good payers affected")
plt.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Random baseline")
plt.xlabel("% of applicants declined (highest risk first)")
plt.ylabel("% captured")
plt.title("Model-based Reject Policy: Gains Curve")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "gains_curve.png", dpi=150)
plt.close()
print("Saved gains_curve.png")
