"""Train & tune a LightGBM gradient boosting model (higher-performance challenger model)."""
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from config import MODELS_DIR
from data_utils import load_master, get_feature_columns, split
from metrics_utils import evaluate_model, plot_confusion_matrix

print("Loading master dataset ...")
df = load_master()
num_cols, cat_cols = get_feature_columns(df)
print(f"{len(num_cols)} numeric cols, {len(cat_cols)} categorical cols")

X_train, X_test, y_train, y_test = split(df)

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_test[c] = X_test[c].astype("category").cat.set_categories(X_train[c].cat.categories)

print("Train:", X_train.shape, "Test:", X_test.shape)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("scale_pos_weight:", scale_pos_weight)

base_clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=400,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

param_dist = {
    "num_leaves": randint(16, 64),
    "learning_rate": uniform(0.01, 0.09),
    "min_child_samples": randint(20, 150),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0, 5),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    base_clf, param_dist, n_iter=20, scoring="roc_auc", cv=cv,
    random_state=42, n_jobs=1, verbose=2,
)
search.fit(X_train, y_train, categorical_feature=cat_cols)

print("Best params:", search.best_params_)
print("Best CV AUC:", search.best_score_)

best_model = search.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]

metrics, fpr, tpr = evaluate_model("LightGBM", y_test.values, y_prob)
metrics["best_params"] = search.best_params_
metrics["cv_auc"] = search.best_score_

plot_confusion_matrix(np.array(metrics["confusion_matrix"]), "LightGBM", "confusion_matrix_lightgbm.png")

joblib.dump(best_model, MODELS_DIR / "lightgbm_model.joblib")
np.save(MODELS_DIR / "lightgbm_fpr.npy", fpr)
np.save(MODELS_DIR / "lightgbm_tpr.npy", tpr)
np.save(MODELS_DIR / "lightgbm_y_test.npy", y_test.values)
np.save(MODELS_DIR / "lightgbm_y_prob.npy", y_prob)

with open(MODELS_DIR / "lightgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# feature importance
fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False)
fi.to_csv(MODELS_DIR / "lightgbm_feature_importance.csv", index=False)
print(fi.head(20))

print("Done. Saved lightgbm_model.joblib + lightgbm_metrics.json")
