"""Train & tune a Logistic Regression baseline (interpretable scorecard model).

Preprocessing (impute+scale+one-hot) is fit ONCE outside the CV loop and the
resulting sparse matrix is reused across folds/workers. Doing it inside a
sklearn Pipeline passed to GridSearchCV would force joblib to pickle the full
heterogeneous DataFrame + refit the encoder for every fold/worker, which blew
past this machine's 8GB RAM and thrashed swap.
"""
import gc
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import MODELS_DIR
from data_utils import load_master, get_feature_columns, split
from metrics_utils import evaluate_model, plot_confusion_matrix

print("Loading master dataset ...")
df = load_master()
num_cols, cat_cols = get_feature_columns(df)
print(f"{len(num_cols)} numeric cols, {len(cat_cols)} categorical cols")

for c in num_cols:
    df[c] = df[c].astype("float32")
for c in cat_cols:
    df[c] = df[c].astype("category")

X_train, X_test, y_train, y_test = split(df)
del df
gc.collect()
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train default rate:", y_train.mean(), "Test default rate:", y_test.mean())

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), cat_cols),
    ]
)

print("Fitting preprocessor once and transforming train/test ...")
X_train_t = preprocessor.fit_transform(X_train, y_train)
X_test_t = preprocessor.transform(X_test)
print("Transformed shapes:", X_train_t.shape, X_test_t.shape)

del X_train, X_test
gc.collect()

param_grid = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
base_clf = LogisticRegression(class_weight="balanced", max_iter=300, solver="lbfgs", random_state=42)

search = GridSearchCV(base_clf, param_grid, scoring="roc_auc", cv=cv, n_jobs=3, verbose=2)
search.fit(X_train_t, y_train)

print("Best params:", search.best_params_)
print("Best CV AUC:", search.best_score_)

best_clf = search.best_estimator_
y_prob = best_clf.predict_proba(X_test_t)[:, 1]

metrics, fpr, tpr = evaluate_model("LogisticRegression", y_test.values, y_prob)
metrics["best_params"] = search.best_params_
metrics["cv_auc"] = search.best_score_

plot_confusion_matrix(np.array(metrics["confusion_matrix"]), "Logistic Regression", "confusion_matrix_logreg.png")

# repackage as a single deployable pipeline (preprocessor + tuned classifier)
full_pipeline = Pipeline([("prep", preprocessor), ("clf", best_clf)])
joblib.dump(full_pipeline, MODELS_DIR / "logreg_model.joblib")

np.save(MODELS_DIR / "logreg_fpr.npy", fpr)
np.save(MODELS_DIR / "logreg_tpr.npy", tpr)
np.save(MODELS_DIR / "logreg_y_test.npy", y_test.values)
np.save(MODELS_DIR / "logreg_y_prob.npy", y_prob)

# feature names + coefficients for interpretability
feat_names = preprocessor.get_feature_names_out()
coef_df_data = {"feature": feat_names, "coefficient": best_clf.coef_[0]}
import pandas as pd
coef_df = pd.DataFrame(coef_df_data).sort_values("coefficient", key=abs, ascending=False)
coef_df.to_csv(MODELS_DIR / "logreg_coefficients.csv", index=False)

with open(MODELS_DIR / "logreg_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done. Saved logreg_model.joblib + logreg_metrics.json")
