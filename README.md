# Home Credit Default Risk — Credit Scoring Analysis

Rakamin Academy x Home Credit Indonesia Virtual Internship — Task 5.

End-to-end credit default risk modeling on the Kaggle "Home Credit Default
Risk" dataset: data cleaning, feature engineering across 6 relational tables,
two machine learning models (Logistic Regression + LightGBM), evaluation, and
business recommendations (risk-based loan structuring, acquisition targeting).

## Notebook

See [`notebooks/Home_Credit_Default_Risk_Analysis.ipynb`](notebooks/Home_Credit_Default_Risk_Analysis.ipynb)
for the full, executed, end-to-end analysis.

## Results

| Model | Test AUC | KS |
|---|---|---|
| Logistic Regression (baseline) | 0.772 | 0.409 |
| LightGBM (tuned) | 0.786 | 0.437 |

## Project structure

```
src/                    reusable feature engineering, training & evaluation code
  fe_bureau.py          aggregate bureau.csv + bureau_balance.csv
  fe_previous_application.py
  fe_pos_cash.py
  fe_credit_card.py
  fe_installments.py
  build_master_dataset.py   merge + engineer ratio features -> master dataset
  train_logreg.py       Logistic Regression + GridSearchCV
  train_lightgbm.py     LightGBM + RandomizedSearchCV
  business_insights.py  segment analysis + decile gains table
notebooks/              end-to-end analysis notebook
reports/figures/        saved charts
models/                 saved model artifacts + metrics (gitignored)
```

## Data

Raw data is the Kaggle "Home Credit Default Risk" competition dataset
(`application_train/test.csv`, `bureau.csv`, `bureau_balance.csv`,
`previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`,
`installments_payments.csv`, `HomeCredit_columns_description.csv`). Not
included in this repo (competition data / size); download from Kaggle and
place under `../home-credit-default-risk/` relative to this project, or adjust
`src/config.py:RAW_DIR`.

## Reproducing

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/Home_Credit_Default_Risk_Analysis.ipynb
```
