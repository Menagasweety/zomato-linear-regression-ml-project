# Zomato Delivery Time — Linear Regression ML Project

## Project objective

Predict `Time_taken (min)` — the total delivery time in minutes — **at the moment the customer places the order**.

That prediction timing creates an explicit leakage rule:

> `Time_Order_picked` must not be used as an input because pickup happens after the prediction moment.

## Dataset

The uploaded dataset contains **45,584 rows and 20 columns**.

The repository is based on the uploaded `Zomato Dataset.csv`. GitHub references were used as methodology/context checks, not as substitutes for inspecting this file.

Reference repositories:
- https://github.com/Parth-Malik/Zomato-Delivery-Time-Prediction
- https://github.com/CodeByMiraj/Zomato-Delivery-Analytics
- https://github.com/Varunnagar/Zomato-Food-Delivery-Analysis
- https://github.com/PN-Piyush/Zomato-Delivery-Analytics

## Project structure

```text
Zomato_Linear_Regression_ML_Project/
├── data/
│   └── Zomato Dataset.csv
├── notebooks/
│   ├── 01_Data_Audit.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Baseline_Model.ipynb
│   ├── 06_Linear_Regression.ipynb
│   ├── 07_Regularization_Evaluation.ipynb
│   ├── 08_Final_Evaluation.ipynb
│   └── 09_Optional_Classification_Confusion_Matrix.ipynb
├── outputs/
│   ├── cleaned/
│   ├── model_results/
│   ├── figures/
│   └── models/
├── README.md
├── requirements.txt
└── .gitignore
```

## Learning path

1. **Data Audit** — inspect before changing anything.
2. **Cleaning** — standardize types, missing values, invalid values and duplicates.
3. **EDA** — understand target distribution and relationships.
4. **Feature Engineering** — create distance and order-time features available at prediction time.
5. **Baseline** — establish a simple benchmark.
6. **Linear Regression** — learn least-squares optimization and coefficients.
7. **Regularization** — compare Ridge, Lasso and ElasticNet using cross-validation.
8. **Final Evaluation** — inspect MAE, RMSE, R², train/test gap and residuals.
9. **Optional Classification** — only here do we use a confusion matrix.

## Core ML pipeline

```text
Raw CSV
  ↓
Audit
  ↓
Cleaning
  ↓
Feature Engineering
  ↓
Train / Test Split
  ↓
ColumnTransformer
  ├── Numeric → median imputation → scaling
  └── Categorical → most-frequent imputation → one-hot encoding
  ↓
Regression Model
  ├── Linear Regression
  ├── Ridge
  ├── Lasso
  └── ElasticNet
  ↓
Cross-validation / hyperparameter tuning
  ↓
Final test evaluation
  ↓
Residual diagnostics
```

## Why Pipeline?

The scikit-learn `Pipeline` ensures preprocessing is learned from training data rather than accidentally using information from the test set.

This is an important form of leakage prevention.

## Metrics

### MAE
Average absolute prediction error in minutes.

### RMSE
Square-root of mean squared error. Larger errors receive more penalty.

### R²
Measures how much variation in the target is explained relative to a mean-prediction baseline.

## Regularization

- **Linear Regression:** minimizes squared error.
- **Ridge:** squared error + L2 coefficient penalty.
- **Lasso:** squared error + L1 coefficient penalty.
- **ElasticNet:** squared error + combination of L1 and L2 penalties.

The best model is selected from validation/test evidence, not from the name of the algorithm.

## Important modeling rule

`Time_Order_picked` is intentionally excluded.

If the model were deployed when the customer places an order, pickup time would not yet exist. Including it would make offline evaluation unrealistically optimistic.

## Confusion matrix

A confusion matrix is **not appropriate for the main regression problem** because delivery time is continuous.

The optional notebook converts the historical target into a binary classification label (`Delayed` vs `On Time`) to demonstrate when a confusion matrix is appropriate.

## Reproducibility

Run notebooks in this order:

`01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`

Notebook 09 is optional.

## Important note

Some GitHub references report additional cleaning decisions for similar copies of this dataset. This project does not blindly copy those decisions. The notebooks inspect and document the actual uploaded CSV before applying transformations.
