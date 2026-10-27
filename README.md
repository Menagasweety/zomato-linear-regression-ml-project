# 🍽️ Zomato Delivery Time Prediction | Machine Learning

A hands-on machine learning project focused on predicting food delivery time using Python and Scikit-learn. This project follows a structured regression workflow, from raw data auditing and exploratory analysis to model development, regularization, and final evaluation.

> **Project Goal:** Predict the total delivery time (`Time_taken (min)`) at the moment a customer places an order, using only information available at that time.

## 📌 Project Overview

Food delivery time depends on several factors, including delivery partner characteristics, restaurant and order information, traffic conditions, and weather. This project explores the dataset and develops regression models to estimate delivery time while applying appropriate data preprocessing and leakage-prevention practices.

- **Project type:** Supervised Machine Learning — Regression
- **Dataset size:** 45,584 rows × 20 columns
- **Target variable:** `Time_taken (min)`
- **Primary objective:** Predict delivery duration in minutes
- **Primary models:** Linear Regression, Ridge, Lasso, and ElasticNet

## 🎯 Project Objectives

- Audit and understand the raw dataset before modifying it.
- Handle missing values, inconsistent data types, invalid values, and duplicates based on evidence from the data.
- Perform Exploratory Data Analysis (EDA) to identify patterns and relationships.
- Engineer meaningful features using information available at prediction time.
- Establish a baseline before training more advanced regression models.
- Understand linear regression coefficients and regularization.
- Evaluate models using appropriate regression metrics.
- Diagnose residuals and investigate potential overfitting.
- Maintain a reproducible workflow using Scikit-learn pipelines.

## 🗂️ Project Structure

```text
Zomato_Linear_Regression_ML_Project/
│
├── data/
│   └── Zomato Dataset.csv
│
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
│
├── outputs/
│   ├── cleaned/
│   ├── model_results/
│   ├── figures/
│   └── models/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 🔄 Machine Learning Workflow

| Stage | Notebook | Purpose |
|---|---|---|
| 1 | `01_Data_Audit.ipynb` | Inspect dataset structure, data types, missing values, duplicates, and target |
| 2 | `02_Data_Cleaning.ipynb` | Apply justified cleaning and validation rules |
| 3 | `03_EDA.ipynb` | Explore target distribution, feature relationships, and patterns |
| 4 | `04_Feature_Engineering.ipynb` | Create meaningful prediction-time features |
| 5 | `05_Baseline_Model.ipynb` | Establish a benchmark for model comparison |
| 6 | `06_Linear_Regression.ipynb` | Train and interpret a Linear Regression model |
| 7 | `07_Regularization_Evaluation.ipynb` | Compare regularized models using validation and cross-validation |
| 8 | `08_Final_Evaluation.ipynb` | Evaluate generalization and inspect residuals |
| 9 | `09_Optional_Classification_Confusion_Matrix.ipynb` | Demonstrate classification metrics on a separately defined binary target |

## 🧠 Models and Techniques

**Regression models**
- Baseline model
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

**Data preprocessing**
- Missing-value imputation
- Numerical feature scaling
- Categorical feature encoding
- `ColumnTransformer` for column-specific preprocessing
- Scikit-learn `Pipeline` for reproducible model workflows

**Model validation**
- Train/test split
- Cross-validation
- Hyperparameter tuning
- Train/test performance comparison
- Residual analysis

## 🛡️ Data Leakage Prevention

A central design decision in this project is to predict delivery time **when the customer places the order**.

Therefore, `Time_Order_picked` is excluded from the model inputs because pickup occurs after the prediction point. Using this information would allow the model to access information that would not be available in a real deployment scenario.

The preprocessing pipeline is also fitted on training data within the validation workflow, preventing information from validation or test data from influencing learned preprocessing parameters.

These practices help produce a more realistic evaluation of model performance.

## 📏 Evaluation Metrics

The regression models are assessed using the following metrics:

| Metric | What it tells us |
|---|---|
| **MAE** | Average absolute prediction error, measured in minutes |
| **RMSE** | Penalizes larger prediction errors more strongly |
| **R² Score** | Measures predictive performance relative to a mean-target baseline |

Model selection is based on validation results, appropriate test-set evaluation, and the overall generalization gap—not simply on the algorithm used.

## ⚖️ Understanding Regularization

Regularization helps control model complexity by penalizing large coefficients.

- **Linear Regression:** Minimizes the sum of squared prediction errors.
- **Ridge (L2):** Adds a penalty based on squared coefficient magnitudes.
- **Lasso (L1):** Adds a penalty based on absolute coefficient magnitudes and can shrink some coefficients to zero.
- **ElasticNet:** Combines L1 and L2 regularization.

Regularization strength is selected through validation rather than assumed in advance.

## 📊 Regression vs. Classification

The primary task is a **regression problem** because delivery time is a continuous numerical value.

A confusion matrix is not an appropriate primary evaluation tool for this task. The optional classification notebook demonstrates a separate binary classification example by converting the historical target into `Delayed` and `On Time` categories using an explicitly defined threshold.

This optional experiment is separate from the main delivery-time regression workflow.

## 🛠️ Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Version control:** Git and GitHub

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Zomato_Linear_Regression_ML_Project
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the notebooks in sequence

Execute notebooks `01` through `08` in order. Notebook `09` is optional.

Ensure dataset paths match the repository structure and execute earlier notebooks before those that depend on their outputs.

## 📁 Dataset and References

The project uses the uploaded `Zomato Dataset.csv`, containing 45,584 rows and 20 columns.

The following public repositories were consulted as methodology and context references. Their cleaning decisions are not automatically treated as correct for this dataset.

- [Zomato Delivery Time Prediction — Parth Malik](https://github.com/Parth-Malik/Zomato-Delivery-Time-Prediction)
- [Zomato Delivery Analytics — CodeByMiraj](https://github.com/CodeByMiraj/Zomato-Delivery-Analytics)
- [Zomato Food Delivery Analysis — Varunnagar](https://github.com/Varunnagar/Zomato-Food-Delivery-Analysis)
- [Zomato Delivery Analytics — PN-Piyush](https://github.com/PN-Piyush/Zomato-Delivery-Analytics)

Dataset provenance and redistribution permissions should be verified before publishing the CSV publicly.

## 📚 Key Learning Outcomes

Through this project, the intended learning outcomes include:

- Understanding the complete regression modeling lifecycle.
- Performing data auditing and evidence-based cleaning.
- Exploring data through meaningful visualizations.
- Building prediction-time features while avoiding target and temporal leakage.
- Understanding coefficients, regularization, and model evaluation.
- Using pipelines to organize preprocessing and model training.
- Comparing models using appropriate validation metrics.
- Interpreting residuals and evaluating model generalization.

## ⚠️ Project Status and Results

This repository is intended as a practical machine learning learning project. Final performance metrics, model comparisons, and conclusions should be reported only after the notebooks have been executed and the results verified.

No specific model accuracy or best-performing algorithm is claimed in advance.

---

**Author:** Menaga Krishnamurthy

*Built as part of a hands-on journey into Data Science and Machine Learning.*
