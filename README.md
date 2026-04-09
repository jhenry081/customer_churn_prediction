# Telecom Customer Churn Prediction

A machine learning project predicting customer churn for a telecom company.

---

## Dataset

- **Source:** [UCI Telecom Churn Dataset](https://archive.ics.uci.edu/ml/datasets/Iranian+Churn+Dataset) via [Kaggle](https://www.kaggle.com/royjafari/customer-churn)
- **Size:** 3,150 customers, 14 features
- **Target:** `churn` i.e. whether a customer churned over a 12-month period

### Features

| Column | Description |
| --- | --- |
| Call Failure | Number of call failures |
| Complaints | Binary (0: no complaint, 1: complaint) |
| Subscription Length | Total months of subscription |
| Charge Amount | Ordinal (0: lowest, 9: highest) |
| Seconds of Use | Total seconds of calls |
| Frequency of use | Total number of calls |
| Frequency of SMS | Total number of text messages |
| Distinct Called Numbers | Total number of distinct phone numbers called |
| Age Group | Ordinal (1: youngest, 5: oldest) |
| Tariff Plan | Binary (1: pay-as-you-go, 2: contractual) |
| Status | Binary (1: active, 2: non-active) |
| Age | Age of customer |
| Customer Value | Calculated customer value score |
| Churn | Target label (1: churned, 0: retained) |

---

## Project Structure

```text
customer churn prediction/
├── notebook.ipynb  # analysis
├── customer_churn.csv  # dataset
├── requirements.txt # python dependencies
└── README.md
```

---

## Notebook Sections

### 1. Setup & Data Loading
Imports all libraries and loads `customer_churn.csv`. Displays data types, missing values, and summary statistics.

### 2. Exploratory Data Analysis

- Churn distribution (count + pie chart)
- Numerical feature distributions and boxplots split by churn status
- Churn rate per categorical feature (Complaints, Age Group, Tariff Plan, Status, Charge Amount)
- Correlation heatmap

**Challenge analyses:**

- Which age groups send more SMS messages than make phone calls?
- Distinct called numbers by age group, differentiated by call length (short/medium/long)
- Statistical comparison of call duration between tariff plans (Mann-Whitney U test)

### 3. Data Preprocessing

- Encodes `Tariff Plan` (1/2 → 0/1) and `Status` (1/2 → 0/1) as binary flags
- All other features are already numeric i.e. no missing values

### 4. Feature Engineering

| Feature | Description |
| --- | --- |
| `Avg Call Duration` | Seconds of Use / Frequency of use |
| `SMS Ratio` | SMS share of total communications |
| `Failure Rate` | Call failures per call made |
| `Tenure Group` | 0 = new (≤12m), 1 = mid (12–36m), 2 = loyal (>36m) |

### 5. Model Training & Evaluation

- 80/20 stratified train/test split
- **SMOTE** applied to the training set to address class imbalance
- **StandardScaler** applied after SMOTE
- Seven models trained and compared:

| Model | Notes |
| --- | --- |
| Logistic Regression (L1) | L1 penalty for feature selection and churn driver identification |
| Logistic Regression (L2) | Standard ridge-penalised baseline |
| Decision Tree | Interpretable tree, tuned via GridSearchCV |
| K-Nearest Neighbours | k=7 |
| Random Forest | 200 estimators |
| Gradient Boosting | 200 estimators, lr=0.05 |
| XGBoost | 200 estimators, lr=0.05 |

- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Outputs: model comparison chart, ROC curves, confusion matrices, classification report

**Hyperparameter tuning:** `GridSearchCV` over `max_depth`, `min_samples_leaf`, and `criterion` for the Decision Tree, with a depth vs. AUC plot.

### 6. Feature Importance

- Tree-based feature importances (top 15) for the best ensemble model
- L1 Logistic Regression coefficient chart for non-zero coefficients that identifies the true churn drivers (positive = increases risk, negative = decreases risk)

### 7. Summary & Business Insights

Key drivers of churn and six actionable retention recommendations.

---

## Key Findings

| Driver | Insight |
| --- | --- |
| **Complaints** | Strongest single churn signal i.e. complaint filers churn at far higher rates |
| **Status** | Non-active customers almost always churn |
| **Subscription Length** | Shorter tenure correlates with higher churn; loyal customers rarely leave |
| **Call Failure Rate** | High failure rates frustrate customers and drive churn |
| **Age Group** | Younger customers (Group 1) churn the most |
| **Tariff Plan** | Pay-as-you-go customers churn more than contractual customers |

---

## Setup & Usage

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## Dependencies

| Package | Version |
| --- | --- |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.5.0 |
| xgboost | 2.0.3 |
| imbalanced-learn | 0.12.3 |
| jupyter | 1.0.0 |
