# E-Commerce Behavior Analysis Projects

A collection of machine learning projects built on real e-commerce behavioral data, covering recommender systems, purchase prediction, and user behavior analysis.

**Dataset:** [eCommerce Behavior Data from Multi Category Store](https://www.kaggle.com/code/varshnidevi/ecommerce-behavior-data-from-multi-category-store/notebook) — 1M+ user interaction records (Nov 2019)

**Key Fields:** `event_time` · `event_type` · `product_id` · `user_id` · `category_code` · `brand` · `price` · `user_session`

---

## Projects Overview

| # | Project | Method | Status |
|---|---------|--------|--------|
| 1 | Recommender System | Item-based Collaborative Filtering | ✅ Complete |
| 2 | Purchase Prediction | Logistic Regression + Time Split | ✅ Complete |

---

## 1. Recommender System

### Goal
Build a product recommendation system based on user interaction history, recommending items the user is likely to be interested in.

### Method — Item-based Collaborative Filtering

1. Map `event_type` to interaction strength (`event_score`)
   - view = 1 · cart = 2 · purchase = 3
2. Build a `product_id × user_id` interaction matrix
3. Compute item-to-item similarity using cosine similarity
4. Aggregate similarity scores based on user history
5. Exclude already-interacted items and return Top-N recommendations

### Train / Test Split
Time-based split to prevent data leakage:
- **Train:** earlier interactions
- **Test:** later interactions (purchase events only, as ground truth)

### Results

![User Behavior Funnel](images/funnel_chart.png)

![Precision Recall F1 by K](images/Precision_recall_f1.png)

| K | Precision@K | Recall@K | F1@K |
|---|-------------|----------|------|
| 3  | 0.049 | 0.140 | 0.073 |
| 10 | 0.028 | 0.267 | 0.051 |
| 20 | 0.019 | 0.359 | 0.036 |

### Key Learnings
- Item-based Collaborative Filtering generates recommendations via item similarity scores
- Cosine similarity is well-suited for sparse interaction matrices
- As K increases, Recall rises but Precision drops — K=3 achieved the best F1 balance
- Recommendation evaluation focuses not just on model building, but on hit rate validation

---

## 2. Purchase Prediction

### Goal
Using past behavioral data to predict whether a user will purchase a specific product in the future.

### Problem Setting
- **Features (X):** built from past interactions (view count, cart count)
- **Label (y):** whether the user made a purchase in the future window
- **Split:** time-based to avoid look-ahead bias

### Class Imbalance

Label distribution is highly imbalanced — non-purchase events far outnumber purchases.

![Label Distribution](images/Label_Distribution.png)

### Model — Logistic Regression (Baseline)

Applied `class_weight="balanced"` to handle imbalance, and StandardScaler for normalization.

### Results

![Confusion Matrix](images/confusion_matrix.png)

![Confusion Matrix Normalized](images/confusion_matrix%20values%25.png)

| Metric | Score |
|--------|-------|
| Precision | 0.06 |
| Recall | 0.22 |
| F1-score | 0.09 |

### Threshold Comparison

| Threshold | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| 0.3 | 0.005 | 1.000 | 0.010 |
| 0.5 | 0.026 | 0.584 | 0.049 |
| 0.7 | 0.038 | 0.363 | 0.069 |
| 0.9 | 0.056 | 0.217 | 0.089 |

### Before vs After Standardization

| Setting | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Without StandardScaler | 0.056 | 0.217 | 0.089 |
| With StandardScaler | 0.060 | 0.220 | 0.090 |

### Key Learnings
- Time-based splitting is essential to prevent data leakage in sequential data
- Class imbalance significantly impacts model performance — accuracy alone is misleading
- Threshold tuning directly controls the Precision / Recall trade-off
- FN (missed buyers) is the most costly error in a purchase prediction context
- StandardScaler provides marginal improvement for Logistic Regression

### Conclusion
This project established a purchase prediction pipeline using time-based splitting and Logistic Regression as a baseline. Results confirm that under severe class imbalance, threshold adjustment and feature standardization improve performance. Next steps include feature engineering (session-level aggregation, recency features) and resampling techniques (SMOTE) to further improve recall on the minority class.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-lightgrey)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-green)
![Seaborn](https://img.shields.io/badge/Seaborn-teal)

```
pandas · numpy · scikit-learn · matplotlib · seaborn
```

## File Structure

```
├── recommender_system.ipynb        # Item-based CF recommender
├── recommender_system_practice.ipynb  # Practice notebook
├── Purchase Prediction.ipynb       # Purchase prediction model
├── images/                         # Visualization outputs
│   ├── funnel_chart.png
│   ├── Precision_recall_f1.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix values%.png
│   └── Label_Distribution.png
├── requirements.txt
└── README.md
```
