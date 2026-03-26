# Walmart Sales Forecasting - Time Series Pipeline

## Overview
End-to-end machine learning pipeline to forecast weekly sales across 45 Walmart stores and 99 departments, based on 408,000+ historical transactions (2010-2012).

## Objectives
1. Predict weekly sales for each store and department
2. Identify factors influencing sales (promotions, holidays, economic variables)
3. Support inventory management and purchase planning decisions

## Methodology
1. **Data Preparation** - Merging stores, features and sales datasets, handling missing values
2. **Feature Engineering** - Lag variables (Lag_1, Lag_2, Lag_4), rolling means (Rolling_Mean_4), seasonal flags (Black_Friday, Thanksgiving, Xmas_Week), Target Encoding without data leakage
3. **Validation** - TimeSeriesSplit (5 folds) to respect chronological order
4. **Custom Metric** - Weighted MAE (WMAE) with x5 weight on holiday periods
5. **Phase 1** - Comparison of 3 target variants (raw, log transformation, imputation + flag) using Random Forest - avg R²: 0.94
6. **Phase 2 (in progress)** - Comparison of 4 algorithms: Random Forest, XGBoost, LightGBM, Linear Regression + hyperparameter optimization

## Key Results
- Best R² avg: **0.94** on validation set
- Feature importance analysis for model interpretability
- WMAE custom metric adapted to Kaggle competition standards

## Tools & Libraries
- Python, pandas, NumPy
- scikit-learn (RandomForestRegressor, TimeSeriesSplit)
- XGBoost, LightGBM
- Matplotlib

## Dataset
[Kaggle - Walmart Store Sales Forecasting](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)
