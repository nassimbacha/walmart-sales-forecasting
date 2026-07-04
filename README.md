# Walmart Weekly Sales Forecasting
# Prévision des ventes hebdomadaires Walmart

---

## FR - Français

### Description

Ce projet met en oeuvre un pipeline complet de prévision des ventes hebdomadaires pour 45 magasins Walmart et 81 départements. Il couvre l'exploration des données, l'ingénierie des variables, la comparaison de plusieurs algorithmes et la sélection du meilleur modèle.

### Dataset

Source : Kaggle Walmart Store Sales Forecasting

3 fichiers sources :
- train.csv : 421 570 lignes de ventes hebdomadaires par magasin et département
- stores.csv : 45 magasins avec type (A, B, C) et taille
- features.csv : variables économiques (CPI, chômage, température, promotions, IsHoliday)

Après merge et nettoyage : 407 117 lignes x 33 colonnes

### Pipeline

Etape 1 - Exploration et ingénierie des variables (notebook 01)
- Analyse exploratoire (distribution des ventes, saisonnalité, fêtes)
- Gestion des valeurs négatives de Weekly_Sales (3 variantes comparées)
- Construction de 30 variables : temporelles, cycliques (sin/cos), lags, rolling, fêtes dédiées, target encoding
- Validation croisée temporelle (TimeSeriesSplit, 5 folds) sur Random Forest et XGBoost

Etape 2 - Comparaison des modèles (notebook 02)
- 5 variantes comparées : Random Forest V0, Random Forest V2, XGBoost V2, LightGBM Optuna, XGBoost + sin/cos
- Métrique principale : WMAE (Weighted Mean Absolute Error, pondération x5 sur les semaines de fêtes)
- Progression : WMAE 4 249$ (baseline) vers 1 591$ (modèle final), soit -62.6%

Etape 3 - Modèle final (notebook 03)
- XGBoost avec encodage circulaire de la semaine (Week_sin, Week_cos)
- Validation par coupure temporelle : train 2010-2011, test 2012
- WMAE final : 1 591$, R2 : 0.97

### Résultat

Modèle retenu : XGBoost avec encodage cyclique sin/cos
WMAE : 1 591$
R2 : 0.97 (le modèle explique 97% de la variance des ventes hebdomadaires)
Amélioration vs baseline : -62.6% sur le WMAE

### Structure du repo

- 01_EDA_FeatureEngineering_Exploration.ipynb : exploration, ingénierie des variables, TimeSeriesSplit
- 02_Model_Comparison.ipynb : comparaison des 5 variantes de modèles
- 03_Final_XGBoost_Model.ipynb : pipeline final, validation, résultats
- README.md : présentation du projet
- NOTES_TECHNIQUES.md : documentation technique détaillée

### Outils et technologies

Python, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, pandas, matplotlib

---

## EN - English

### Description

This project implements a complete weekly sales forecasting pipeline for 45 Walmart stores and 81 departments. It covers data exploration, feature engineering, multi-algorithm comparison and final model selection.

### Dataset

Source : Kaggle Walmart Store Sales Forecasting

3 source files :
- train.csv : 421,570 rows of weekly sales by store and department
- stores.csv : 45 stores with type (A, B, C) and size
- features.csv : economic variables (CPI, unemployment, temperature, markdowns, IsHoliday)

After merge and cleaning : 407,117 rows x 33 columns

### Pipeline

Step 1 - Exploration and feature engineering (notebook 01)
- Exploratory data analysis (sales distribution, seasonality, holidays)
- Handling of negative Weekly_Sales values (3 variants compared)
- Construction of 30 features : temporal, cyclical (sin/cos), lags, rolling statistics, dedicated holiday flags, target encoding
- Time series cross-validation (TimeSeriesSplit, 5 folds) on Random Forest and XGBoost

Step 2 - Model comparison (notebook 02)
- 5 variants compared : Random Forest V0, Random Forest V2, XGBoost V2, LightGBM Optuna, XGBoost + sin/cos
- Main metric : WMAE (Weighted Mean Absolute Error, x5 weight on holiday weeks)
- Progression : WMAE 4,249$ (baseline) to 1,591$ (final model), -62.6% improvement

Step 3 - Final model (notebook 03)
- XGBoost with circular week encoding (Week_sin, Week_cos)
- Temporal split validation : train 2010-2011, test 2012
- Final WMAE : 1,591$, R2 : 0.97

### Results

Selected model : XGBoost with sin/cos cyclical encoding
WMAE : 1,591$
R2 : 0.97 (the model explains 97% of weekly sales variance)
Improvement vs baseline : -62.6% on WMAE

### Repository Structure

- 01_EDA_FeatureEngineering_Exploration.ipynb : exploration, feature engineering, TimeSeriesSplit
- 02_Model_Comparison.ipynb : comparison of 5 model variants
- 03_Final_XGBoost_Model.ipynb : final pipeline, validation, results
- README.md : project overview
- NOTES_TECHNIQUES.md : detailed technical documentation

### Tools and Technologies

Python, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, pandas, matplotlib
