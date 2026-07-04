# Notes Techniques - Walmart Weekly Sales Forecasting
# Technical Notes - Walmart Weekly Sales Forecasting

---

## FR - Français

### 1. Dataset

3 fichiers sources :
- train.csv : 421 570 lignes (ventes hebdomadaires par magasin et département)
- stores.csv : 45 magasins (Type A, B, C et taille)
- features.csv : 8 190 lignes (CPI, chômage, température, MarkDowns, IsHoliday)

Après merge et nettoyage : 407 117 lignes x 33 colonnes

Chaque ligne représente une combinaison unique magasin / département / semaine.


### 2. Problème des valeurs négatives de Weekly_Sales

1 285 valeurs négatives identifiées, soit 0.30% du dataset.
Ces valeurs représentent des retours nets (retours clients supérieurs aux ventes).
Elles concernent 45 magasins et 50 départements sur 81.

3 variantes ont été évaluées avec Random Forest (TimeSeriesSplit, 5 folds) :

Variante 1 : Conserver les valeurs négatives telles quelles.
Résultat : pas de transformation log possible, distribution non corrigée.

Variante 2 : Supprimer les lignes négatives et appliquer une transformation log.
Résultat : R2 moyen de 0.94, meilleur des 3 variantes. Variante retenue.

Variante 3 : Imputer à 0 et créer une colonne flag Has_Returns.
Résultat : moins performant que la Variante 2.

Justification de la Variante 2 :
La distribution des ventes présente une forte asymétrie positive (valeurs jusqu'à 700 000$ pour les magasins Type A).
La transformation log corrige cette hétéroscédasticité et améliore les performances du modèle.
Les 0.30% de lignes supprimées ont un impact marginal sur la représentativité du dataset.

Note : MarkDown2 et MarkDown3 contenaient également des valeurs négatives, corrigées avec clip(lower=0).

Formulation pour entretien :
"Nous avons évalué trois approches pour gérer les valeurs négatives. La transformation log était souhaitable
pour corriger la forte asymétrie de la distribution des ventes. La Variante 2, supprimer les 0.30% de lignes
négatives et appliquer le log, a produit les meilleures performances avec un R2 moyen de 0.94 et se justifie
statistiquement par la correction de l'hétéroscédasticité."


### 3. Analyse exploratoire (EDA)

5 analyses clés réalisées dans le notebook 01.

A1 - Distribution des ventes :
- Forte asymétrie positive, nécessite une transformation log
- Valeurs extrêmes jusqu'à 700 000$ pour les magasins Type A

A2 - Promotions (MarkDowns) :
- MarkDown2 et MarkDown3 contiennent des valeurs négatives, corrigées avec clip(lower=0)
- Impact modéré des promotions sur les ventes

A3 - Variables économiques (CPI, chômage) :
- Faible corrélation linéaire avec Pearson, car les relations sont catégorielles ou non linéaires
- Impact indirect via la localisation des magasins

A4 - Type de magasin et IsHoliday :
- Magasins Type A : ventes moyennes élevées mais forte asymétrie
- IsHoliday très déséquilibré : environ 93% semaines normales vs 7% semaines de fêtes
- Corrélation Pearson de IsHoliday avec les ventes : 0.01 (relation catégorielle, Pearson ne la capte pas)

A5 - Analyse temporelle :
- 143 semaines consécutives, aucun trou dans la série
- Découverte clé sur les fêtes :
  Xmas_Week (semaine du 24 décembre) : pic numéro 1 à 27 000$ par semaine, mais IsHoliday = False
  Thanksgiving : pic numéro 2 à 22 300$ par semaine, seule vraie fête officielle avec impact significatif
  Black_Friday : tendance haussière post-Thanksgiving à 16 800$ par semaine, mais IsHoliday = False
  SuperBowl, LaborDay, NewYear : IsHoliday = True mais niveau de semaines ordinaires
- Conclusion : IsHoliday est insuffisant pour capter les vrais pics commerciaux
  3 colonnes dédiées ont été créées : Thanksgiving, Black_Friday, Xmas_Week


### 4. Ingénierie des variables (Feature Engineering)

30 variables finales construites.

Variables temporelles :
- Year, Month, WeekOfYear

Encodage cyclique :
- Week_sin = sin(2 x pi x WeekOfYear / 52)
- Week_cos = cos(2 x pi x WeekOfYear / 52)

Justification : Un entier WeekOfYear crée une rupture artificielle entre la semaine 52 et la semaine 1.
L'encodage sin/cos crée une continuité qui reflète la vraie saisonnalité annuelle et permet au modèle
de comprendre que la semaine 52 est proche de la semaine 1.

Formulation pour entretien :
"WeekOfYear encodé en entier implique que la distance entre la semaine 52 et la semaine 1 est de 51,
alors qu'elles sont adjacentes dans le calendrier. L'encodage circulaire sin/cos corrige cela en
représentant la semaine sur un cercle trigonométrique."

Lags (décalages temporels) :
- Lag_1 : ventes de la semaine précédente
- Lag_2 : ventes de 2 semaines avant
- Lag_4 : ventes de 4 semaines avant

Justification : Les ventes d'une semaine sont corrélées aux ventes des semaines précédentes
(inertie commerciale, niveaux de stocks, comportements d'achat). Les lags permettent au modèle
de voir l'historique récent sans introduire de fuite de données (data leakage).

Rolling features (statistiques glissantes) :
- Rolling_Mean_4 : moyenne glissante sur 4 semaines
- Rolling_Std_4 : écart-type glissant sur 4 semaines

Justification : Capturent la tendance locale (Rolling_Mean_4) et la volatilité locale (Rolling_Std_4),
complémentaires aux lags ponctuels.

Fêtes dédiées :
- Thanksgiving : booléen, semaines de Thanksgiving
- Black_Friday : booléen, semaine suivant Thanksgiving
- Xmas_Week : booléen, semaine du 24 décembre

Justification : IsHoliday ne capte pas les vrais pics commerciaux (Xmas_Week et Black_Friday
ont IsHoliday = False). Ces 3 colonnes dédiées permettent au modèle de distinguer
les semaines à fort impact commercial.

Encodage du type de magasin :
- One-hot encoding : Type_A, Type_B, Type_C
- Is_Promo : variable binaire dérivée des MarkDowns

Target encoding :
- Store_enc : moyenne des Weekly_Sales par magasin, calculée sur le train uniquement
- Dept_enc : moyenne des Weekly_Sales par département, calculée sur le train uniquement

Important : le target encoding est calculé uniquement sur les données d'entraînement de chaque fold
pour éviter toute fuite de données vers le set de validation.


### 5. Métrique : WMAE

WMAE = Weighted Mean Absolute Error

Les semaines de fêtes (IsHoliday = True) sont pondérées x5 dans le calcul de l'erreur.

Justification : IsHoliday est très déséquilibré (environ 7% des semaines).
Sans pondération, le modèle pourrait ignorer les fêtes et obtenir un bon score moyen
tout en étant très mauvais sur les semaines commercialement critiques.
La pondération x5 force le modèle à bien prédire les semaines de fêtes.

Formulation pour entretien :
"La métrique WMAE penalise davantage les erreurs sur les semaines de fêtes, qui représentent
seulement 7% des données mais sont les plus importantes pour Walmart. Sans cette pondération,
un modèle pourrait obtenir un bon score global en ignorant complètement ces semaines critiques."


### 6. Stratégie de validation

Deux approches utilisées selon la version du projet.

Version intermédiaire (notebook 01) : TimeSeriesSplit avec 5 folds
- Chaque fold utilise uniquement le passé pour prédire le futur (respect de la chronologie)
- 5 estimations moyennées : variance de l'estimation plus faible, estimation plus robuste
- Résultats RF V2 : Fold 1 WMAE 4 327$, Fold 2 WMAE 1 494$, Fold 3 WMAE 2 456$,
  Fold 4 WMAE 1 848$, Fold 5 WMAE 1 319$
- Résultats XGBoost V2 : Fold 1 WMAE 4 511$, Fold 2 WMAE 1 493$, Fold 3 WMAE 2 098$,
  Fold 4 WMAE 1 823$, Fold 5 WMAE 1 347$

Version finale (notebook 03) : Coupure temporelle stricte
- Train : 2010-2011 (280 257 lignes)
- Test : 2012 (126 860 lignes)
- Une seule estimation mais plus réaliste pour le déploiement (on prédit le futur à partir du passé)
- Permet une comparaison propre et équitable entre tous les modèles sur le même test set

Les deux approches sont valides pour les séries temporelles.
TimeSeriesSplit offre une estimation statistiquement plus robuste.
La coupure temporelle est plus proche du scénario de déploiement réel.

Formulation pour entretien :
"Nous avons utilisé une coupure temporelle stricte pour le modèle final car c'est la configuration
la plus proche du déploiement réel. Dans la phase exploratoire, nous avions aussi évalué avec
TimeSeriesSplit sur 5 folds, qui donnait des estimations similaires mais avec une variance plus faible.
Les deux approches respectent la chronologie des données et sont valides pour les séries temporelles."


### 7. Comparaison des modèles

5 variantes comparées dans le notebook 02.

V0 - Random Forest (baseline) :
- Features de base, pas de transformation log
- WMAE : 4 249$, R2 : 0.82

V2 - Random Forest :
- Feature engineering complet + transformation log
- Amélioration significative vs V0 : -46.1% sur le WMAE

V2 - XGBoost :
- Même feature engineering que RF V2, changement d'algorithme
- Légère amélioration supplémentaire vs RF V2

V2 - LightGBM Optuna :
- Sélection des variables par SHAP sur le Fold 5
- Optimisation des hyperparamètres par Optuna
- WMAE : 2 095$

VF - XGBoost + sin/cos (modèle final) :
- Ajout de l'encodage cyclique Week_sin et Week_cos
- Validation par coupure temporelle 2012
- WMAE : 1 591$, R2 : 0.97

Progression globale :
- WMAE baseline : 4 249$ vers WMAE final : 1 591$, soit -62.6%
- R2 baseline : 0.82 vers R2 final : 0.97, soit +18.9%


### 8. Modèle final

Algorithme : XGBoost
Encodage temporel : Week_sin et Week_cos (encodage cyclique)
Validation : coupure temporelle 2012
WMAE final : 1 591$
R2 : 0.97

Le modèle explique 97% de la variance des ventes hebdomadaires de Walmart.
L'encodage sin/cos de la semaine est le facteur clé de l'amélioration finale par rapport à LightGBM Optuna.


### 9. Structure du repo et chronologie des fichiers

01_EDA_FeatureEngineering_Exploration.ipynb
- Contenu : merge des 3 datasets, EDA (5 analyses), gestion des valeurs négatives (3 variantes),
  feature engineering (30 variables), TimeSeriesSplit 5 folds sur RF V2 et XGBoost V2,
  sélection des variables par SHAP
- Correspond à la version intermédiaire du projet

02_Model_Comparison.ipynb
- Contenu : comparaison des 5 variantes de modèles sur les 4 métriques (WMAE, MAE, RMSE, R2),
  visualisation de la progression, sélection du meilleur modèle
- Point de jonction entre l'exploration et le modèle final

03_Final_XGBoost_Model.ipynb
- Contenu : pipeline final avec XGBoost + sin/cos, coupure temporelle 2012,
  résultats finaux, courbe d'apprentissage
- Correspond à la version finale du projet

---

## EN - English

### 1. Dataset

3 source files :
- train.csv : 421,570 rows (weekly sales by store and department)
- stores.csv : 45 stores (Type A, B, C and size)
- features.csv : 8,190 rows (CPI, unemployment, temperature, MarkDowns, IsHoliday)

After merge and cleaning : 407,117 rows x 33 columns

Each row represents a unique store / department / week combination.


### 2. Handling Negative Weekly_Sales Values

1,285 negative values identified, representing 0.30% of the dataset.
These values represent net returns (customer returns exceeding sales).
They concern 45 stores and 50 out of 81 departments.

3 variants were evaluated using Random Forest (TimeSeriesSplit, 5 folds) :

Variant 1 : Keep negative values as is.
Result : log transformation not possible, distribution not corrected.

Variant 2 : Remove negative rows and apply log transformation.
Result : mean R2 of 0.94, best of the 3 variants. Selected variant.

Variant 3 : Impute to 0 and create a Has_Returns flag column.
Result : lower performance than Variant 2.

Justification for Variant 2 :
The sales distribution shows strong positive skewness (values up to 700,000$ for Type A stores).
The log transformation corrects this heteroscedasticity and improves model performance.
The 0.30% of removed rows have a marginal impact on dataset representativeness.

Note : MarkDown2 and MarkDown3 also contained negative values, corrected with clip(lower=0).

Interview formulation :
"We evaluated three approaches to handle negative values. The log transformation was desirable
to correct the strong skewness of the sales distribution. Variant 2, removing the 0.30% of negative
rows and applying the log, produced the best performance with a mean R2 of 0.94 and is statistically
justified by the correction of heteroscedasticity."


### 3. Exploratory Data Analysis (EDA)

5 key analyses performed in notebook 01.

A1 - Sales distribution :
- Strong positive skewness, requires log transformation
- Extreme values up to 700,000$ for Type A stores

A2 - Promotions (MarkDowns) :
- MarkDown2 and MarkDown3 contain negative values, corrected with clip(lower=0)
- Moderate impact of promotions on sales

A3 - Economic variables (CPI, unemployment) :
- Low linear correlation with Pearson, because relationships are categorical or non-linear
- Indirect impact via store location

A4 - Store type and IsHoliday :
- Type A stores : high average sales but strong skewness
- IsHoliday heavily imbalanced : approximately 93% normal weeks vs 7% holiday weeks
- Pearson correlation of IsHoliday with sales : 0.01 (categorical relationship, Pearson does not capture it)

A5 - Temporal analysis :
- 143 consecutive weeks, no gaps in the series
- Key discovery on holidays :
  Xmas_Week (week of December 24) : peak number 1 at 27,000$ per week, but IsHoliday = False
  Thanksgiving : peak number 2 at 22,300$ per week, only true official holiday with significant impact
  Black_Friday : upward trend post-Thanksgiving at 16,800$ per week, but IsHoliday = False
  SuperBowl, LaborDay, NewYear : IsHoliday = True but ordinary week level
- Conclusion : IsHoliday is insufficient to capture true commercial peaks
  3 dedicated columns were created : Thanksgiving, Black_Friday, Xmas_Week


### 4. Feature Engineering

30 final features constructed.

Temporal features :
- Year, Month, WeekOfYear

Cyclical encoding :
- Week_sin = sin(2 x pi x WeekOfYear / 52)
- Week_cos = cos(2 x pi x WeekOfYear / 52)

Justification : An integer WeekOfYear creates an artificial gap between week 52 and week 1.
The sin/cos encoding creates continuity that reflects true annual seasonality and allows the model
to understand that week 52 is close to week 1.

Interview formulation :
"WeekOfYear encoded as an integer implies that the distance between week 52 and week 1 is 51,
whereas they are adjacent in the calendar. Circular sin/cos encoding corrects this by representing
the week on a trigonometric circle."

Lag features :
- Lag_1 : sales from the previous week
- Lag_2 : sales from 2 weeks ago
- Lag_4 : sales from 4 weeks ago

Justification : Sales of a given week are correlated with sales of previous weeks
(commercial inertia, stock levels, purchasing behaviors). Lags allow the model
to see recent history without introducing data leakage.

Rolling features :
- Rolling_Mean_4 : 4-week rolling average
- Rolling_Std_4 : 4-week rolling standard deviation

Justification : Capture local trend (Rolling_Mean_4) and local volatility (Rolling_Std_4),
complementary to point-in-time lags.

Dedicated holiday features :
- Thanksgiving : boolean, Thanksgiving weeks
- Black_Friday : boolean, week following Thanksgiving
- Xmas_Week : boolean, week of December 24

Justification : IsHoliday does not capture true commercial peaks (Xmas_Week and Black_Friday
have IsHoliday = False). These 3 dedicated columns allow the model to distinguish
weeks with high commercial impact.

Store type encoding :
- One-hot encoding : Type_A, Type_B, Type_C
- Is_Promo : binary variable derived from MarkDowns

Target encoding :
- Store_enc : mean of Weekly_Sales per store, computed on train set only
- Dept_enc : mean of Weekly_Sales per department, computed on train set only

Important : target encoding is computed only on training data of each fold
to avoid any data leakage into the validation set.


### 5. Metric : WMAE

WMAE = Weighted Mean Absolute Error

Holiday weeks (IsHoliday = True) are weighted x5 in the error calculation.

Justification : IsHoliday is heavily imbalanced (approximately 7% of weeks).
Without weighting, the model could ignore holidays and achieve a good average score
while performing very poorly on commercially critical weeks.
The x5 weighting forces the model to predict holiday weeks accurately.

Interview formulation :
"The WMAE metric penalizes errors on holiday weeks more heavily. These weeks represent
only 7% of the data but are the most important for Walmart. Without this weighting,
a model could achieve a good overall score by completely ignoring these critical weeks."


### 6. Validation Strategy

Two approaches used depending on the project version.

Intermediate version (notebook 01) : TimeSeriesSplit with 5 folds
- Each fold uses only past data to predict the future (chronological order respected)
- 5 averaged estimates : lower estimation variance, more robust estimation
- RF V2 results : Fold 1 WMAE 4,327$, Fold 2 WMAE 1,494$, Fold 3 WMAE 2,456$,
  Fold 4 WMAE 1,848$, Fold 5 WMAE 1,319$
- XGBoost V2 results : Fold 1 WMAE 4,511$, Fold 2 WMAE 1,493$, Fold 3 WMAE 2,098$,
  Fold 4 WMAE 1,823$, Fold 5 WMAE 1,347$

Final version (notebook 03) : Strict temporal split
- Train : 2010-2011 (280,257 rows)
- Test : 2012 (126,860 rows)
- Single estimate but more realistic for deployment (predicting the future from the past)
- Allows fair and consistent comparison of all models on the same test set

Both approaches are valid for time series.
TimeSeriesSplit provides a statistically more robust estimate.
The temporal split is closer to the real deployment scenario.

Interview formulation :
"We used a strict temporal split for the final model as it is the closest configuration to real deployment.
In the exploratory phase, we also evaluated with TimeSeriesSplit on 5 folds, which gave similar estimates
but with lower variance. Both approaches respect the chronological order of the data and are valid
for time series."


### 7. Model Comparison

5 variants compared in notebook 02.

V0 - Random Forest (baseline) :
- Basic features, no log transformation
- WMAE : 4,249$, R2 : 0.82

V2 - Random Forest :
- Full feature engineering + log transformation
- Significant improvement vs V0 : -46.1% on WMAE

V2 - XGBoost :
- Same feature engineering as RF V2, algorithm change
- Slight additional improvement vs RF V2

V2 - LightGBM Optuna :
- Variable selection using SHAP on Fold 5
- Hyperparameter optimization using Optuna
- WMAE : 2,095$

VF - XGBoost + sin/cos (final model) :
- Addition of cyclical encoding Week_sin and Week_cos
- Temporal split validation on 2012
- WMAE : 1,591$, R2 : 0.97

Overall progression :
- Baseline WMAE : 4,249$ to final WMAE : 1,591$, -62.6% improvement
- Baseline R2 : 0.82 to final R2 : 0.97, +18.9% improvement


### 8. Final Model

Algorithm : XGBoost
Temporal encoding : Week_sin and Week_cos (cyclical encoding)
Validation : temporal split on 2012
Final WMAE : 1,591$
R2 : 0.97

The model explains 97% of the variance of Walmart weekly sales.
The sin/cos encoding of the week is the key factor in the final improvement over LightGBM Optuna.


### 9. Repository Structure and File Chronology

01_EDA_FeatureEngineering_Exploration.ipynb
- Content : merge of 3 datasets, EDA (5 analyses), handling of negative values (3 variants),
  feature engineering (30 variables), TimeSeriesSplit 5 folds on RF V2 and XGBoost V2,
  variable selection using SHAP
- Corresponds to the intermediate version of the project

02_Model_Comparison.ipynb
- Content : comparison of 5 model variants on 4 metrics (WMAE, MAE, RMSE, R2),
  progression visualization, selection of the best model
- Bridge between exploration and final model

03_Final_XGBoost_Model.ipynb
- Content : final pipeline with XGBoost + sin/cos, temporal split 2012,
  final results, learning curve
- Corresponds to the final version of the project
