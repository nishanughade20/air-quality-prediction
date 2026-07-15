# Air Quality (CO Level) Prediction

## Problem
Predict Carbon Monoxide (CO) concentration in air using sensor readings for other pollutants and weather conditions (temperature, humidity). Built as a regression project to complement a prior classification project (breast cancer prediction).

## Dataset
UCI Air Quality Dataset — 9,357 hourly readings from a gas multisensor device deployed in an Italian city, including CO, NOx, NO2, Benzene (C6H6), temperature, and humidity.

## Approach
1. **Data cleaning**: Replaced the dataset's `-200` missing-value placeholder with `NaN`, dropped the `NMHC(GT)` column (90% missing), removed fully empty rows, and imputed remaining gaps with median values.
2. **Modeling**: Trained and compared two regressors:
   - Linear Regression (baseline)
   - Random Forest Regressor (100 trees)
3. **Evaluation**: RMSE, MAE, and R² on a held-out 20% test set.

## Results
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.549 | 0.375 | 0.835 |
| Random Forest | 0.453 | 0.283 | **0.888** |

Random Forest outperformed the linear baseline on all three metrics, capturing non-linear relationships between pollutant sensors that a linear model misses.

## Feature Importance
The most influential predictors of CO levels were the correlated pollutant sensor readings (see `models/feature_importance.png`), which aligns with the physical reality that CO co-occurs with other combustion byproducts.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn

## Next Steps
- Hyperparameter tuning (GridSearchCV) on Random Forest
- Try gradient boosting (XGBoost/LightGBM)
- Time-based train/test split to better reflect real-world deployment