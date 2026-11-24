# Used Car Price Prediction

> Machine Learning coursework, The British University in Egypt. Team of 3. I implemented the data-cleaning, feature-engineering and model-tuning pipeline in this notebook.

Predicting the resale price of used cars from listing data, with a lot of feature engineering on messy free-text fields.

## Data
- 19,109 listings × 36 columns (a Kaggle used-car listings dataset; `train.csv` / `test.csv`, not included).
- After cleaning: 18,984 training rows and 4,749 test rows, with **77 features**.
- The price came as text (`"$1,000"`, `"Not Priced"`) and was parsed to numbers.

## Feature engineering
- **Target:** log-transformed price (the raw distribution is right-skewed). Listings under $1K or over $200K were removed.
- **Missing values:** numeric columns get the training median; categorical columns get `"Unknown"`.
- **Outliers:** capped at 1.5 × IQR.
- **Parsed from the engine text:** engine size, number of cylinders, turbo flag.
- **Colours:** reduced to ~20 base colours, plus an `is_metallic` flag.
- **Transmission:** type (Automatic, CVT, Dual-Clutch, Manual, Other) and number of gears.
- **Feature selection:** dropped features with negligible correlation to price. The strongest signals were model, exterior colour, year, mileage, navigation system, memory seats and power liftgate.

## Results (test set)

| Model | RMSE (log) | R² (log) | RMSE ($) | R² ($) |
|---|---|---|---|---|
| **XGBoost (tuned)** | **0.191** | **0.900** | **$8,645** | **0.858** |
| Gradient Boosting | 0.214 | 0.874 | $10,594 | 0.787 |
| Random Forest | 0.243 | 0.838 | $12,964 | 0.681 |
| KNN | 0.249 | 0.829 | $12,811 | 0.689 |
| Decision Tree | 0.258 | 0.816 | $11,919 | 0.731 |

XGBoost was tuned with randomized search over 50 configurations and 5-fold cross-validation.

## Known issues (found in review)
- Scaling was applied to the target-encoded **brand** and **model** columns in a way that made them effectively constant, so the two strongest predictors weren't used.
- IQR capping turned some binary flags (damaged, Bluetooth, alloy wheels) into constants.

Fixing both should improve on the current scores. They're documented here rather than hidden.

## Run it
```bash
pip install -r requirements.txt
# place train.csv and test.csv next to the notebook
jupyter notebook notebooks/used_car_price_prediction.ipynb
```

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · category_encoders · matplotlib · seaborn
