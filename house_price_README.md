# 🏠 House Price Prediction

## Overview
End-to-end regression pipeline predicting house sale prices using
feature engineering, log-transformation, and ensemble models.

---

## Repository Structure
```
house-price-prediction/
├── House_Price_Prediction_Notebook.ipynb   # Full notebook
├── house_price_best_model.joblib           # Saved best model bundle
├── inference.py                            # Standalone CLI predictor
├── data.csv                                # Dataset (4600 rows)
├── README.md
└── plots/
    ├── 01_price_dist.png      06_boxplots.png
    ├── 02_corr.png            07_model_compare.png
    ├── 03_scatter.png         08_actual_vs_pred.png
    ├── 04_city_price.png      09_residuals.png
    ├── 05_sqft_price.png      10_feat_importance.png
```

---

## Quick Start

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib scipy
```

### Run demo prediction
```bash
python inference.py
```

### Interactive input
```bash
python inference.py --interactive
```

### Python API
```python
import numpy as np
from inference import predict_price

result = predict_price({
    "log_sqft_living": np.log1p(1800),
    "log_sqft_lot":    np.log1p(5000),
    "bedrooms": 3, "bathrooms": 2.0, "floors": 1.0,
    "waterfront": 0, "view": 0, "condition": 3,
    "sqft_above": 1800, "sqft_basement": 0,
    "house_age": 34, "was_renovated": 0, "years_since_reno": 0,
    "bath_bed_ratio": 0.5, "living_lot_ratio": 0.36,
    "basement_ratio": 0.0, "city_enc": 12, "sale_month": 6,
})
print(f"Predicted: ${result['predicted_price']:,.0f}")
```

---

## Model Results

| Model              | RMSE        | MAE         | R²     | CV R²  |
|--------------------|-------------|-------------|--------|--------|
| Linear Regression  | $390,501    | $166,165    | 0.014  | 0.560  |
| Ridge Regression   | $390,975    | $166,239    | 0.012  | 0.560  |
| Random Forest      | $236,919    | $135,035    | 0.637  | 0.651  |
| **Gradient Boosting** | **$225,975** | **$117,114** | **0.670** | **0.682** |

**Best model saved:** Gradient Boosting

### Key Findings
- Log-transforming `price` and `sqft_living` significantly improves model stability
- `log_sqft_living`, `city_enc`, `view`, `waterfront` are top predictors
- Linear models underperform due to non-linear price dynamics in luxury segment
- Gradient Boosting outperforms on both RMSE and MAE

---

## Dataset
- **Source:** [Kaggle – House Price Prediction](https://www.kaggle.com/datasets/bhanupratapbiswas/house-price-prediction)
- 4,600 rows · 18 features · King County, Washington state

---

## Author
InternSpark Internship Task Submission
