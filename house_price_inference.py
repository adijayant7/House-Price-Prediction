"""
House Price Predictor – Inference Script
=========================================
Usage:
    python inference.py                  # runs built-in demo sample
    python inference.py --interactive    # prompts for each feature
"""
import sys, joblib, numpy as np, pandas as pd

MODEL_PATH = "house_price_best_model.joblib"

def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"], bundle["feature_names"]

def predict_price(features: dict, model_path=MODEL_PATH) -> dict:
    clf, scaler, feat_names = load_model(model_path)
    sample = pd.DataFrame([features])[feat_names]
    scaled = scaler.transform(sample)
    log_pred = clf.predict(scaled)[0]
    price = np.expm1(log_pred)
    return {"predicted_price": round(float(price), 2),
            "log_price": round(float(log_pred), 4)}

DEMO = {
    "log_sqft_living": np.log1p(1800),
    "log_sqft_lot":    np.log1p(5000),
    "bedrooms":        3,
    "bathrooms":       2.0,
    "floors":          1.0,
    "waterfront":      0,
    "view":            0,
    "condition":       3,
    "sqft_above":      1800,
    "sqft_basement":   0,
    "house_age":       34,
    "was_renovated":   0,
    "years_since_reno":0,
    "bath_bed_ratio":  2/4,
    "living_lot_ratio":1800/5000,
    "basement_ratio":  0,
    "city_enc":        12,
    "sale_month":      6,
}

if __name__ == "__main__":
    print("=" * 50)
    print("  🏠  House Price Predictor")
    print("=" * 50)

    if "--interactive" in sys.argv:
        feats = {}
        feats["log_sqft_living"] = np.log1p(float(input("Sqft Living (e.g. 1800): ")))
        feats["log_sqft_lot"]    = np.log1p(float(input("Sqft Lot   (e.g. 5000): ")))
        feats["bedrooms"]        = float(input("Bedrooms   (e.g. 3): "))
        feats["bathrooms"]       = float(input("Bathrooms  (e.g. 2): "))
        feats["floors"]          = float(input("Floors     (e.g. 1): "))
        feats["waterfront"]      = int(input("Waterfront (0/1): "))
        feats["view"]            = int(input("View score (0-4): "))
        feats["condition"]       = int(input("Condition  (1-5): "))
        feats["sqft_above"]      = float(input("Sqft Above (e.g. 1800): "))
        feats["sqft_basement"]   = float(input("Sqft Basement (e.g. 0): "))
        feats["house_age"]       = float(input("House Age  (e.g. 30): "))
        feats["was_renovated"]   = int(input("Renovated? (0/1): "))
        feats["years_since_reno"]= float(input("Years since reno (0 if never): "))
        bed = feats["bedrooms"]
        feats["bath_bed_ratio"]  = feats["bathrooms"] / (bed + 1)
        feats["living_lot_ratio"]= np.expm1(feats["log_sqft_living"]) / (np.expm1(feats["log_sqft_lot"]) + 1)
        feats["basement_ratio"]  = feats["sqft_basement"] / (np.expm1(feats["log_sqft_living"]) + 1)
        feats["city_enc"]        = int(input("City code  (0-20, Seattle≈12): "))
        feats["sale_month"]      = int(input("Sale month (1-12): "))
    else:
        feats = DEMO
        print("\nRunning demo sample:")
        print(f"  3 bed | 2 bath | 1800 sqft | built 1990 | Seattle")

    result = predict_price(feats)
    print(f"\n  💰 Predicted Price : ${result['predicted_price']:,.0f}")
    print(f"  log(Price)         : {result['log_price']}")
    print("=" * 50)
