import joblib
import pickle
from pathlib import Path

folder = Path(__file__).resolve().parent / "models"

files = [
    "xgboost_model.pkl",
    "xgboost_preprocessor.pkl",
    "xgboost_feature_columns.pkl",
    "xgboost_threshold.pkl"
]

for filename in files:
    path = folder / filename

    print(f"\nChecking: {filename}")

    try:
        obj = joblib.load(path)
        print("Joblib: SUCCESS")
    except Exception as e:
        print("Joblib: FAILED:", e)