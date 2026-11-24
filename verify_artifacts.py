
import joblib
from pathlib import Path

# Location of saved XGBoost artifacts
artifact_path = Path(r"E:\UPI_Fraud_Detection\xgboost_artifacts")

files = {
    "Model": "xgboost_model.pkl",
    "Preprocessor": "xgboost_preprocessor.pkl",
    "Threshold": "xgboost_threshold.pkl",
    "Feature Columns": "xgboost_feature_columns.pkl"
}

loaded_objects = {}

for name, filename in files.items():
    file_path = artifact_path / filename

    try:
        obj = joblib.load(file_path)
        loaded_objects[name] = obj

        print(f"\n{name}: Successfully loaded")
        print("Object type:", type(obj))

    except Exception as error:
        print(f"\n{name}: Loading failed")
        print("Error:", error)

# Additional information
if "Threshold" in loaded_objects:
    print("\nSaved threshold:", loaded_objects["Threshold"])

if "Feature Columns" in loaded_objects:
    columns = loaded_objects["Feature Columns"]
    print("Number of input features:", len(columns))
    print("Feature names:", list(columns))

if "Preprocessor" in loaded_objects:
    preprocessor = loaded_objects["Preprocessor"]
    print("Preprocessor fitted:", hasattr(preprocessor, "transform"))
