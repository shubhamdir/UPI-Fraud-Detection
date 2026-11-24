
import joblib
import os

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# XGBoost artifact directory
MODEL_DIR = os.path.join(
    BASE_DIR,
    "upi_fraud_app",
    "models"
)

# Artifact paths
model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "xgboost_preprocessor.pkl")
features_path = os.path.join(MODEL_DIR, "xgboost_feature_columns.pkl")
threshold_path = os.path.join(MODEL_DIR, "xgboost_threshold.pkl")

print("=" * 50)
print("UPI FRAUD DETECTION - ARTIFACT VERIFICATION")
print("=" * 50)

# Load artifacts
model = joblib.load(model_path)
print("\n1. XGBoost model loaded successfully")

preprocessor = joblib.load(preprocessor_path)
print("2. Preprocessor loaded successfully")

feature_columns = joblib.load(features_path)
print("3. Feature columns loaded successfully")

threshold = joblib.load(threshold_path)
print("4. Classification threshold loaded successfully")

# Display verification details
print("\n" + "=" * 50)
print("ARTIFACT DETAILS")
print("=" * 50)

print("\nModel type:", type(model))
print("Preprocessor type:", type(preprocessor))
print("Number of features:", len(feature_columns))
print("Feature columns:", feature_columns)
print("Classification threshold:", threshold)

print("\nModel expected features:",
      getattr(model, "n_features_in_", "Not available"))

print("\n" + "=" * 50)
print("ARTIFACT VERIFICATION COMPLETED")
print("=" * 50)
# Verify preprocessor feature dimensions
transformed_features = preprocessor.get_feature_names_out()

print("\nPreprocessor Verification")
print("-------------------------")
print("Original features:", len(feature_columns))
print("Transformed features:", len(transformed_features))
print("Model expected features:", model.n_features_in_)

print("\nFirst 15 transformed features:")
print(transformed_features[:15])

assert len(transformed_features) == model.n_features_in_, \
    "Feature count mismatch!"

print("\nPreprocessor verification successful!")

# ==========================================
# FINAL TEST PREDICTION VALIDATION
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Load the original dataset
DATA_PATH = r"E:\UPI_Fraud_Detection\data\upi_transactions_synthetic_enhanced_final.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

# Reconstruct original features and target
X = df[feature_columns].copy()
y = df["is_fraud"].copy()

# Reproduce the original notebook split
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool,
    y_train_pool,
    test_size=0.20,
    random_state=42,
    stratify=y_train_pool
)

# Transform test data using the SAVED preprocessor
X_test_processed = preprocessor.transform(X_test)

# Generate fraud probabilities
y_test_prob = model.predict_proba(X_test_processed)[:, 1]

# Apply saved optimized threshold
y_test_pred = (
    y_test_prob >= threshold
).astype(int)

# Display results
print("\nFINAL TEST CLASSIFICATION REPORT")
print(classification_report(y_test, y_test_pred, digits=4))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_test_pred))

print("\nFINAL TEST METRICS")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_test_prob):.4f}")

print("\nTEST DATA SHAPE:", X_test_processed.shape)
print("TEST FRAUD CASES:", int(y_test.sum()))
print("\nPREDICTION VALIDATION COMPLETED!")
