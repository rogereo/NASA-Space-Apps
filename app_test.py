"""
predict_koi_sample.py
---------------------
Use a saved model pickle (GradientBoosting / XGBoost / LightGBM)
to predict on the saved KOI testing sample.

Run example:
    python predict_koi_sample.py --model models/gradient_boosting.pkl
"""

import argparse
import pickle
import pandas as pd
from pathlib import Path

# -------------------------------------------------------
# 1. Parse CLI argument for which model to load
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Run KOI test predictions from a pickled model.")
parser.add_argument("--model", type=str, required=True, help="Path to saved model pickle (.pkl)")  # ✅ use --model
args = parser.parse_args()

pkl_path = Path(args.model)
if not pkl_path.exists():
    raise FileNotFoundError(f"❌ Model pickle not found: {pkl_path}")

# -------------------------------------------------------
# 2. Load model bundle
# -------------------------------------------------------
print(f"📦 Loading model from: {pkl_path}")
with open(pkl_path, "rb") as f:
    bundle = pickle.load(f)

model_name = bundle.get("model_name", "unknown_model")
model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]

print(f"✅ Model loaded: {model_name}")
print(f"Features expected: {len(feature_cols)} ({', '.join(feature_cols)})")

# -------------------------------------------------------
# 3. Load KOI testing sample
# -------------------------------------------------------
X_test_path = Path("data/koi_testing_sample/X_test.csv")
if not X_test_path.exists():
    raise FileNotFoundError(f"❌ Testing sample not found: {X_test_path}")

X_test = pd.read_csv(X_test_path)
print(f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Ensure correct column order
X_test = X_test[feature_cols]

# -------------------------------------------------------
# 4. Run prediction
# -------------------------------------------------------
print("\n🚀 Running predictions ...")
y_pred_encoded = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# -------------------------------------------------------
# 5. Save + Display Results
# -------------------------------------------------------
results = X_test.copy()
results["pred_encoded"] = y_pred_encoded
results["pred_label"] = y_pred_labels

out_path = Path("data/koi_testing_sample/predictions.csv")
results.to_csv(out_path, index=False)
print(f"\n💾 Predictions saved to: {out_path.resolve()}\n")

# Display top 10 predictions
print("=== Sample Predictions ===")
print(results.head(10))
