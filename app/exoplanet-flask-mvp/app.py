import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from model_utils import load_model_bundle, predict_with_bundle, heuristic_predict

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "model.pkl"))
FEATURES_PATH = os.environ.get("FEATURES_PATH", os.path.join("models", "feature_names.json"))
bundle = load_model_bundle(MODEL_PATH, FEATURES_PATH)

@app.route("/")
def index():
    return render_template("index.html", model_loaded=bundle is not None)

@app.route("/health")
def health():
    return jsonify(ok=True, model_loaded=bundle is not None)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    features = data.get("features", data)
    if not isinstance(features, dict) or not features:
        return jsonify(ok=False, error="Send JSON with a 'features' object containing numeric values."), 400
    try:
        if bundle is not None:
            proba = predict_with_bundle(bundle, features)
        else:
            proba = heuristic_predict(features)
        pred = int(proba >= 0.5)
        return jsonify(ok=True, prediction=pred, proba=float(proba), used_features=features)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
