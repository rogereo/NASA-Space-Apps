import os, json, math
import numpy as np
try:
    import joblib
except Exception:
    joblib = None

class ModelBundle:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

def load_model_bundle(model_path, feature_names_path):
    if not (os.path.exists(model_path) and os.path.exists(feature_names_path) and joblib is not None):
        return None
    try:
        model = joblib.load(model_path)
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        if not isinstance(feature_names, list):
            raise ValueError("feature_names.json must be a JSON list of strings.")
        return ModelBundle(model, feature_names)
    except Exception:
        return None

def predict_with_bundle(bundle, features: dict) -> float:
    xs = []
    for name in bundle.feature_names:
        val = features.get(name, 0.0)
        try:
            xs.append(float(val))
        except Exception:
            xs.append(0.0)
    X = np.array(xs, dtype=float).reshape(1, -1)
    model = bundle.model
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(X).ravel()[0])
        proba = 1.0 / (1.0 + math.exp(-margin))
    else:
        pred = float(model.predict(X).ravel()[0])
        proba = pred
    return float(proba)

def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def heuristic_predict(features: dict) -> float:
    snr = float(features.get("koi_model_snr", 0.0) or 0.0)
    depth = float(features.get("koi_depth", 0.0) or 0.0)  # ppm
    duration = float(features.get("koi_duration", 0.0) or 0.0)  # hours
    prad = float(features.get("koi_prad", 0.0) or 0.0)  # Earth radii
    score = 0.0
    score += 0.15 * min(snr, 100) / 10.0
    score += 0.10 * min(duration, 10) / 5.0
    score += 0.10 * min(depth, 20000) / 5000.0
    score += 0.05 * max(0.0, 3.0 - abs(prad - 2.0))
    return _sigmoid(score - 0.8)
