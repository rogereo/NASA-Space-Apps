# Exoplanet ML – Flask MVP

A minimal Flask app to host an exoplanet classifier API + simple UI.

## Quickstart (Local)
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py  # http://localhost:8000
```

## Endpoints
- `GET /` – Web UI
- `GET /health` – Health check
- `POST /predict` – JSON body:
```json
{ "features": { "koi_model_snr": 25.8, "koi_depth": 875, "koi_duration": 3.1, "koi_prad": 2.2 } }
```

## Deploy (Gunicorn)
```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

## Plugging in your trained model
Save your trained model and feature order:
```python
import joblib, json
joblib.dump(model, "models/model.pkl")
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)
```
Then start the app; it will auto-load the bundle if present.

### Environment variables
- `MODEL_PATH` (default: `models/model.pkl`)
- `FEATURES_PATH` (default: `models/feature_names.json`)
- `PORT` (default: `8000`)
- `FLASK_DEBUG` (`1` for dev, `0` for prod`)
