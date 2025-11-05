from __future__ import annotations

from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

app = Flask(__name__)

# Optional legacy constant kept for compatibility with older views
HIDDEN_COLS = {"gif", "nasa_url", "educational_summary"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dashboard")
def dashboard():
    # The dashboard loads datasets client-side. We only pass through the
    # selected dataset so the page can initialize correctly.
    source = request.args.get("source", "koi")
    return render_template("dashboard.html", source=source)


# Serve dataset CSVs to the frontend (whitelisted)
BASE_DIR = Path(__file__).resolve().parent
DATASETS = {
    "koi": "data_koi.csv",
    "tess": "data_tess.csv",
    "combo": "data_combo.csv",
}


@app.route("/api/data/<source>")
def api_data(source: str):
    key = source.lower()
    filename = DATASETS.get(key)
    if not filename:
        abort(404)
    path = BASE_DIR / filename
    if not path.exists():
        abort(404)
    # Serve as text/csv for fetch().text()
    return send_from_directory(BASE_DIR, filename, mimetype="text/csv")


@app.route("/search")
def search():
    # Lightweight search fallback against a local CSV if present
    csv_path = Path(__file__).with_name("data.csv")
    if not csv_path.exists():
        return jsonify({"data": [], "total_results": 0})

    query = request.args.get("q", "").lower().strip()
    try:
        df = pd.read_csv(csv_path)
        if query:
            mask = df.astype(str).apply(lambda s: s.str.lower().str.contains(query)).any(axis=1)
            df = df[mask]
        data = df.to_dict("records")
        return jsonify({"data": data, "total_results": len(data)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.route("/error")
def error():
    try:
        _ = 1 / 0
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", debug=True)
