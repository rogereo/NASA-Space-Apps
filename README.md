<p align="center">
  <img src="assets/SpaceApps.png" alt="NASA International Space Apps Challenge" width="360" />
</p>

# Exoplanet Explorer (NASA Space Apps)

An interactive Flask web app for exploring exoplanet datasets (KOI and TESS) with search, summary stats, charts, and a clean dashboard UI.

## Challenge Context
Built for the NASA Space Apps Challenge (2025) — “A World Away: Hunting for Exoplanets with AI.” The challenge asks teams to analyze open exoplanet datasets and prototype AI/ML methods to help identify exoplanets. This project delivers:
- A fast, user-friendly way to browse KOI and TESS data
- Basic insights such as confirmation status and key features per object
- A starting point for model development via an exploration notebook

## What’s Inside
- Web app: Flask backend with HTML templates for Home, Dashboard, About, and 404
- Data: CSVs for KOI (`data_koi.csv`), TESS (`data_tess.csv`), and combined (`data_combo.csv`)
- Dashboard features:
  - Dataset selector (KOI / TESS / KOI & TESS)
  - Quick search and summary stats
  - Paginated, sortable table with highlight for confirmed planets
  - Visualization area with space for educational summaries and lightcurve views
- Notebooks: early experiments and model training (`exoplanet_model_training.ipynb`)

## Quick Start
1. Python 3.9+ recommended
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`
4. Open: `http://localhost:5000`

## Key Routes
- `/` — Landing page
- `/about` — Project overview
- `/dashboard` — Data dashboard (use dropdown or `?source=koi|tess|combo`)
- `/api/data/<source>` — Serves CSVs to the dashboard (`koi`, `tess`, `combo`)

## Files of Note
- `app.py` — Flask app, routes, and data serving
- `templates/` — UI pages (`index.html`, `dashboard.html`, `about.html`, `404.html`)
- `data_koi.csv`, `data_tess.csv`, `data_combo.csv` — Datasets used by the dashboard
- `requirements.txt` — Minimal dependencies (`Flask`, `pandas`)
- `exoplanet_model_training.ipynb` — Early model training exploration

## How It Works (At a Glance)
- The dashboard fetches CSVs from the Flask app and renders them client-side for speed
- Search, stats, and pagination are handled in the browser

## Data Sources
- KOI (Kepler Objects of Interest) — NASA Exoplanet Archive
- TESS Exoplanet Candidates — NASA Exoplanet Archive

## Roadmap
- Link to official dataset pages from the UI
- Integrate lightcurve rendering (e.g., via Lightkurve)
- Surface AI/ML results on the dashboard

