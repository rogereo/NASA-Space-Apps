# Exoplanet Explorer (NASA Space Apps)

A lightweight Flask web app to explore exoplanet datasets with an interactive dashboard. It ships with KOI (Kepler Objects of Interest), TESS, and a combined dataset, plus simple search, charts, and a clean UI.

## What’s Inside
- Web app: Flask backend with HTML templates for Home, Dashboard, About, and 404
- Data: CSVs for KOI (`data_koi.csv`), TESS (`data_tess.csv`), and combined (`data_combo.csv`)
- Dashboard features:
  - Dataset selector (KOI / TESS / KOI & TESS)
  - Quick search and summary stats
  - Paginated, sortable table with highlight for confirmed planets
  - Visualization area with room for educational summaries and lightcurve views
  - Optional 3D embedding viewer (served from a sibling repo path)
- Notebooks: early experiments and model training (`exoplanet_model_training.ipynb`)

## Quick Start
1. Python 3.9+ recommended.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `python app.py`
4. Open: `http://localhost:5000`

## Key Routes
- `/` — Landing page
- `/about` — Project overview and team section
- `/dashboard` — Data dashboard (choose dataset via dropdown or `?source=koi|tess|combo`)
- `/api/data/<source>` — Serves CSVs to the dashboard (`koi`, `tess`, `combo`)
- `/assets/embedding/<file>` — Serves embedding assets (see note below)

## Embedding Viewer (Optional)
If you have a sibling repo at `../rogereo.github.io/assets/embedding`, the dashboard can embed 3D projections and related assets. Without it, those views simply 404 and the rest of the dashboard still works.

## Files of Note
- `app.py` — Flask app, routes, and data serving
- `templates/` — UI pages (`index.html`, `dashboard.html`, `about.html`, `404.html`)
- `data_koi.csv`, `data_tess.csv`, `data_combo.csv` — Datasets used by the dashboard
- `requirements.txt` — Minimal dependencies (`Flask`, `pandas`)
- `exoplanet_model_training.ipynb` — Model training exploration

## How It Works (At a Glance)
- The dashboard fetches CSVs from the Flask app and renders them client‑side.
- Search, stats, and table pagination are done in the browser for responsiveness.
- Embedding assets (if present) are served via `/assets/embedding/...` from the sibling site folder.

## Next Steps (Ideas)
- Add dataset source links and metadata (e.g., NASA Exoplanet Archive references)
- Integrate live lightcurve rendering (e.g., via Lightkurve)
- Expand model training notebook and surface results in the dashboard

## License
No license specified. Add one if you plan to distribute.
