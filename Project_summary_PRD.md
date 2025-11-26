# UAP Explorer – PRD & Implementation Roadmap

---

## 1) PRODUCT REQUIREMENTS DOCUMENT (PRD)

### 1. Overview

**What we’re building**

UAP Explorer is a web application that visualizes UFO/UAP sighting reports on an interactive map and uses machine learning to highlight spatiotemporal anomalies and narrative patterns in the data. All heavy ML work is done offline; the deployed app reads precomputed JSON “tiles” and metadata, so it feels fast and responsive while remaining simple to host (static/serverless).

**Who it’s for**

- Curious, data-minded users interested in UAP/UFO sightings.
- Tech/ML recruiters and interviewers evaluating your applied ML and system design skills.
- Developers and researchers who want a rigorous, non-sensational overview of UAP reporting patterns.

**What problem it solves**

Raw UAP datasets are messy CSV dumps that are hard to explore meaningfully. UAP Explorer turns them into:
- A clean, explorable map with intuitive filters and a time slider.
- An anomaly score per report (and per region/time) combining spatiotemporal patterns and text-based clusters.
- A digestible way to answer “What’s happening where and when?” without claiming to prove anything extraordinary.

---

### 2. Goals & Non-Goals

**Goals (v1)**

- Build an **interactive world map** that loads quickly and can handle tens of thousands of UAP reports.
- Precompute and expose a **per-report anomaly score** using ML (regression + anomaly detection).
- Cluster textual descriptions into a small set of **phenomenology clusters** (e.g., “lights,” “triangles,” “orbs”) using embeddings and unsupervised learning.
- Provide **core filters** (time range, anomaly threshold, cluster type, shape) and a **time slider** to explore patterns.
- Host the app using a **static/serverless architecture** (e.g., Next.js on Vercel with static JSON assets), minimizing operational overhead.
- Make the system easy to explain in interviews: clear modeling choices, performance decisions, and tradeoffs.

**Non-Goals (v1)**

- **No real-time ingestion** of new reports or streaming data.
- **No user accounts, auth, or personalization** (favorites, comments, social features) in v1.
- **No complex backend orchestration** (no Kubernetes, no long-running servers; v1 is static/serverless).
- **No live ML inference in production**; all ML is precomputed offline.
- **No attempt to “prove aliens”**; the product is about anomaly detection in human reporting data, not claims about the underlying phenomena.
- **No mobile native apps**; v1 is a responsive web app only.

---

### 3. Target Users & Use Cases

**Primary user persona**

- **Curious Analyst / Tech-savvy Enthusiast**
  - Comfortable with web apps and data visualizations.
  - Interested in UAPs but values rigor and data over sensationalism.
  - Might be an interviewer or hiring manager checking your portfolio.

**Key use cases**

1. **Spatiotemporal exploration**
   - *As a curious user, I want to scrub through time on a map so that I can see when and where UAP reports cluster or spike.*

2. **Anomaly hunting**
   - *As a data-minded user, I want to filter by anomaly score so that I can focus on the most statistically unusual reports or regions.*

3. **Narrative pattern discovery**
   - *As a user, I want to filter by descriptive clusters (e.g., “triangular craft,” “fast lights”) so that I can explore different types of reported phenomena.*

4. **Drilling into an area**
   - *As a user, I want to click into a region or cluster to see individual reports, their descriptions, and why they were considered anomalous so that I can understand the model’s reasoning.*

5. **Portfolio / interview demo**
   - *As the project owner, I want to demo the app in an interview so that I can walk through the architecture, ML pipeline, and performance decisions clearly and quickly.*

---

### 4. Success Metrics (for v1)

**Qualitative / functional criteria**

- User can:
  - Load the map and see UAP sighting points or clusters within ~2–3 seconds on a typical connection.
  - Adjust a time range (time slider) and see the map update smoothly.
  - Filter by anomaly score and description cluster without the app freezing or crashing.
  - Click a point/cluster to see a detail panel with:
    - Raw description
    - Basic metadata (location, date, shape, duration)
    - Anomaly score and a short explanation.

**Quantitative-ish “done” criteria**

- Dataset size: Supports at least **50k–100k** sighting records without noticeable UI lag at typical zoom levels.
- Map responsiveness: Panning/zooming at interactive frame rates (> 30 FPS) on a modern laptop for typical views.
- Build pipeline:
  - One command (or documented set of commands) can:
    - Rebuild ML artifacts.
    - Regenerate static JSON tiles.
    - Redeploy the app.

---

### 5. Requirements

#### 5.1 Core Functional Requirements

1. **Data ingestion & cleaning (offline)**
   - Load one or more UAP/UFO sighting datasets (e.g., NUFORC-style CSVs).
   - Normalize:
     - Date/time (convert to a consistent timezone, or store as UTC).
     - Location (lat/lon, country, region).
     - Basic fields: shape, duration, description text.
   - Handle missing or obviously invalid data (drop or impute).

2. **Spatiotemporal aggregation & anomaly baseline (offline ML)**
   - Aggregate sightings into grid cells (e.g., hexes or lat/lon bins) and time windows (e.g., monthly).
   - Train a model to predict expected sighting counts per cell/time using features like:
     - Historical counts (autoregressive).
     - Month/seasonality.
     - Region-level properties (e.g., population density from external data, optional).
   - Compute anomaly scores (e.g., residuals or z-scores) for each cell/time combination.

3. **Text embeddings & clustering (offline ML)**
   - Generate embeddings for each sighting description using a suitable sentence embedding model.
   - Cluster embeddings into a small, interpretable number of clusters (e.g., 10–30).
   - Assign each sighting a cluster ID and cluster label (manually named from top terms/examples).

4. **Per-report anomaly scoring (offline ML)**
   - For each sighting, engineer features (e.g., duration, shape, local density, cell anomaly score, cluster ID).
   - Train and apply an anomaly detection model (e.g., Isolation Forest).
   - Store an anomaly score (0–1) for each report.

5. **Static data export**
   - Export:
     - A master JSON/Parquet file with all sightings and model outputs.
     - Tiled JSON per zoom-level or pre-clustered data for map display (e.g., `public/data/tiles/z/x_y.json`).
     - Metadata JSON: cluster labels, global stats (e.g., totals per year, per cluster).

6. **Interactive map UI**
   - Display the map using Mapbox GL or deck.gl.
   - Load sightings via static JSON tiles.
   - Show:
     - Clustered markers at low zoom levels.
     - Individual points at higher zoom levels.
   - Color/size encodes anomaly score by default.

7. **Filters & time slider**
   - Controls for:
     - Time range (start/end date).
     - Anomaly score threshold.
     - Cluster ID (e.g., “Triangles,” “Lights,” etc.).
     - Optional: shape, duration range.
   - Updates the map view and counts in real time (no full page reload).

8. **Sighting detail panel**
   - Clicking a point (or a cluster and then a specific sighting within it) opens a side panel containing:
     - Date/time, location (city/region, country).
     - Shape, duration.
     - Raw description.
     - Anomaly score.
     - Short explanation (e.g., “Unusual duration for this region/time” / “More reports than usual in this area/month”).

#### 5.2 Nice-to-Have / Stretch Requirements

1. **Insight dashboard**
   - Separate tab with charts:
     - Sightings over time.
     - Sightings by cluster.
     - Sightings by country/region.

2. **Explainability snippets**
   - Use a simple feature-importance explanation (e.g., SHAP offline) to generate a short human-readable rationale for anomaly scores.

3. **Bookmarkable state**
   - Encode filters and map viewport in the URL so users can share specific views.

4. **Minimal serverless endpoints**
   - For example, a `/api/stats` route that returns global stats computed from static data.

5. **Dark/light mode**
   - Theme toggle consistent with map style.

---

### 6. UX / UI Notes (High-level)

**Main screens/flows**

1. **Landing / Map Screen**
   - **Purpose:** Primary interaction surface where users explore sightings and anomalies.
   - **Key elements:**
     - Full-screen map (center).
     - Top bar:
       - App title (“UAP Explorer”).
       - Basic navigation (e.g., “Map”, “Insights”).
     - Left or right sidebar:
       - Filter controls (time range, anomaly threshold, cluster/shape).
       - Time slider with play/pause for animation.
     - On map:
       - Clusters or individual points.
       - Hover tooltip with basic info (date, location, anomaly score).
     - Detail panel (slide-out from side) when a point is clicked.
   - **Layout notes:**
     - Map should occupy most of the viewport.
     - Sidebar collapsible on smaller screens.
     - Use a clean, minimal design (few colors, consistent typography).

2. **Sighting Detail Panel**
   - **Purpose:** Deep dive into a single report.
   - **Key elements:**
     - Heading: date + location.
     - A few key stats in a compact card (shape, duration, anomaly score).
     - Description text in a scrollable area.
     - Simple explanation line: e.g., “Higher than typical for this region and time of year.”
   - **Layout notes:**
     - Slide-out from the right side.
     - Close button (X) and click outside to close.

3. **Insights / Analytics Screen (stretch)**
   - **Purpose:** High-level view of trends without the map.
   - **Key elements:**
     - Line chart of sightings over years/months.
     - Bar chart by cluster.
     - Map summary stats (e.g., top regions by anomalies).
   - **Layout notes:**
     - Simple grid of cards and charts.

---

### 7. Technical Constraints & Assumptions

- **Frontend:**
  - Next.js (React, preferably TypeScript).
  - Map rendering via Mapbox GL JS or deck.gl (on top of Mapbox or a similar basemap).
  - Deployed on Vercel as a mostly-static site.

- **Data & Offline ML:**
  - Python for data ingestion and ML (Jupyter notebooks + scripts).
  - Libraries: pandas, numpy, scikit-learn, sentence-transformers (or equivalent for embeddings), HDBSCAN/KMeans, possibly shap (for explanations).
  - Data and model artifacts stored locally, exported to JSON/Parquet for the app.

- **Runtime / Hosting:**
  - v1 uses static JSON data hosted via Vercel’s `public/` directory or a CDN bucket.
  - No always-on backend services.
  - Optional: a few Next.js Route Handlers / API routes for simple stats (serverless).

- **Performance considerations:**
  - Precompute tiles and clusters; no heavy ML inference at request time.
  - Avoid sending the entire dataset to the client at once; load per-tile or per-filter.
  - Use WebGL rendering to keep the map smooth.

- **Privacy / compliance:**
  - All data is public sighting reports (no PII expected).
  - No user accounts or sensitive user data in v1.

---

### 8. Risks & Open Questions

**Risks**

- **Data quality:** UAP datasets can be noisy, incomplete, or inconsistent across sources.
- **Performance risk:** If tiling is not done properly, loading too many points at once could cause lag.
- **Model interpretability:** Anomaly scores might be hard to explain simply; users could misinterpret them.
- **Map complexity:** Implementing clustering, tiles, and filters together can get tricky for a junior engineer.

**Open questions**

- Which **exact dataset(s)** will be used first (one NUFORC-style CSV, or multiple sources merged)?
- Which **embedding model** will be used for descriptions (local vs remote)? (For v1, assume a local sentence-transformer run offline.)
- Exact tiling strategy: Mapbox’s `z/x/y` style tiles vs custom “zoom bucket + bounding-box” JSON slices.
- Do we want **basic tests** around the ML pipeline (e.g., sanity checks on feature ranges, output counts)?

---

## 2) IMPLEMENTATION ROADMAP FOR A JUNIOR ENGINEER

### A. High-Level Phases

1. **Phase 1 – Repo Setup & Scaffolding**
   - Set up the Next.js app structure and Python data/ML environment.
2. **Phase 2 – Data Ingestion & Offline ML Pipeline**
   - Load and clean the UAP dataset; compute embeddings, clusters, and anomaly scores.
3. **Phase 3 – Static Data Export & Tiling**
   - Transform processed data into static JSON files optimized for the frontend (tiles + metadata).
4. **Phase 4 – Core Frontend Map & Filters**
   - Implement the interactive map UI, filters, and time slider using static data.
5. **Phase 5 – Detail Panel, Insights & Polish**
   - Add sighting detail panel, basic insights screen, and UI polish.
6. **Phase 6 – Testing, Deployment & Documentation**
   - Add basic tests, manual QA, deployment to Vercel, and final documentation.

---

### B. Detailed Task Breakdown

#### Phase 1 – Repo Setup & Scaffolding

**Task 1.1 – Initialize monorepo / project structure**

- **Description:** Create a single Git repo containing:
  - `app/` (Next.js project)
  - `ml/` (Python notebooks/scripts)
- **Implementation hints:**
  - Use `npx create-next-app@latest` for `app/`.
  - Inside `ml/`, set up a `requirements.txt` or `pyproject.toml`.
- **Dependencies:** None.
- **Acceptance criteria:**
  - Repo exists with both directories.
  - Both Next.js dev server and a Python virtual environment can run.

---

**Task 1.2 – Configure Next.js app basics**

- **Description:** Set up TypeScript (if not already), basic pages, and a layout.
- **Implementation hints:**
  - Enable TypeScript: run `npm run dev` and follow prompts.
  - Create a main page in `app/page.tsx` with placeholder text: “UAP Explorer”.
- **Dependencies:** Task 1.1.
- **Acceptance criteria:**
  - App runs locally at `localhost:3000`.
  - Visiting the homepage shows “UAP Explorer”.

---

**Task 1.3 – Setup styling & UI foundation**

- **Description:** Add a simple, consistent styling system (Tailwind CSS or similar).
- **Implementation hints:**
  - Follow Tailwind setup guide for Next.js.
  - Add a base layout with a header bar and content area.
- **Dependencies:** Task 1.2.
- **Acceptance criteria:**
  - Global styles applied.
  - Header with app name visible on all pages.

---

**Task 1.4 – Setup Python environment for ML**

- **Description:** Create and document a Python environment for the ML pipeline.
- **Implementation hints:**
  - Use `python -m venv .venv` and `pip install` for `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `matplotlib` (optional), etc.
  - Create `ml/README.md` with setup instructions.
- **Dependencies:** Task 1.1.
- **Acceptance criteria:**
  - `python` environment can import core libraries.
  - `ml/README.md` describes how to set it up.

---

#### Phase 2 – Data Ingestion & Offline ML Pipeline

**Task 2.1 – Download and inspect UAP dataset**

- **Description:** Place the chosen UAP/UFO dataset file(s) in `ml/data/raw/` and inspect basic columns and ranges.
- **Implementation hints:**
  - Create a Jupyter notebook: `ml/notebooks/01_explore_data.ipynb`.
  - Load CSV with `pandas.read_csv`.
  - Print head(), describe(), and value_counts for key fields.
- **Dependencies:** Phase 1 complete.
- **Acceptance criteria:**
  - Notebook shows basic EDA (row counts, key columns, example rows).
  - Clear note of which columns will be used (date, location, description, shape, duration).

---

**Task 2.2 – Implement data cleaning script**

- **Description:** Convert the EDA work into a script that outputs a cleaned CSV/Parquet file.
- **Implementation hints:**
  - Create `ml/scripts/clean_data.py`.
  - Steps:
    - Parse dates to a standard format.
    - Ensure lat/lon exist or drop rows without them.
    - Normalize text fields (strip whitespace).
  - Output to `ml/data/processed/cleaned_sightings.parquet`.
- **Dependencies:** Task 2.1.
- **Acceptance criteria:**
  - Running `python ml/scripts/clean_data.py` produces a processed file.
  - Row count and column names are logged.

---

**Task 2.3 – Spatiotemporal aggregation & baseline modeling**

- **Description:** Aggregate sightings by spatial grid and time, and build a baseline model for expected counts.
- **Implementation hints:**
  - Create `ml/notebooks/02_spatiotemporal_baseline.ipynb` for experimentation, then a script.
  - Choose a simple grid (e.g., round lat/lon to 1° or 0.5°, or use a hex library if desired).
  - Create features: `year`, `month`, `grid_lat`, `grid_lon`, and counts.
  - Train a simple model (start with linear regression, then consider gradient boosting).
  - Compute predicted vs actual counts and residuals/z-scores.
- **Dependencies:** Task 2.2.
- **Acceptance criteria:**
  - For each (grid, month/year), you have:
    - actual_count
    - predicted_count
    - anomaly_score_cell (e.g., positive z-score for higher-than-expected).
  - Results saved to `ml/data/processed/grid_time_anomalies.parquet`.

---

**Task 2.4 – Text embeddings & clustering**

- **Description:** Create embeddings for descriptions and cluster them into a small number of narrative categories.
- **Implementation hints:**
  - Create `ml/notebooks/03_text_clusters.ipynb`.
  - Use a sentence-transformer model to embed descriptions.
  - Cluster with KMeans (easier) or HDBSCAN (more advanced).
  - For each cluster:
    - Inspect top keywords/phrases.
    - Assign a short human-readable label by hand (e.g., “bright lights”, “triangular craft”).
  - Save:
    - Per-sighting `cluster_id`.
    - A mapping file `cluster_id -> label`.
- **Dependencies:** Task 2.2.
- **Acceptance criteria:**
  - Each sighting in processed data has a cluster ID.
  - `ml/data/processed/cluster_labels.json` contains label text per cluster.

---

**Task 2.5 – Per-report anomaly model**

- **Description:** Use features to train an anomaly detection model per report.
- **Implementation hints:**
  - Create `ml/notebooks/04_per_report_anomaly.ipynb`.
  - Merge:
    - Cleaned sightings
    - Cell-level anomaly score (from Task 2.3)
    - Cluster IDs (from Task 2.4)
  - Feature set examples:
    - `duration`, `hour_of_day`, `month`, `cell_anomaly_score`, `cluster_id_one_hot`, `shape_one_hot`.
  - Train an IsolationForest (or similar) and output anomaly scores.
  - Save per-sighting anomaly scores back into a processed Parquet/CSV.
- **Dependencies:** Tasks 2.3, 2.4.
- **Acceptance criteria:**
  - Each sighting has:
    - `anomaly_score_report` (0–1 or similar).
  - File saved as `ml/data/processed/sightings_with_scores.parquet`.

---

#### Phase 3 – Static Data Export & Tiling

**Task 3.1 – Design tile format & zoom strategy**

- **Description:** Decide how to split the dataset into manageable JSON files.
- **Implementation hints:**
  - For v1, start simple:
    - At medium zoom levels, bucket by large grid cell (e.g., 1° x 1°).
    - Create JSON file per grid cell with all sightings in that cell.
  - Later, refine to `z/x/y` if needed.
  - Define JSON structure, e.g.:
    ```json
    {
      "grid_id": "latX_lonY",
      "sightings": [
        {
          "id": "...",
          "lat": ...,
          "lon": ...,
          "timestamp": "...",
          "cluster_id": 3,
          "cluster_label": "Bright lights",
          "anomaly_score_report": 0.92,
          "cell_anomaly_score": 1.5,
          "shape": "triangle",
          "duration": 120
        }
      ]
    }
    ```
- **Dependencies:** Task 2.5.
- **Acceptance criteria:**
  - A documented JSON schema in `ml/README.md` or a separate doc.
  - Clear choice of how tiles/segments are defined.

---

**Task 3.2 – Implement export script**

- **Description:** Create a Python script that reads `sightings_with_scores.parquet` and writes static JSON tiles.
- **Implementation hints:**
  - Create `ml/scripts/export_tiles.py`.
  - For each grid cell (or tile bucket):
    - Filter relevant rows.
    - Write to `app/public/data/tiles/{grid_id}.json`.
  - Also write:
    - `app/public/data/metadata/cluster_labels.json`.
    - Optional: `app/public/data/metadata/global_stats.json`.
- **Dependencies:** Task 3.1.
- **Acceptance criteria:**
  - Running `python ml/scripts/export_tiles.py` populates `app/public/data/tiles/` and `app/public/data/metadata/`.
  - A few sample files inspected and valid JSON.

---

#### Phase 4 – Core Frontend Map & Filters

**Task 4.1 – Integrate map library**

- **Description:** Add a map component to the Next.js app.
- **Implementation hints:**
  - Use Mapbox GL JS (or similar).
  - Create a `MapView` component in `app/components/MapView.tsx`.
  - Hard-code a center and zoom level for now.
- **Dependencies:** Phase 1 complete.
- **Acceptance criteria:**
  - Map renders full-screen area on the main page.
  - You can pan and zoom.

---

**Task 4.2 – Load and render static tiles**

- **Description:** Fetch and render sightings from a small set of JSON tiles.
- **Implementation hints:**
  - Create a simple function to decide which tiles to load based on current viewport (or start by loading a fixed tile to keep it simple).
  - Use `fetch('/data/tiles/GRID_ID.json')`.
  - Plot points using the map library’s layer system (e.g., GeoJSON layer).
- **Dependencies:** Task 3.2, Task 4.1.
- **Acceptance criteria:**
  - Sightings appear as points on the map.
  - Panning/zooming doesn’t break the map (even if tile loading is still naive).

---

**Task 4.3 – Add basic filters (cluster, anomaly threshold)**

- **Description:** Add UI controls and filter logic.
- **Implementation hints:**
  - Create a `FiltersPanel` component with:
    - Multi-select for cluster(s) (using labels from `cluster_labels.json`).
    - Slider for anomaly score (0–1).
  - In state, maintain current filters.
  - Filter the loaded sightings before rendering.
- **Dependencies:** Task 4.2.
- **Acceptance criteria:**
  - Changing cluster selection updates visible points.
  - Moving the anomaly threshold slider hides/shows points accordingly.

---

**Task 4.4 – Implement time slider**

- **Description:** Allow users to select a date range or move a time window.
- **Implementation hints:**
  - Decide on granularity (e.g., by month or year).
  - Add a slider component with min & max dates available in metadata.
  - Filter sightings by timestamp converted to numeric form (e.g., epoch).
  - Optional: add play/pause button to auto-advance the window.
- **Dependencies:** Task 4.2.
- **Acceptance criteria:**
  - Moving the time slider updates the map.
  - If play is implemented, the map animates through time without crashing.

---

#### Phase 5 – Detail Panel, Insights & Polish

**Task 5.1 – Sighting detail panel**

- **Description:** Show detailed info when the user clicks on a point.
- **Implementation hints:**
  - On point click, store selected sighting in state.
  - Create `SightingDetailPanel` component:
    - Show date/time, location, shape, duration.
    - Show cluster label and both anomaly scores.
    - Show full text description.
  - Slide-in animation from the side (optional but nice).
- **Dependencies:** Task 4.2.
- **Acceptance criteria:**
  - Clicking a point opens the detail panel.
  - Closing the panel returns to the normal map view.

---

**Task 5.2 – Basic insights screen (stretch)**

- **Description:** Provide a simple “Insights” tab with charts.
- **Implementation hints:**
  - Create a new page `app/insights/page.tsx`.
  - Load `global_stats.json` (if available) or precomputed aggregated JSON.
  - Use a chart library to show:
    - Sightings over time.
    - Sightings by cluster.
- **Dependencies:** Task 3.2.
- **Acceptance criteria:**
  - User can navigate to `/insights`.
  - Charts render without errors on sample data.

---

**Task 5.3 – UI polish**

- **Description:** Improve the look and usability.
- **Implementation hints:**
  - Align fonts, spacing, and color palette (light/dark map, subtle UI colors).
  - Ensure filters panel design is clean and readable.
  - Add a small legend explaining color encodings for anomaly scores.
- **Dependencies:** Core features complete (Phases 4 & 5.1).
- **Acceptance criteria:**
  - UI looks cohesive and is easy to understand for a new user.
  - No overlapping elements on common screen sizes.

---

#### Phase 6 – Testing, Deployment & Documentation

**Task 6.1 – Basic frontend tests**

- **Description:** Add a few tests for critical components.
- **Implementation hints:**
  - Use Jest/React Testing Library.
  - Test:
    - FiltersPanel changes state correctly.
    - MapView renders without crashing when given sample data.
- **Dependencies:** Core frontend implemented.
- **Acceptance criteria:**
  - `npm test` passes.
  - At least 3–5 meaningful tests exist.

---

**Task 6.2 – Manual QA checklist**

- **Description:** Perform a set of manual checks and fix any obvious bugs.
- **Implementation hints:**
  - Create `QA_CHECKLIST.md` with items like:
    - Map loads within a few seconds.
    - Filters respond quickly.
    - Time slider works.
    - Detail panel shows correct data.
    - No JS errors in console during normal usage.
- **Dependencies:** All main features implemented.
- **Acceptance criteria:**
  - QA checklist completed and issues fixed or documented.

---

**Task 6.3 – Deployment to Vercel**

- **Description:** Deploy the app to a public URL.
- **Implementation hints:**
  - Hook GitHub repo to Vercel.
  - Ensure `public/data/` is included in the build.
  - Configure Mapbox access token via environment variable if needed.
- **Dependencies:** App runs locally without major issues.
- **Acceptance criteria:**
  - App accessible at a public URL.
  - You can open it in a fresh browser and use core features.

---

**Task 6.4 – Final documentation**

- **Description:** Write clear docs for future you (and interviewers).
- **Implementation hints:**
  - Update root `README.md` with:
    - Project overview (short).
    - Setup steps for ML pipeline and Next.js app.
    - How to regenerate tiles and redeploy.
    - High-level architecture diagram (even ASCII is fine).
  - Update `ml/README.md` with:
    - Data pipeline steps.
    - Commands to run each script.
    - Description of models used and features.
- **Dependencies:** Entire project done.
- **Acceptance criteria:**
  - A new contributor can read the README files and:
    - Set up the environment.
    - Regenerate data.
    - Run the app locally.

---

### C. Milestones & Suggested Timeline

Assume 10–20 hours/week for one junior engineer.

- **M1 (Week 1–2): Basic structure & data pipeline started**
  - Phase 1 completed.
  - Tasks 2.1–2.2 done (data cleaned).
- **M2 (Week 3–4): ML pipeline outputs ready**
  - Tasks 2.3–2.5 completed.
  - `sightings_with_scores.parquet` produced.
- **M3 (Week 5): Static tiles + minimal map**
  - Phase 3 completed.
  - Tasks 4.1–4.2 completed.
  - Map shows points from tiles.
- **M4 (Week 6–7): Filters, time slider & detail panel**
  - Tasks 4.3, 4.4, 5.1 completed.
  - Map is fully interactive and filterable.
  - **This is a demo-ready v1.**
- **M5 (Week 8): Polish, insights & deployment**
  - Tasks 5.2, 5.3, Phase 6 completed.
  - App deployed to production URL with documentation.

Critical demo milestone: **M4**  
By M4 you can already show:
- ML-derived anomaly scores and clusters.
- A snappy, interactive map with filters and details.

---

### D. Testing & QA Plan

**Types of tests**

- **Unit tests** (frontend):
  - Basic component behavior (FiltersPanel, time slider).
- **Integration tests** (lightweight):
  - The main page loads and can apply filters without crashing.
- **Manual e2e tests:**
  - Full flow from loading app → applying filters → clicking points.

**What to test (critical features)**

1. **Data loading**
   - App can load tile JSON files:
     - Valid JSON.
     - Fail gracefully if a tile file is missing.

2. **Filters**
   - Changing cluster selection updates the visible points.
   - Changing anomaly threshold does not crash rendering.
   - Time slider adjusts sightings according to selected range.

3. **Map behavior**
   - Panning/zooming does not cause UI freezes.
   - Tooltips and click events work.

4. **Detail panel**
   - Clicking a point shows the correct details.
   - Closing the panel works and doesn’t affect the map.

**Manual QA checklist (examples)**

- Load homepage in incognito:
  - Map appears within a few seconds.
- Apply filters:
  - No console errors.
  - Number of points visually changes in a consistent way.
- Rapid sliding of time slider:
  - App remains responsive.
- Resize window (desktop → tablet width):
  - Sidebar and map layout still usable.
- Hard refresh:
  - App still loads; no missing assets.

---

### E. Documentation & Handover

**What to document**

- **Root README:**
  - Project overview and goals.
  - Tech stack summary.
  - How to run the Next.js app.
  - High-level architecture (offline ML → static data → frontend).
  - Link to deployed URL.

- **ML README (in `ml/`):**
  - Data sources and any preprocessing assumptions.
  - Explanation of each script/notebook:
    - `clean_data.py`
    - `export_tiles.py`
    - Key notebooks and what they explore.
  - Description of ML models:
    - Baseline regression for cells.
    - Text clustering approach.
    - Anomaly detection and feature set.

- **Architecture notes:**
  - How tiles are structured and named.
  - How the frontend decides which tiles to fetch.

**Keeping docs up to date**

- After each significant change (new features, changed data formats), update:
  - Relevant README sections.
  - Any JSON schema notes.
- For breaking changes in data schema:
  - Add a small “Changelog” section at the bottom of the root README.

---

## First Day Plan (for a Junior Engineer)

1. **Clone the repo** from GitHub to your local machine.
2. **Read the root `README.md`** to understand:
   - What UAP Explorer is.
   - The high-level architecture.
3. **Set up the Next.js app:**
   - `cd app`
   - `npm install`
   - `npm run dev`
   - Confirm the basic homepage loads at `localhost:3000`.
4. **Set up the Python environment:**
   - `cd ml`
   - Create a virtual environment and install dependencies (`pip install -r requirements.txt`).
5. **Open the first notebook** (`ml/notebooks/01_explore_data.ipynb`) and:
   - Run the first few cells to confirm the dataset loads correctly.
6. **Skim through the PRD sections**:
   - Overview
   - Requirements
   - Technical Constraints
   - Implementation Roadmap (Phases & Tasks)
7. **Start with the first assigned task** (likely Task 2.1 or any remaining Task 1.x):
   - Make sure you understand its acceptance criteria.
   - Ask questions if anything about the data or environment is unclear.

From there, proceed task-by-task, checking off acceptance criteria as you go.
