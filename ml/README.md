# UAP Explorer - ML Pipeline

This directory contains the offline machine learning pipeline for processing UAP sighting data and generating anomaly scores.

## Setup

### 1. Create a Python virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Directory Structure

```
ml/
├── data/
│   ├── raw/          # Raw UAP datasets (CSV files)
│   └── processed/    # Cleaned and processed data
├── notebooks/        # Jupyter notebooks for exploration
├── scripts/          # Python scripts for data processing
└── requirements.txt  # Python dependencies
```

## Pipeline Overview

The ML pipeline consists of several stages:

1. **Data Ingestion & Cleaning** (`scripts/clean_data.py`)
   - Load raw UAP sighting datasets
   - Normalize dates, locations, and text fields
   - Handle missing data

2. **Spatiotemporal Aggregation** (`notebooks/02_spatiotemporal_baseline.ipynb`)
   - Aggregate sightings by grid cells and time windows
   - Train baseline model for expected sighting counts
   - Compute cell-level anomaly scores

3. **Text Embeddings & Clustering** (`notebooks/03_text_clusters.ipynb`)
   - Generate embeddings for sighting descriptions
   - Cluster into narrative categories (lights, triangles, orbs, etc.)
   - Assign human-readable labels

4. **Per-Report Anomaly Scoring** (`notebooks/04_per_report_anomaly.ipynb`)
   - Engineer features combining spatiotemporal and textual data
   - Train anomaly detection model (Isolation Forest)
   - Generate per-report anomaly scores

5. **Static Data Export** (`scripts/export_tiles.py`)
   - Export processed data to JSON tiles for frontend
   - Generate metadata files

## Getting Started

1. Place your UAP dataset CSV file(s) in `data/raw/`
2. Start with the exploration notebook:
   ```bash
   jupyter notebook notebooks/01_explore_data.ipynb
   ```
3. Follow the notebooks in order (01, 02, 03, 04)
4. Run the export script to generate frontend data:
   ```bash
   python scripts/export_tiles.py
   ```

## Models & Approach

### Spatiotemporal Baseline
- Uses historical counts and seasonality to predict expected sighting rates
- Residuals identify anomalous spikes or drops in reporting

### Text Clustering
- Sentence-transformers for semantic embeddings
- KMeans or HDBSCAN for clustering
- Manual labeling of clusters based on top examples

### Anomaly Detection
- Isolation Forest on engineered features
- Features include: duration, time of day, cell anomaly score, cluster ID, shape
- Output: anomaly score (0-1) per sighting

## Notes

- All ML work is done offline; the web app reads static JSON files
- Models are retrained when new data is added
- Processing ~50k-100k sightings should complete in minutes to hours depending on hardware
