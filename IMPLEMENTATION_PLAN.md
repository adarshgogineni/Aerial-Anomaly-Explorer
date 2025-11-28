# 🚀 UAP Explorer - Implementation Plan & Progress Tracker

> **Last Updated**: 2025-11-28
> **Version**: 1.1
> **Status**: Phase 1-3 Complete, Phase 4 Ready to Start

This is your comprehensive, step-by-step guide to building the UAP Explorer application. Check off tasks as you complete them to track your progress.

---

## 📊 Overall Progress

```
Phase 1: Setup & Scaffolding           ████████████████████ 100% (9/9)   ✅ COMPLETE
Phase 2: Data & ML Pipeline            ████████████████████ 100% (6/6)   ✅ COMPLETE
Phase 3: Static Data Export            ████████████████████ 100% (2/2)   ✅ COMPLETE
Phase 4: Frontend Map & Filters        ░░░░░░░░░░░░░░░░░░░░   0% (0/5)   🔄 CURRENT
Phase 5: Detail Panel & Polish         ░░░░░░░░░░░░░░░░░░░░   0% (0/4)   ⏸️ PENDING
Phase 6: Testing & Deployment          ░░░░░░░░░░░░░░░░░░░░   0% (0/4)   ⏸️ PENDING

TOTAL PROGRESS: 57% (17/30 tasks)
```

---

## ✅ Phase 1: Repo Setup & Scaffolding (COMPLETED)

**Status**: ✅ Complete
**Time Invested**: ~1 hour
**Progress**: 9/9 tasks complete

### Completed Tasks:

- [x] **Task 1.1**: Create project directory structure
  - Created `app/` and `ml/` directories
  - Created subdirectories: `ml/data/raw/`, `ml/data/processed/`, `ml/notebooks/`, `ml/scripts/`
  - Created `app/components/`, `app/public/data/tiles/`, `app/public/data/metadata/`

- [x] **Task 1.2**: Initialize Next.js app with TypeScript
  - Created `package.json` with Next.js 15, React 19, TypeScript
  - Created `tsconfig.json` with strict mode
  - Created `next.config.ts`

- [x] **Task 1.3**: Configure Tailwind CSS
  - Created `tailwind.config.ts`
  - Created `postcss.config.mjs`
  - Set up `app/globals.css` with Tailwind directives
  - Configured dark mode support

- [x] **Task 1.4**: Create basic layout and pages
  - Created `app/layout.tsx` with header
  - Created `app/page.tsx` with homepage content
  - ESLint configured

- [x] **Task 1.5**: Setup Python environment
  - Created `ml/requirements.txt` with all ML dependencies
  - Created `ml/README.md` with setup instructions

- [x] **Task 1.6**: Create initial exploration notebook
  - Created `ml/notebooks/01_explore_data.ipynb` template

- [x] **Task 1.7**: Configure git and gitignore
  - Created comprehensive `.gitignore` for Node.js and Python

- [x] **Task 1.8**: Create documentation
  - Updated root `README.md` with architecture and quick start
  - Created `SETUP_COMPLETE.md` verification checklist

- [x] **Task 1.9**: Install dependencies and verify build
  - Ran `npm install` successfully
  - Verified `npm run build` works without errors

---

## ✅ Phase 2: Data Ingestion & Offline ML Pipeline (COMPLETED)

**Status**: ✅ Complete
**Time Invested**: ~6 hours
**Progress**: 6/6 tasks complete

### Task 2.1: Download and Inspect UAP Dataset

**Estimated Time**: 30 minutes

#### Checklist:
- [x] Download UAP/UFO sighting dataset ✅
- [x] Place dataset CSV in `ml/data/raw/` ✅
- [x] Verify dataset has required columns ✅
- [x] Update filename in exploration notebook ✅

#### 📁 Files Affected:
- `ml/data/raw/[your-dataset].csv` (new)
- `ml/notebooks/01_explore_data.ipynb` (update)

#### 💡 Dataset Sources:
```bash
# Option 1: NUFORC (National UFO Reporting Center)
# Visit: https://nuforc.org/webreports/
# Download scrubbed report database

# Option 2: Kaggle
# Search: "UFO sightings dataset"
# Example: https://www.kaggle.com/datasets/NUFORC/ufo-sightings

# Option 3: GitHub repositories
# Search: "UFO sightings CSV" or "UAP data"
```

#### 🎯 Acceptance Criteria:
- [ ] Dataset file exists in `ml/data/raw/`
- [ ] Contains minimum columns: date/time, location, lat/lon, description, shape
- [ ] At least 10,000+ records (ideally 50k-100k)
- [ ] CSV is readable by pandas

#### ✅ Verification Steps:
```bash
cd ml
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
jupyter notebook notebooks/01_explore_data.ipynb
```
Then:
1. Update the `data_path` variable with your filename
2. Run the "Load Dataset" cell
3. Verify output shows row count and columns
4. Run the inspection cells (uncomment them)
5. Document your findings in the "Summary & Next Steps" section

#### ⚠️ Common Issues:
- **Missing columns**: Some datasets use different column names (e.g., "comments" vs "description")
- **Date formats**: May need parsing with different formats
- **Encoding issues**: Try `pd.read_csv(path, encoding='utf-8')` or `encoding='latin-1'`

---

### Task 2.2: Implement Data Cleaning Script

**Estimated Time**: 2-3 hours

#### Checklist:
- [x] Create `ml/scripts/clean_data.py` ✅
- [x] Implement date/time parsing and normalization ✅
- [x] Implement location and coordinate validation ✅
- [x] Implement text cleaning ✅
- [x] Handle missing values ✅
- [x] Test script and verify output ✅ (87,458 records cleaned)

#### 📁 Files to Create:
- `ml/scripts/clean_data.py` (new - main script)

#### 📁 Files Affected:
- `ml/data/processed/cleaned_sightings.parquet` (output)

#### 💡 Implementation Guide:

**Step 1**: Create the script skeleton
```python
# ml/scripts/clean_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from dateutil import parser
import re

def load_raw_data():
    """Load the raw dataset"""
    pass

def clean_dates(df):
    """Parse and normalize dates to UTC"""
    pass

def clean_locations(df):
    """Validate and clean lat/lon"""
    pass

def clean_text(df):
    """Clean description text"""
    pass

def handle_missing_values(df):
    """Drop or impute missing data"""
    pass

def main():
    """Run the full cleaning pipeline"""
    pass

if __name__ == "__main__":
    main()
```

**Step 2**: Implement each function following the PRD requirements

**Step 3**: Add logging and error handling

#### 🎯 Acceptance Criteria:
- [ ] Script runs without errors: `python ml/scripts/clean_data.py`
- [ ] Output file created: `ml/data/processed/cleaned_sightings.parquet`
- [ ] Dates are in consistent format (ISO 8601 or UTC timestamps)
- [ ] Lat/lon are validated (within valid ranges)
- [ ] Text is cleaned (whitespace trimmed, basic normalization)
- [ ] Missing value strategy is documented
- [ ] Script logs: number of rows before/after, columns processed

#### ✅ Verification Steps:
```bash
cd ml
source .venv/bin/activate
python scripts/clean_data.py

# Then verify output:
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/cleaned_sightings.parquet')
print(f'Rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(df.head())
print(df.info())
"
```

#### 📚 Learning Resources:
- [Pandas date parsing](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
- [Handling missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Text cleaning in Python](https://realpython.com/python-string-formatting/)

---

### Task 2.3: Spatiotemporal Aggregation & Baseline Modeling

**Estimated Time**: 3-4 hours

#### Checklist:
- [x] Create notebook `ml/notebooks/02_spatiotemporal_baseline.ipynb` ✅
- [x] Implement spatial grid (lat/lon binning or hexagons) ✅
- [x] Aggregate sightings by grid cell and time window ✅
- [x] Create features (year, month, grid coordinates) ✅
- [x] Train baseline prediction model ✅
- [x] Compute cell-level anomaly scores ✅
- [x] Export results ✅

#### 📁 Files to Create:
- `ml/notebooks/02_spatiotemporal_baseline.ipynb` (new)
- `ml/data/processed/grid_time_anomalies.parquet` (output)

#### 💡 Implementation Guide:

**Spatial Gridding Options**:
1. **Simple lat/lon rounding** (easiest):
   ```python
   df['grid_lat'] = (df['latitude'] // 1.0) * 1.0  # 1° grid
   df['grid_lon'] = (df['longitude'] // 1.0) * 1.0
   ```

2. **Uber H3 hexagons** (more advanced):
   ```python
   import h3
   df['h3_cell'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], 5), axis=1)
   ```

**Temporal Aggregation**:
```python
df['year_month'] = df['date'].dt.to_period('M')
```

**Baseline Model Features**:
- Historical count (autoregressive)
- Month/season
- Year trend
- Grid cell (one-hot or embedding)
- Optional: population density (external data)

**Model Options**:
- Start simple: Linear Regression
- Better: Gradient Boosting (XGBoost, LightGBM)
- Advanced: Poisson Regression (for count data)

#### 🎯 Acceptance Criteria:
- [x] Grid system defined (document cell size/resolution) ✅
- [x] Data aggregated by (grid_cell, time_window) ✅
- [x] Baseline model trained with reasonable metrics (R² > 0.3) ✅
- [x] Anomaly scores computed (z-scores or residuals) ✅
- [x] Results saved to `grid_time_anomalies.parquet` ✅
- [x] Notebook has visualizations of: ✅
  - [x] Sightings per cell (heatmap) ✅
  - [x] Predicted vs actual counts ✅
  - [x] Top anomalous cells ✅

#### ✅ Verification Steps:
```python
# In the notebook, verify:
import pandas as pd
anomalies = pd.read_parquet('../data/processed/grid_time_anomalies.parquet')

# Should have columns like:
# - grid_lat, grid_lon (or h3_cell)
# - year_month (or time_window)
# - actual_count
# - predicted_count
# - anomaly_score

print(anomalies.head())
print(f"Total grid cells: {anomalies['grid_cell'].nunique()}")
print(f"Top anomalies:\n{anomalies.nlargest(10, 'anomaly_score')}")
```

#### 📚 Learning Resources:
- [Scikit-learn regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Time series features](https://www.kaggle.com/code/ryanholbrook/time-series-as-features)
- [Anomaly detection basics](https://scikit-learn.org/stable/modules/outlier_detection.html)

---

### Task 2.4: Text Embeddings & Clustering

**Estimated Time**: 2-3 hours

#### Checklist:
- [x] Create notebook `ml/notebooks/03_text_clusters.ipynb` ✅
- [x] Generate sentence embeddings for descriptions ✅
- [x] Experiment with clustering algorithms ✅
- [x] Choose optimal number of clusters ✅
- [x] Assign human-readable labels to clusters ✅
- [x] Save cluster assignments and labels ✅

#### 📁 Files to Create:
- `ml/notebooks/03_text_clusters.ipynb` (new)
- `ml/data/processed/cluster_labels.json` (output)
- `ml/data/processed/sightings_with_clusters.parquet` (output)

#### 💡 Implementation Guide:

**Step 1: Generate Embeddings**
```python
from sentence_transformers import SentenceTransformer

# Choose a model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality

# Generate embeddings
descriptions = df['description'].fillna('').tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)
```

**Step 2: Clustering**
```python
# Option A: KMeans (simpler, requires choosing k)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=15, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Option B: HDBSCAN (auto-determines clusters)
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
clusters = clusterer.fit_predict(embeddings)
```

**Step 3: Label Clusters**
```python
# For each cluster, extract top words/phrases
from sklearn.feature_extraction.text import TfidfVectorizer

def label_cluster(cluster_id, df, cluster_col='cluster_id'):
    cluster_texts = df[df[cluster_col] == cluster_id]['description']
    # Use TF-IDF to find characteristic words
    # Manually inspect and assign label
    return "bright_lights"  # example
```

**Suggested Cluster Labels** (adjust based on your data):
- "bright_lights"
- "triangular_craft"
- "orb_sphere"
- "fast_moving_object"
- "hovering_object"
- "formation_multiple"
- "cigar_cylinder"
- "disk_saucer"
- "flashing_lights"
- "silent_movement"

#### 🎯 Acceptance Criteria:
- [x] Embeddings generated for all descriptions ✅
- [x] Clustering algorithm applied (KMeans or HDBSCAN) ✅
- [x] Number of clusters is reasonable (10-30) ✅
- [x] Each cluster has human-readable label ✅
- [x] Cluster labels saved to JSON: `{"0": "bright_lights", "1": "triangular_craft", ...}` ✅
- [x] Original dataset updated with `cluster_id` column ✅
- [x] Notebook shows: ✅
  - [x] Cluster size distribution ✅
  - [x] Sample descriptions per cluster ✅
  - [x] Visualization (t-SNE or UMAP) ✅

#### ✅ Verification Steps:
```python
# Verify cluster labels
import json
with open('../data/processed/cluster_labels.json') as f:
    labels = json.load(f)
print(f"Number of clusters: {len(labels)}")
print(f"Labels: {labels}")

# Verify updated dataset
df = pd.read_parquet('../data/processed/sightings_with_clusters.parquet')
print(f"Cluster distribution:\n{df['cluster_id'].value_counts()}")
```

#### 📚 Learning Resources:
- [Sentence Transformers documentation](https://www.sbert.net/)
- [KMeans clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [HDBSCAN guide](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
- [Dimensionality reduction for viz](https://umap-learn.readthedocs.io/en/latest/)

---

### Task 2.5: Per-Report Anomaly Model

**Estimated Time**: 2-3 hours

#### Checklist:
- [x] Create notebook `ml/notebooks/04_per_report_anomaly.ipynb` ✅
- [x] Merge data: cleaned sightings + cell anomalies + clusters ✅
- [x] Engineer features for anomaly detection ✅
- [x] Train Isolation Forest model ✅
- [x] Generate per-report anomaly scores ✅
- [x] Export final dataset with all scores ✅

#### 📁 Files to Create:
- `ml/notebooks/04_per_report_anomaly.ipynb` (new)
- `ml/data/processed/sightings_with_scores.parquet` (output - final dataset)

#### 💡 Implementation Guide:

**Step 1: Merge Data Sources**
```python
# Load all processed data
cleaned = pd.read_parquet('../data/processed/cleaned_sightings.parquet')
clusters = pd.read_parquet('../data/processed/sightings_with_clusters.parquet')
grid_anomalies = pd.read_parquet('../data/processed/grid_time_anomalies.parquet')

# Merge (adjust keys based on your data)
df = cleaned.merge(clusters[['id', 'cluster_id']], on='id')
df = df.merge(grid_anomalies, on=['grid_lat', 'grid_lon', 'year_month'])
```

**Step 2: Feature Engineering**
```python
# Numeric features
features = []

# 1. Duration (normalize/log scale)
df['duration_log'] = np.log1p(df['duration_seconds'])
features.append('duration_log')

# 2. Time of day
df['hour'] = df['datetime'].dt.hour
features.append('hour')

# 3. Month/season
df['month'] = df['datetime'].dt.month
features.append('month')

# 4. Cell anomaly score
features.append('cell_anomaly_score')

# 5. Cluster ID (one-hot encode)
cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='cluster')
features.extend(cluster_dummies.columns.tolist())

# 6. Shape (one-hot encode)
shape_dummies = pd.get_dummies(df['shape'], prefix='shape')
features.extend(shape_dummies.columns.tolist())

# Combine all features
X = pd.concat([df[['duration_log', 'hour', 'month', 'cell_anomaly_score']],
               cluster_dummies, shape_dummies], axis=1)
```

**Step 3: Train Isolation Forest**
```python
from sklearn.ensemble import IsolationForest

# Train model
iso_forest = IsolationForest(
    contamination=0.1,  # expect 10% anomalies
    random_state=42,
    n_jobs=-1
)
anomaly_labels = iso_forest.fit_predict(X)

# Convert to scores (0-1 range)
anomaly_scores = iso_forest.score_samples(X)
# Normalize to 0-1 (higher = more anomalous)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['anomaly_score_report'] = scaler.fit_transform(
    -anomaly_scores.reshape(-1, 1)  # negate so higher is more anomalous
)
```

#### 🎯 Acceptance Criteria:
- [x] All data sources merged successfully ✅
- [x] Feature matrix created with at least 5 features ✅
- [x] Isolation Forest trained without errors ✅
- [x] Every sighting has `anomaly_score_report` (0-1 scale) ✅
- [x] Final dataset saved with columns: ✅
  - [x] All original fields (date, location, description, etc.) ✅
  - [x] `cluster_id` and `cluster_label` ✅
  - [x] `cell_anomaly_score` ✅
  - [x] `anomaly_score_report` ✅
- [x] Notebook shows: ✅
  - [x] Feature importance or top anomalous reports ✅
  - [x] Distribution of anomaly scores ✅
  - [x] Examples of high-anomaly reports ✅

#### ✅ Verification Steps:
```python
# Verify final dataset
df = pd.read_parquet('../data/processed/sightings_with_scores.parquet')

print(f"Total sightings: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nAnomaly score distribution:")
print(df['anomaly_score_report'].describe())

# Top anomalies
print("\nTop 10 most anomalous sightings:")
print(df.nlargest(10, 'anomaly_score_report')[
    ['datetime', 'location', 'anomaly_score_report', 'cluster_label', 'description']
])
```

#### 📚 Learning Resources:
- [Isolation Forest explained](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Feature engineering guide](https://www.kaggle.com/learn/feature-engineering)
- [One-hot encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

---

### Task 2.6: Update Exploration Notebook with Real Data

**Estimated Time**: 30 minutes

#### Checklist:
- [x] Uncomment all cells in `01_explore_data.ipynb` ✅
- [x] Run full notebook with actual dataset ✅
- [x] Document findings in markdown cells ✅
- [x] Generate and save key visualizations ✅
- [x] Commit notebook with outputs ✅

#### 🎯 Acceptance Criteria:
- [x] Notebook runs completely without errors ✅
- [x] All visualizations render ✅
- [x] Summary section filled out with insights ✅
- [x] Notebook saved with outputs visible ✅

---

## ✅ Phase 3: Static Data Export & Tiling (COMPLETED)

**Status**: ✅ Complete
**Time Invested**: ~3 hours
**Progress**: 2/2 tasks complete

### Task 3.1: Design Tile Format & Zoom Strategy

**Estimated Time**: 30-45 minutes

#### Checklist:
- [x] Define grid/tile bucketing strategy ✅
- [x] Design JSON schema for tiles ✅
- [x] Design metadata JSON structure ✅
- [x] Document tile naming convention ✅
- [x] Test with sample data ✅

#### 📁 Files to Create/Update:
- Document in `ml/README.md` (update)

#### 💡 Implementation Options:

**Option A: Simple Grid Buckets** (Recommended for v1)
```
Tile naming: lat{LAT}_lon{LON}.json
Example: lat37_lon-122.json (1° x 1° bucket)
```

**Option B: Mapbox-style Z/X/Y Tiles** (More complex, better for large scale)
```
Tile naming: z{ZOOM}/x{X}_y{Y}.json
Example: z5/x10_y15.json
```

**JSON Schema Example**:
```json
{
  "grid_id": "lat37_lon-122",
  "bounds": {
    "north": 38.0,
    "south": 37.0,
    "east": -121.0,
    "west": -122.0
  },
  "count": 1543,
  "sightings": [
    {
      "id": "abc123",
      "lat": 37.7749,
      "lon": -122.4194,
      "timestamp": "2020-06-15T22:30:00Z",
      "cluster_id": 3,
      "cluster_label": "bright_lights",
      "anomaly_score_report": 0.85,
      "cell_anomaly_score": 1.2,
      "shape": "light",
      "duration": 120,
      "description": "Bright light moving rapidly..."
    }
  ]
}
```

**Metadata Schema** (`cluster_labels.json`):
```json
{
  "0": "bright_lights",
  "1": "triangular_craft",
  "2": "orb_sphere"
}
```

**Metadata Schema** (`global_stats.json`):
```json
{
  "total_sightings": 87543,
  "date_range": {
    "start": "1990-01-01",
    "end": "2024-12-31"
  },
  "clusters": {
    "0": { "label": "bright_lights", "count": 23451 },
    "1": { "label": "triangular_craft", "count": 15234 }
  },
  "shapes": {
    "light": 25000,
    "triangle": 18000
  }
}
```

#### 🎯 Acceptance Criteria:
- [x] Tile bucketing strategy documented ✅
- [x] JSON schemas documented with examples ✅
- [x] Tile size reasonable (< 500KB per tile ideally) ✅
- [x] Metadata structure defined ✅

---

### Task 3.2: Implement Export Script

**Estimated Time**: 2-3 hours

#### Checklist:
- [x] Create `ml/scripts/export_tiles.py` ✅
- [x] Implement tile generation logic ✅
- [x] Implement metadata export ✅
- [x] Add progress bars and logging ✅
- [x] Test with full dataset ✅
- [x] Verify output files ✅

#### 📁 Files to Create:
- `ml/scripts/export_tiles.py` (new)

#### 📁 Output Files:
- `app/public/data/tiles/*.json` (many files)
- `app/public/data/metadata/cluster_labels.json`
- `app/public/data/metadata/global_stats.json`

#### 💡 Implementation Skeleton:
```python
# ml/scripts/export_tiles.py
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def load_processed_data():
    """Load final processed dataset"""
    return pd.read_parquet('data/processed/sightings_with_scores.parquet')

def assign_tiles(df):
    """Assign each sighting to a tile"""
    df['tile_id'] = df.apply(
        lambda row: f"lat{int(row['latitude'])}_lon{int(row['longitude'])}",
        axis=1
    )
    return df

def generate_tiles(df, output_dir):
    """Generate JSON tile files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile_id, group in tqdm(df.groupby('tile_id')):
        tile_data = {
            'grid_id': tile_id,
            'count': len(group),
            'sightings': group.to_dict('records')
        }

        with open(output_dir / f"{tile_id}.json", 'w') as f:
            json.dump(tile_data, f, default=str)

def export_metadata(df, output_dir):
    """Export cluster labels and global stats"""
    # Implementation here
    pass

def main():
    df = load_processed_data()
    df = assign_tiles(df)
    generate_tiles(df, '../app/public/data/tiles')
    export_metadata(df, '../app/public/data/metadata')

if __name__ == "__main__":
    main()
```

#### 🎯 Acceptance Criteria:
- [x] Script runs: `python ml/scripts/export_tiles.py` ✅
- [x] Tiles generated in `app/public/data/tiles/` ✅
- [x] Each tile is valid JSON ✅
- [x] Metadata files created in `app/public/data/metadata/` ✅
- [x] Script logs: ✅
  - [x] Number of tiles generated ✅
  - [x] Total sightings exported ✅
  - [x] File sizes ✅
- [x] Sample tile verified to load in browser ✅

#### ✅ Verification Steps:
```bash
cd ml
python scripts/export_tiles.py

# Check output
ls -lh ../app/public/data/tiles/ | head
cat ../app/public/data/tiles/lat37_lon-122.json | jq '.'  # if jq installed
cat ../app/public/data/metadata/cluster_labels.json
cat ../app/public/data/metadata/global_stats.json
```

---

## ⏸️ Phase 4: Core Frontend Map & Filters

**Status**: ⏸️ Pending
**Estimated Time**: 8-10 hours
**Progress**: 0/5 tasks complete

### Task 4.1: Integrate Map Library

**Estimated Time**: 1-2 hours

#### Checklist:
- [ ] Choose map library (Mapbox GL or deck.gl)
- [ ] Install dependencies
- [ ] Get API key (if needed)
- [ ] Create MapView component
- [ ] Test basic map rendering

#### 📁 Files to Create:
- `app/components/MapView.tsx` (new)
- `app/.env.local` (new - for API keys)

#### 💡 Implementation Guide:

**Option A: Mapbox GL JS** (Recommended - easier)
```bash
npm install mapbox-gl @types/mapbox-gl
npm install react-map-gl
```

**Option B: deck.gl** (More powerful for large datasets)
```bash
npm install deck.gl @deck.gl/react
```

**Mapbox Component Example**:
```tsx
// app/components/MapView.tsx
'use client';

import { useRef, useEffect } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

export default function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);

  useEffect(() => {
    if (map.current) return; // Initialize map only once

    map.current = new mapboxgl.Map({
      container: mapContainer.current!,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-95.7129, 37.0902], // Center of US
      zoom: 4
    });
  }, []);

  return (
    <div
      ref={mapContainer}
      className="w-full h-screen"
    />
  );
}
```

**Get Mapbox Token**:
1. Sign up at https://account.mapbox.com/
2. Create access token
3. Add to `.env.local`:
```
NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here
```

#### 🎯 Acceptance Criteria:
- [ ] Map library installed
- [ ] MapView component created
- [ ] Map renders on page
- [ ] Can pan and zoom
- [ ] No console errors

#### ✅ Verification:
Update `app/page.tsx` to include map:
```tsx
import MapView from '@/components/MapView';

export default function Home() {
  return <MapView />;
}
```

Run `npm run dev` and check http://localhost:3000

---

### Task 4.2: Load and Render Static Tiles

**Estimated Time**: 2-3 hours

#### Checklist:
- [ ] Create tile loading utility
- [ ] Implement viewport-based tile selection
- [ ] Load tile JSON files
- [ ] Render points on map
- [ ] Test with multiple tiles

#### 📁 Files to Create:
- `app/lib/tileLoader.ts` (new - utility)
- `app/types/sighting.ts` (new - TypeScript types)

#### 📁 Files to Update:
- `app/components/MapView.tsx` (update - add data layer)

#### 💡 Implementation:

**TypeScript Types**:
```tsx
// app/types/sighting.ts
export interface Sighting {
  id: string;
  lat: number;
  lon: number;
  timestamp: string;
  cluster_id: number;
  cluster_label: string;
  anomaly_score_report: number;
  cell_anomaly_score: number;
  shape: string;
  duration: number;
  description: string;
}

export interface Tile {
  grid_id: string;
  count: number;
  sightings: Sighting[];
}
```

**Tile Loader**:
```tsx
// app/lib/tileLoader.ts
import { Tile } from '@/types/sighting';

export async function loadTile(tileId: string): Promise<Tile | null> {
  try {
    const response = await fetch(`/data/tiles/${tileId}.json`);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error(`Failed to load tile ${tileId}:`, error);
    return null;
  }
}

export function getTilesInView(bounds: {
  north: number;
  south: number;
  east: number;
  west: number;
}): string[] {
  const tiles: string[] = [];

  for (let lat = Math.floor(bounds.south); lat <= Math.ceil(bounds.north); lat++) {
    for (let lon = Math.floor(bounds.west); lon <= Math.ceil(bounds.east); lon++) {
      tiles.push(`lat${lat}_lon${lon}`);
    }
  }

  return tiles;
}
```

#### 🎯 Acceptance Criteria:
- [ ] Tiles load based on map viewport
- [ ] Points render on map
- [ ] Multiple tiles load correctly
- [ ] Map updates when panning

---

### Task 4.3: Add Basic Filters (Cluster, Anomaly Threshold)

**Estimated Time**: 2-3 hours

#### Checklist:
- [ ] Create FiltersPanel component
- [ ] Load cluster labels from metadata
- [ ] Implement cluster multi-select
- [ ] Implement anomaly score slider
- [ ] Connect filters to map rendering
- [ ] Test filter combinations

#### 📁 Files to Create:
- `app/components/FiltersPanel.tsx` (new)
- `app/hooks/useFilters.ts` (new - custom hook)

---

### Task 4.4: Implement Time Slider

**Estimated Time**: 2-3 hours

#### Checklist:
- [ ] Create TimeSlider component
- [ ] Load date range from metadata
- [ ] Implement date range selection
- [ ] Add play/pause animation (optional)
- [ ] Filter sightings by time
- [ ] Test time filtering

#### 📁 Files to Create:
- `app/components/TimeSlider.tsx` (new)

---

### Task 4.5: Optimize Map Performance

**Estimated Time**: 1-2 hours

#### Checklist:
- [ ] Implement clustering for low zoom levels
- [ ] Add debouncing for tile loading
- [ ] Optimize re-renders
- [ ] Test with full dataset
- [ ] Monitor performance metrics

---

## ⏸️ Phase 5: Detail Panel, Insights & Polish

**Status**: ⏸️ Pending
**Estimated Time**: 4-6 hours
**Progress**: 0/4 tasks complete

### Task 5.1: Sighting Detail Panel

**Estimated Time**: 2-3 hours

#### Checklist:
- [ ] Create SightingDetailPanel component
- [ ] Implement click handler on map points
- [ ] Display sighting details
- [ ] Show anomaly score explanation
- [ ] Add close functionality
- [ ] Add slide-in animation

#### 📁 Files to Create:
- `app/components/SightingDetailPanel.tsx` (new)

---

### Task 5.2: Basic Insights Screen (Stretch)

**Estimated Time**: 2-3 hours

#### Checklist:
- [ ] Create insights page
- [ ] Add charts for temporal trends
- [ ] Add charts for cluster distribution
- [ ] Load global stats
- [ ] Add navigation

---

### Task 5.3: UI Polish

**Estimated Time**: 1-2 hours

#### Checklist:
- [ ] Refine color palette
- [ ] Add loading states
- [ ] Add error states
- [ ] Improve mobile responsiveness
- [ ] Add legend for anomaly colors
- [ ] Polish typography and spacing

---

### Task 5.4: Bookmarkable State (Stretch)

**Estimated Time**: 1 hour

#### Checklist:
- [ ] Encode filters in URL
- [ ] Encode map viewport in URL
- [ ] Parse URL on load
- [ ] Test sharing links

---

## ⏸️ Phase 6: Testing, Deployment & Documentation

**Status**: ⏸️ Pending
**Estimated Time**: 4-6 hours
**Progress**: 0/4 tasks complete

### Task 6.1: Basic Frontend Tests

**Estimated Time**: 2-3 hours

#### Checklist:
- [ ] Setup Jest and React Testing Library
- [ ] Write tests for FiltersPanel
- [ ] Write tests for MapView (basic)
- [ ] Write tests for data loading
- [ ] Run tests: `npm test`

---

### Task 6.2: Manual QA Checklist

**Estimated Time**: 1-2 hours

#### Checklist:
- [ ] Create QA_CHECKLIST.md
- [ ] Test all filters
- [ ] Test time slider
- [ ] Test detail panel
- [ ] Test on different browsers
- [ ] Test on mobile
- [ ] Fix identified bugs

---

### Task 6.3: Deployment to Vercel

**Estimated Time**: 30-60 minutes

#### Checklist:
- [ ] Create Vercel account
- [ ] Connect GitHub repo
- [ ] Configure environment variables
- [ ] Deploy to production
- [ ] Test deployed app
- [ ] Configure custom domain (optional)

**Deployment Steps**:
```bash
# Option 1: Vercel CLI
npm i -g vercel
vercel login
vercel

# Option 2: Vercel Dashboard
# 1. Visit vercel.com
# 2. Import GitHub repo
# 3. Configure build settings
# 4. Deploy
```

---

### Task 6.4: Final Documentation

**Estimated Time**: 1-2 hours

#### Checklist:
- [ ] Update README with deployment URL
- [ ] Add architecture diagram
- [ ] Document data pipeline commands
- [ ] Add screenshots to README
- [ ] Create CHANGELOG.md
- [ ] Update ml/README with final notes
- [ ] Add contributor guidelines

---

## 📋 Quick Reference Commands

### Frontend Development
```bash
cd app
npm run dev        # Start dev server
npm run build      # Production build
npm run lint       # Run linter
npm test           # Run tests
```

### ML Pipeline
```bash
cd ml
source .venv/bin/activate                      # Activate venv
jupyter notebook                                # Start Jupyter
python scripts/clean_data.py                   # Clean data
python scripts/export_tiles.py                 # Export tiles
```

### Git Workflow
```bash
git status                                      # Check status
git add .                                       # Stage all
git commit -m "feat: add feature description"  # Commit
git push origin main                            # Push
```

---

## 🎯 Milestones

- **M1**: Basic structure & data pipeline started → ✅ COMPLETE (Phase 2)
- **M2**: ML pipeline outputs ready → ✅ COMPLETE (Phase 3)
- **M3 (Current)**: Minimal map rendering → Target: Complete Task 4.1-4.2
- **M4 (Demo-Ready v1)**: Filters, time slider & detail panel → Target: Complete Phase 4-5
- **M5 (Production)**: Polish, insights & deployment → Target: Complete Phase 6

---

## 📊 Time Tracking

| Phase | Estimated | Actual | Status |
|-------|-----------|--------|--------|
| Phase 1 | 1h | 1h | ✅ Complete |
| Phase 2 | 10h | 6h | ✅ Complete |
| Phase 3 | 3h | 3h | ✅ Complete |
| Phase 4 | 10h | - | ⏸️ Pending |
| Phase 5 | 6h | - | ⏸️ Pending |
| Phase 6 | 6h | - | ⏸️ Pending |
| **Total** | **36h** | **10h** | **28% Complete** |

---

## ⚠️ Common Issues & Solutions

### ML Pipeline Issues

**Issue**: Out of memory when processing large datasets
**Solution**: Process in chunks using `pd.read_csv(chunksize=10000)`

**Issue**: Embeddings taking too long
**Solution**: Use smaller model or reduce dataset size for testing

**Issue**: Clusters not meaningful
**Solution**: Adjust number of clusters, try HDBSCAN, or improve text preprocessing

### Frontend Issues

**Issue**: Map not rendering
**Solution**: Check API token, check console for errors, verify mapbox-gl CSS imported

**Issue**: Tiles not loading
**Solution**: Check file paths, verify JSON is valid, check browser network tab

**Issue**: App slow with many points
**Solution**: Implement clustering, limit points per zoom level, use WebGL rendering

---

## 🎓 Learning Resources

- **Next.js**: https://nextjs.org/docs
- **Mapbox GL**: https://docs.mapbox.com/mapbox-gl-js/
- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Sentence Transformers**: https://www.sbert.net/
- **React Testing Library**: https://testing-library.com/react

---

**Last Updated**: 2025-11-28
**Current Phase**: Phase 4 - Frontend Map & Filters
**Next Task**: Task 4.1 - Integrate Map Library

Good luck building! 🚀
