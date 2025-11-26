# UAP Explorer

A web application that visualizes UAP/UFO sighting reports on an interactive map and uses machine learning to highlight spatiotemporal anomalies and narrative patterns in the data. All heavy ML work is done offline; the deployed app reads precomputed JSON "tiles" and metadata, so it feels fast and responsive while remaining simple to host (static/serverless).

## 🎯 What This Project Does

- **Interactive Map Visualization**: Explore tens of thousands of UAP sighting reports on an interactive world map
- **ML-Powered Anomaly Detection**: Identify statistically unusual reports and regions using spatiotemporal analysis
- **Pattern Discovery**: Cluster textual descriptions into phenomenology categories (lights, triangles, orbs, etc.)
- **Temporal Exploration**: Time slider to see how sightings evolve over time
- **Fast & Lightweight**: Static/serverless architecture with precomputed ML results

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           Offline ML Pipeline (Python)          │
│  • Data cleaning & normalization                │
│  • Spatiotemporal aggregation                   │
│  • Text embeddings & clustering                 │
│  • Anomaly detection (Isolation Forest)         │
│  • Export to static JSON tiles                  │
└──────────────────┬──────────────────────────────┘
                   │
                   │ JSON files
                   ↓
┌─────────────────────────────────────────────────┐
│         Frontend (Next.js + Mapbox/deck.gl)     │
│  • Interactive map with clustering              │
│  • Filters (time, anomaly score, clusters)      │
│  • Detail panels for individual sightings       │
│  • Deployed on Vercel (static/serverless)       │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ (for the frontend)
- **Python** 3.9+ (for the ML pipeline)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Aerial-Anomaly-Explorer.git
cd Aerial-Anomaly-Explorer
```

### 2. Setup the Frontend (Next.js)

```bash
cd app
npm install
npm run dev
```

The app will be available at `http://localhost:3000`

### 3. Setup the ML Pipeline (Python)

```bash
cd ml
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See [ml/README.md](./ml/README.md) for detailed instructions on running the ML pipeline.

## 📁 Project Structure

```
Aerial-Anomaly-Explorer/
├── app/                      # Next.js frontend application
│   ├── app/                  # Next.js app directory
│   ├── components/           # React components
│   ├── public/               # Static assets
│   │   └── data/            # ML-generated JSON tiles
│   ├── package.json
│   └── tsconfig.json
├── ml/                       # Python ML pipeline
│   ├── data/                # Raw and processed data
│   ├── notebooks/           # Jupyter notebooks for exploration
│   ├── scripts/             # Data processing scripts
│   ├── requirements.txt     # Python dependencies
│   └── README.md
├── Project_summary_PRD.md   # Detailed project requirements
└── README.md                # This file
```

## 🛠️ Tech Stack

### Frontend
- **Next.js 15** (React, TypeScript)
- **Tailwind CSS** (styling)
- **Mapbox GL JS / deck.gl** (map visualization)
- **Vercel** (deployment)

### ML Pipeline
- **pandas, numpy** (data processing)
- **scikit-learn** (anomaly detection)
- **sentence-transformers** (text embeddings)
- **HDBSCAN/KMeans** (clustering)
- **Jupyter** (exploration & analysis)

## 📊 Data Pipeline

1. **Data Ingestion**: Load UAP/UFO sighting datasets (e.g., NUFORC format)
2. **Cleaning**: Normalize dates, locations, and text; handle missing data
3. **Spatiotemporal Analysis**: Grid-based aggregation and baseline modeling
4. **Text Clustering**: Embed descriptions and cluster into categories
5. **Anomaly Scoring**: Per-report anomaly detection using Isolation Forest
6. **Export**: Generate static JSON tiles optimized for frontend loading

## 🎓 Use Cases

- **Data Exploration**: Understand patterns in UAP reporting over time and space
- **Portfolio Project**: Demonstrate ML, full-stack, and system design skills
- **Research**: Rigorous, data-driven analysis of anomaly reporting patterns
- **Learning**: Study applied ML, data visualization, and serverless architecture

## 🧪 Development Roadmap

See [Project_summary_PRD.md](./Project_summary_PRD.md) for the complete implementation roadmap, including:
- Phase 1: Repo Setup & Scaffolding ✅
- Phase 2: Data Ingestion & Offline ML Pipeline
- Phase 3: Static Data Export & Tiling
- Phase 4: Core Frontend Map & Filters
- Phase 5: Detail Panel, Insights & Polish
- Phase 6: Testing, Deployment & Documentation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please read the PRD document to understand the project goals and architecture before contributing.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and demonstration purposes. Anomaly scores reflect patterns in human reporting data, not claims about underlying phenomena.
