# UAP Explorer - Frontend

Interactive web application for visualizing UAP/UFO sighting reports with ML-powered anomaly detection.

## Features

- **Interactive Map**: Mapbox GL-powered map showing 87,000+ UAP sightings
- **Anomaly Detection**: Color-coded markers based on ML anomaly scores
- **Smart Tile Loading**: Efficient loading of data based on map viewport
- **Cluster Information**: Each sighting categorized by phenomenology type
- **Detailed Popups**: Click any marker to see full sighting details

## Prerequisites

- Node.js 18+
- npm or yarn
- Mapbox account (free tier works)

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Get a Mapbox Access Token

1. Sign up for a free account at [mapbox.com](https://account.mapbox.com/auth/signup/)
2. Navigate to [Access Tokens](https://account.mapbox.com/access-tokens/)
3. Copy your default public token (starts with `pk.`)

### 3. Configure Environment

```bash
# Copy the example environment file
cp .env.local.example .env.local

# Edit .env.local and add your Mapbox token
# NEXT_PUBLIC_MAPBOX_TOKEN=pk.your_actual_token_here
```

### 4. Generate Data Tiles

Before running the frontend, you need to generate the data tiles from the ML pipeline:

```bash
cd ../ml
source .venv/bin/activate
python scripts/export_tiles.py
```

This will create:
- `app/public/data/tiles/` - 2,387 JSON tile files
- `app/public/data/metadata/` - Cluster labels and global statistics
- `app/public/data/index.json` - Tile index for efficient loading

### 5. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
app/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout with header
│   ├── page.tsx           # Home page with map
│   └── globals.css        # Global styles
├── components/
│   └── MapView.tsx        # Main map component
├── lib/
│   └── data.ts            # Data loading utilities
├── public/
│   └── data/              # Generated tile data
│       ├── tiles/         # Per-grid-cell JSON files
│       ├── metadata/      # Cluster labels, stats
│       └── index.json     # Tile index
├── .env.local.example     # Environment template
└── package.json           # Dependencies
```

## Map Controls

- **Pan**: Click and drag
- **Zoom**: Scroll wheel or +/- buttons
- **Inspect**: Click any marker to see sighting details
- **Navigate**: Use the compass/zoom controls in top-right

## Marker Colors

- 🔵 **Blue**: Normal sightings (anomaly score 0-0.5)
- 🟠 **Orange**: Medium anomaly (anomaly score 0.5-0.7)
- 🔴 **Red**: High anomaly (anomaly score 0.7+)

## Data Loading

The app uses a smart tile-based loading system:

1. **Index Load**: Loads `index.json` with all tile IDs and counts
2. **Viewport Detection**: Calculates which tiles are visible
3. **Lazy Loading**: Loads only visible tiles (max 20 at a time)
4. **Caching**: Already-loaded tiles aren't reloaded

This ensures:
- Fast initial load
- Smooth panning/zooming
- Minimal network usage
- Works with 87,000+ markers

## Building for Production

```bash
npm run build
npm run start
```

Or deploy to Vercel:

```bash
vercel deploy
```

## Troubleshooting

### "Mapbox token not configured"
- Make sure you created `.env.local` from `.env.local.example`
- Add your token: `NEXT_PUBLIC_MAPBOX_TOKEN=pk.your_token`
- Restart the dev server

### "Failed to load data"
- Run the export script: `cd ../ml && python scripts/export_tiles.py`
- Check that `app/public/data/index.json` exists
- Check browser console for specific errors

### Map not rendering
- Check browser console for errors
- Verify Mapbox token is valid
- Try clearing browser cache

## Performance Tips

- Zoom in to reduce number of visible markers
- The app automatically limits tiles loaded at once
- High-density areas may take a moment to render
- Performance is best on desktop browsers

## Tech Stack

- **Next.js 15**: React framework with App Router
- **Mapbox GL JS**: Interactive map rendering
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Styling
- **React Hooks**: State management

## License

See root LICENSE file.
