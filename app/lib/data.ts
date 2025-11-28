/**
 * Data loading utilities for UAP sightings
 */

export interface Sighting {
  id: string;
  lat: number;
  lon: number;
  timestamp: string | null;
  year: number | null;
  month: number | null;
  shape: string;
  duration: number;
  description: string;
  location: string;
  cluster_id?: number;
  cluster_label?: string;
  anomaly_score?: number;
  is_anomaly?: boolean;
}

export interface Tile {
  grid_id: string;
  count: number;
  center: {
    lat: number;
    lon: number;
  };
  sightings: Sighting[];
}

export interface TileIndex {
  tiles: Record<string, number>; // grid_id -> count
  total_tiles: number;
  total_sightings: number;
}

export interface ClusterLabels {
  labels: Record<string, string>; // cluster_id -> label
  counts: Record<string, number>; // cluster_id -> count
}

export interface GlobalStats {
  total_sightings: number;
  total_tiles: number;
  date_range: {
    min_year: number;
    max_year: number;
    min_date: string;
    max_date: string;
  };
  geographic: {
    countries: number;
    grid_cells: number;
    lat_range: [number, number];
    lon_range: [number, number];
  };
  shapes: Record<string, number>;
  countries: Record<string, number>;
  clusters: {
    total_clusters: number;
    distribution: Record<string, number>;
  };
  anomalies: {
    mean_score: number;
    median_score: number;
    high_anomaly_count: number;
  };
  exported_at: string;
}

/**
 * Load the tile index
 */
export async function loadTileIndex(): Promise<TileIndex> {
  const response = await fetch('/data/index.json');
  if (!response.ok) {
    throw new Error('Failed to load tile index');
  }
  return response.json();
}

/**
 * Load a specific tile by grid_id
 */
export async function loadTile(gridId: string): Promise<Tile> {
  const response = await fetch(`/data/tiles/${gridId}.json`);
  if (!response.ok) {
    throw new Error(`Failed to load tile: ${gridId}`);
  }
  return response.json();
}

/**
 * Load cluster labels
 */
export async function loadClusterLabels(): Promise<ClusterLabels> {
  const response = await fetch('/data/metadata/cluster_labels.json');
  if (!response.ok) {
    throw new Error('Failed to load cluster labels');
  }
  return response.json();
}

/**
 * Load global statistics
 */
export async function loadGlobalStats(): Promise<GlobalStats> {
  const response = await fetch('/data/metadata/global_stats.json');
  if (!response.ok) {
    throw new Error('Failed to load global stats');
  }
  return response.json();
}

/**
 * Get visible tiles based on map bounds
 * Returns grid IDs that intersect with the given bounds
 */
export function getVisibleTiles(
  bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  },
  tileIndex: TileIndex
): string[] {
  const visibleTiles: string[] = [];

  // Iterate through all tiles and check if they intersect with bounds
  for (const gridId of Object.keys(tileIndex.tiles)) {
    // Parse grid_id: "latXX_lonYY"
    const match = gridId.match(/lat(-?\d+)_lon(-?\d+)/);
    if (!match) continue;

    const gridLat = parseInt(match[1]);
    const gridLon = parseInt(match[2]);

    // Grid cell spans from (gridLat, gridLon) to (gridLat+1, gridLon+1)
    const cellNorth = gridLat + 1;
    const cellSouth = gridLat;
    const cellEast = gridLon + 1;
    const cellWest = gridLon;

    // Check if cell intersects with bounds
    const intersects =
      cellNorth >= bounds.south &&
      cellSouth <= bounds.north &&
      cellEast >= bounds.west &&
      cellWest <= bounds.east;

    if (intersects) {
      visibleTiles.push(gridId);
    }
  }

  return visibleTiles;
}

/**
 * Load multiple tiles in parallel
 */
export async function loadTiles(gridIds: string[]): Promise<Tile[]> {
  const promises = gridIds.map((id) => loadTile(id));
  return Promise.all(promises);
}
