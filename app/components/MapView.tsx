'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import mapboxgl from 'mapbox-gl';
import {
  loadTileIndex,
  loadGlobalStats,
  getVisibleTiles,
  loadTiles,
  type TileIndex,
  type GlobalStats,
  type Sighting,
} from '@/lib/data';

// Mapbox access token
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

export default function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const loadedTilesRef = useRef<Set<string>>(new Set());
  const allFeaturesRef = useRef<GeoJSON.Feature[]>([]);

  const [lng, setLng] = useState(-95.7);
  const [lat, setLat] = useState(37.1);
  const [zoom, setZoom] = useState(3);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [tileIndex, setTileIndex] = useState<TileIndex | null>(null);
  const [globalStats, setGlobalStats] = useState<GlobalStats | null>(null);
  const [sightingCount, setSightingCount] = useState(0);

  // Load metadata on mount
  useEffect(() => {
    async function loadMetadata() {
      try {
        const [index, stats] = await Promise.all([
          loadTileIndex(),
          loadGlobalStats(),
        ]);
        setTileIndex(index);
        setGlobalStats(stats);
        console.log('Loaded metadata:', { totalTiles: index.total_tiles });
      } catch (err) {
        console.error('Failed to load metadata:', err);
        setError('Failed to load data. Please ensure tiles are generated.');
      }
    }
    loadMetadata();
  }, []);

  // Load visible tiles and update map
  const loadVisibleTilesData = useCallback(async () => {
    if (!map.current || !tileIndex) {
      console.log('Cannot load tiles:', { hasMap: !!map.current, hasTileIndex: !!tileIndex });
      return;
    }

    const bounds = map.current.getBounds();
    const visibleGridIds = getVisibleTiles(
      {
        north: bounds.getNorth(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        west: bounds.getWest(),
      },
      tileIndex
    );

    // Filter out already loaded tiles
    const newGridIds = visibleGridIds.filter((id) => !loadedTilesRef.current.has(id));

    if (newGridIds.length === 0) return;

    // Limit tiles to load at once
    const tilesToLoad = newGridIds.slice(0, 30);

    console.log(`Loading ${tilesToLoad.length} tiles...`);

    try {
      const tiles = await loadTiles(tilesToLoad);

      // Convert tiles to GeoJSON features
      const features: GeoJSON.Feature[] = [];
      tiles.forEach((tile) => {
        tile.sightings.forEach((sighting) => {
          features.push({
            type: 'Feature',
            geometry: {
              type: 'Point',
              coordinates: [sighting.lon, sighting.lat],
            },
            properties: {
              id: sighting.id,
              location: sighting.location,
              timestamp: sighting.timestamp,
              shape: sighting.shape,
              cluster_label: sighting.cluster_label || 'unknown',
              anomaly_score: sighting.anomaly_score || 0,
              is_anomaly: sighting.is_anomaly || false,
              description: sighting.description.substring(0, 200),
            },
          });
        });
      });

      // Add features to accumulated list
      allFeaturesRef.current = [...allFeaturesRef.current, ...features];

      // Update the map source
      const source = map.current?.getSource('sightings') as mapboxgl.GeoJSONSource;
      if (source) {
        source.setData({
          type: 'FeatureCollection',
          features: allFeaturesRef.current,
        });

        setSightingCount(allFeaturesRef.current.length);
        console.log(`Updated map with ${allFeaturesRef.current.length} total points (added ${features.length} new)`);
      } else {
        console.error('Source not found!');
      }

      // Mark tiles as loaded
      tilesToLoad.forEach((id) => loadedTilesRef.current.add(id));

      console.log(`Loaded ${features.length} sightings. Total: ${allFeaturesRef.current.length}`);
    } catch (err) {
      console.error('Failed to load tiles:', err);
    }
  }, [tileIndex]);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    if (!MAPBOX_TOKEN) {
      setError('Mapbox token not configured. Please add NEXT_PUBLIC_MAPBOX_TOKEN to .env.local');
      setLoading(false);
      return;
    }

    let mounted = true;

    try {
      mapboxgl.accessToken = MAPBOX_TOKEN;

      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/dark-v11',
        center: [lng, lat],
        zoom: zoom,
        attributionControl: true,
      });

      // Add navigation controls
      map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

      // Update coordinates on move
      map.current.on('move', () => {
        if (!map.current || !mounted) return;
        setLng(parseFloat(map.current.getCenter().lng.toFixed(4)));
        setLat(parseFloat(map.current.getCenter().lat.toFixed(4)));
        setZoom(parseFloat(map.current.getZoom().toFixed(2)));
      });

      // Map loaded - add data layers
      map.current.on('load', () => {
        if (!mounted || !map.current) return;

        console.log('Map loaded, adding layers...');

        // Add empty GeoJSON source
        map.current.addSource('sightings', {
          type: 'geojson',
          data: {
            type: 'FeatureCollection',
            features: [],
          },
          cluster: true,
          clusterMaxZoom: 14,
          clusterRadius: 50,
        });

        // Cluster circle layer
        map.current.addLayer({
          id: 'clusters',
          type: 'circle',
          source: 'sightings',
          filter: ['has', 'point_count'],
          paint: {
            'circle-color': [
              'step',
              ['get', 'point_count'],
              '#51bbd6', 100,
              '#f1f075', 500,
              '#f28cb1', 1000,
              '#f28cb1'
            ],
            'circle-radius': [
              'step',
              ['get', 'point_count'],
              20, 100,
              30, 500,
              40
            ],
            'circle-opacity': 0.8,
          },
        });

        // Cluster count layer
        map.current.addLayer({
          id: 'cluster-count',
          type: 'symbol',
          source: 'sightings',
          filter: ['has', 'point_count'],
          layout: {
            'text-field': '{point_count_abbreviated}',
            'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
            'text-size': 12,
          },
          paint: {
            'text-color': '#ffffff',
          },
        });

        // Individual points layer (colored by anomaly score)
        map.current.addLayer({
          id: 'unclustered-point',
          type: 'circle',
          source: 'sightings',
          filter: ['!', ['has', 'point_count']],
          paint: {
            'circle-color': [
              'step',
              ['get', 'anomaly_score'],
              '#3b82f6', 0.5,  // blue for normal
              '#f59e0b', 0.7,  // orange for medium
              '#ef4444'        // red for high
            ],
            'circle-radius': 6,
            'circle-stroke-width': 1,
            'circle-stroke-color': '#fff',
            'circle-opacity': 0.8,
          },
        });

        // Click on cluster to zoom
        map.current.on('click', 'clusters', (e) => {
          if (!map.current) return;
          const features = map.current.queryRenderedFeatures(e.point, {
            layers: ['clusters'],
          });
          const clusterId = features[0]?.properties?.cluster_id;
          const source = map.current.getSource('sightings') as mapboxgl.GeoJSONSource;

          source.getClusterExpansionZoom(clusterId, (err, zoom) => {
            if (err || !map.current) return;
            const coordinates = (features[0].geometry as GeoJSON.Point).coordinates;
            map.current.easeTo({
              center: [coordinates[0], coordinates[1]],
              zoom: zoom,
            });
          });
        });

        // Show popup on unclustered point click
        map.current.on('click', 'unclustered-point', (e) => {
          if (!map.current || !e.features || !e.features[0]) return;

          const coordinates = (e.features[0].geometry as GeoJSON.Point).coordinates.slice() as [number, number];
          const props = e.features[0].properties;

          new mapboxgl.Popup()
            .setLngLat(coordinates)
            .setHTML(`
              <div class="text-sm p-2">
                <div class="font-bold mb-1">${props?.location || 'Unknown'}</div>
                <div class="text-xs text-gray-600 mb-2">${props?.timestamp || 'Unknown date'}</div>
                <div class="mb-1"><strong>Shape:</strong> ${props?.shape}</div>
                ${props?.cluster_label ? `<div class="mb-1"><strong>Type:</strong> ${props.cluster_label.replace(/_/g, ' ')}</div>` : ''}
                <div class="mb-1"><strong>Anomaly:</strong> ${(props?.anomaly_score || 0).toFixed(3)}</div>
                <div class="text-xs mt-2 max-w-xs">${props?.description || ''}...</div>
              </div>
            `)
            .addTo(map.current);
        });

        // Change cursor on hover
        map.current.on('mouseenter', 'clusters', () => {
          if (map.current) map.current.getCanvas().style.cursor = 'pointer';
        });
        map.current.on('mouseleave', 'clusters', () => {
          if (map.current) map.current.getCanvas().style.cursor = '';
        });
        map.current.on('mouseenter', 'unclustered-point', () => {
          if (map.current) map.current.getCanvas().style.cursor = 'pointer';
        });
        map.current.on('mouseleave', 'unclustered-point', () => {
          if (map.current) map.current.getCanvas().style.cursor = '';
        });

        // Load tiles when map moves
        map.current.on('moveend', loadVisibleTilesData);

        setLoading(false);

        // Load initial tiles
        setTimeout(loadVisibleTilesData, 500);
      });

      // Error handler
      map.current.on('error', (e) => {
        if (!mounted) return;
        console.error('Map error:', e);
        setError(`Map error: ${e.error?.message || 'Unknown error'}`);
        setLoading(false);
      });

    } catch (err) {
      console.error('Error creating map:', err);
      if (mounted) {
        setError(`Failed to initialize map: ${err}`);
        setLoading(false);
      }
    }

    return () => {
      mounted = false;
      map.current?.remove();
      map.current = null;
    };
  }, []); // Only run once

  // Load tiles when tileIndex becomes available
  useEffect(() => {
    if (tileIndex && map.current && !loading) {
      console.log('TileIndex available, loading initial tiles...');
      loadVisibleTilesData();
    }
  }, [tileIndex, loading]);

  return (
    <div className="relative w-full h-full">
      {/* Map Container */}
      <div ref={mapContainer} className="absolute inset-0" />

      {/* Stats Overlay */}
      <div className="absolute top-4 left-4 bg-gray-900/90 text-white px-4 py-3 rounded-lg text-sm font-mono backdrop-blur-sm border border-gray-700 space-y-1">
        <div className="font-bold text-blue-400 mb-2">UAP Explorer</div>
        {globalStats && (
          <>
            <div>Total: {globalStats.total_sightings.toLocaleString()}</div>
            <div>Range: {globalStats.date_range.min_year} - {globalStats.date_range.max_year}</div>
            <div className="border-t border-gray-700 pt-1 mt-1">Loaded: {sightingCount.toLocaleString()}</div>
          </>
        )}
        <div className="text-xs text-gray-400 border-t border-gray-700 pt-1 mt-1">
          <div>Lng: {lng} | Lat: {lat}</div>
          <div>Zoom: {zoom}</div>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-900/90 text-white px-4 py-3 rounded-lg text-xs backdrop-blur-sm border border-gray-700">
        <div className="font-bold mb-2">Anomaly Score</div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full bg-blue-500 border border-white"></div>
          <span>Normal (&lt; 0.5)</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full bg-orange-500 border border-white"></div>
          <span>Medium (0.5-0.7)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500 border border-white"></div>
          <span>High (&gt; 0.7)</span>
        </div>
        <div className="text-gray-400 text-xs mt-2 pt-2 border-t border-gray-700">
          Zoom in to see individual points
        </div>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 backdrop-blur-sm">
          <div className="text-white text-lg">Loading map...</div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/90 backdrop-blur-sm">
          <div className="bg-red-900/50 border border-red-700 text-white px-6 py-4 rounded-lg max-w-md">
            <div className="font-bold mb-2">Error</div>
            <div>{error}</div>
          </div>
        </div>
      )}
    </div>
  );
}
