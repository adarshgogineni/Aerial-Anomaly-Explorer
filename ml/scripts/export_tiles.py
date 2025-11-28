"""
UAP Explorer - Data Export Script

This script exports the ML-processed UAP sightings data into JSON tiles
optimized for frontend consumption.

Steps:
1. Load final ML dataset (sightings_with_scores.parquet)
2. Generate JSON tile files (one per grid cell)
3. Export metadata files (cluster labels, global stats)
4. Create index file for frontend

Output Structure:
- app/public/data/tiles/{grid_id}.json - Individual grid cell data
- app/public/data/metadata/cluster_labels.json - Cluster ID to label mapping
- app/public/data/metadata/global_stats.json - Dataset statistics
- app/public/data/index.json - Grid index with counts

Usage:
    python scripts/export_tiles.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')


def load_ml_data():
    """Load the final ML-processed dataset."""
    print("=" * 60)
    print("UAP EXPLORER - DATA EXPORT PIPELINE")
    print("=" * 60)
    print("\n[1/5] Loading ML-processed data...")

    data_path = Path('data/processed/sightings_with_scores.parquet')

    if not data_path.exists():
        raise FileNotFoundError(
            f"ML dataset not found at {data_path}\n"
            "Please run notebooks 02, 03, and 04 first."
        )

    df = pd.read_parquet(data_path)

    print(f"  ✓ Loaded {len(df):,} sightings")
    print(f"  ✓ Columns: {len(df.columns)}")
    print(f"  ✓ Grid cells: {df['grid_id'].nunique()}")
    print(f"  ✓ Date range: {df['year'].min():.0f} - {df['year'].max():.0f}")

    return df


def prepare_sighting_record(row: pd.Series) -> Dict[str, Any]:
    """Convert a DataFrame row into a frontend-ready sighting record."""
    # Convert datetime to ISO format string
    timestamp = row['datetime']
    if pd.notna(timestamp):
        timestamp_str = timestamp.isoformat()
    else:
        timestamp_str = None

    # Build sighting record with only essential fields
    record = {
        'id': str(row['id']),
        'lat': round(float(row['latitude']), 6),
        'lon': round(float(row['longitude']), 6),
        'timestamp': timestamp_str,
        'year': int(row['year']) if pd.notna(row['year']) else None,
        'month': int(row['month']) if pd.notna(row['month']) else None,
        'shape': str(row['shape']),
        'duration': int(row['duration_seconds']) if pd.notna(row['duration_seconds']) else 0,
        'description': str(row['description'])[:500],  # Limit description length
        'location': str(row['location']),
    }

    # Add ML results
    if 'cluster_id' in row.index and pd.notna(row['cluster_id']):
        record['cluster_id'] = int(row['cluster_id'])

    if 'cluster_label' in row.index and pd.notna(row['cluster_label']):
        record['cluster_label'] = str(row['cluster_label'])

    if 'anomaly_score_report' in row.index and pd.notna(row['anomaly_score_report']):
        record['anomaly_score'] = round(float(row['anomaly_score_report']), 4)

    if 'is_anomaly' in row.index and pd.notna(row['is_anomaly']):
        record['is_anomaly'] = bool(row['is_anomaly'])

    return record


def generate_tiles(df: pd.DataFrame, output_dir: Path) -> Dict[str, int]:
    """Generate JSON tile files, one per grid cell."""
    print("\n[2/5] Generating JSON tiles...")

    tiles_dir = output_dir / 'tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Group by grid_id
    grid_groups = df.groupby('grid_id')

    tile_index = {}
    total_tiles = len(grid_groups)

    for i, (grid_id, group) in enumerate(grid_groups, 1):
        # Convert sightings to records
        sightings = [prepare_sighting_record(row) for _, row in group.iterrows()]

        # Sort by timestamp (most recent first)
        sightings.sort(key=lambda x: x['timestamp'] or '', reverse=True)

        # Create tile data
        tile_data = {
            'grid_id': grid_id,
            'count': len(sightings),
            'center': {
                'lat': float(group['grid_lat'].iloc[0]) + 0.5,  # Center of grid cell
                'lon': float(group['grid_lon'].iloc[0]) + 0.5,
            },
            'sightings': sightings
        }

        # Write tile file
        tile_filename = f"{grid_id}.json"
        tile_path = tiles_dir / tile_filename

        with open(tile_path, 'w', encoding='utf-8') as f:
            json.dump(tile_data, f, ensure_ascii=False, separators=(',', ':'))

        # Add to index
        tile_index[grid_id] = len(sightings)

        # Progress indicator
        if i % 100 == 0 or i == total_tiles:
            print(f"  ✓ Generated {i}/{total_tiles} tiles", end='\r')

    print(f"\n  ✓ Generated {total_tiles:,} tile files")
    print(f"  ✓ Total size: {sum(f.stat().st_size for f in tiles_dir.glob('*.json')) / (1024**2):.2f} MB")

    return tile_index


def export_cluster_labels(df: pd.DataFrame, output_dir: Path):
    """Export cluster ID to label mapping."""
    print("\n[3/5] Exporting cluster labels...")

    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Check if cluster data exists
    if 'cluster_id' not in df.columns or 'cluster_label' not in df.columns:
        print("  ⚠ No cluster data found, skipping cluster labels export")
        return

    # Get unique cluster mappings
    cluster_df = df[['cluster_id', 'cluster_label']].drop_duplicates()
    cluster_df = cluster_df[cluster_df['cluster_id'].notna()]

    # Create mapping
    cluster_map = {}
    for _, row in cluster_df.iterrows():
        cluster_id = int(row['cluster_id'])
        cluster_label = str(row['cluster_label'])
        cluster_map[cluster_id] = cluster_label

    # Also include count per cluster
    cluster_counts = df.groupby('cluster_id').size().to_dict()

    cluster_data = {
        'labels': cluster_map,
        'counts': {int(k): int(v) for k, v in cluster_counts.items() if pd.notna(k)}
    }

    # Write to file
    output_path = metadata_dir / 'cluster_labels.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Exported {len(cluster_map)} cluster labels")
    print(f"  ✓ Saved to: {output_path}")


def export_global_stats(df: pd.DataFrame, tile_index: Dict[str, int], output_dir: Path):
    """Export global dataset statistics."""
    print("\n[4/5] Exporting global statistics...")

    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    stats = {
        'total_sightings': int(len(df)),
        'total_tiles': len(tile_index),
        'date_range': {
            'min_year': int(df['year'].min()) if pd.notna(df['year'].min()) else None,
            'max_year': int(df['year'].max()) if pd.notna(df['year'].max()) else None,
            'min_date': df['datetime'].min().isoformat() if pd.notna(df['datetime'].min()) else None,
            'max_date': df['datetime'].max().isoformat() if pd.notna(df['datetime'].max()) else None,
        },
        'geographic': {
            'countries': int(df['country'].nunique()),
            'grid_cells': int(df['grid_id'].nunique()),
            'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())],
            'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())],
        },
        'shapes': df['shape'].value_counts().head(10).to_dict(),
        'countries': df['country'].value_counts().head(10).to_dict(),
    }

    # Add ML statistics if available
    if 'cluster_id' in df.columns:
        stats['clusters'] = {
            'total_clusters': int(df['cluster_id'].nunique()),
            'distribution': df.groupby('cluster_label').size().to_dict() if 'cluster_label' in df.columns else {}
        }

    if 'anomaly_score_report' in df.columns:
        anomaly_scores = df['anomaly_score_report'].dropna()
        stats['anomalies'] = {
            'mean_score': float(anomaly_scores.mean()),
            'median_score': float(anomaly_scores.median()),
            'high_anomaly_count': int((df['is_anomaly'] == True).sum()) if 'is_anomaly' in df.columns else None
        }

    # Export timestamp
    stats['exported_at'] = datetime.now().isoformat()

    # Write to file
    output_path = metadata_dir / 'global_stats.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Exported global statistics")
    print(f"  ✓ Saved to: {output_path}")


def export_grid_index(tile_index: Dict[str, int], output_dir: Path):
    """Export grid index file with tile counts."""
    print("\n[5/5] Exporting grid index...")

    # Sort by count (descending) for frontend optimization
    sorted_index = dict(sorted(tile_index.items(), key=lambda x: x[1], reverse=True))

    index_data = {
        'tiles': sorted_index,
        'total_tiles': len(sorted_index),
        'total_sightings': sum(sorted_index.values()),
    }

    # Write to file
    output_path = output_dir / 'index.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Exported grid index")
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ Indexed {len(sorted_index):,} tiles")


def print_export_summary(output_dir: Path):
    """Print summary of exported data."""
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    tiles_dir = output_dir / 'tiles'
    metadata_dir = output_dir / 'metadata'

    # Count files
    tile_files = list(tiles_dir.glob('*.json'))
    metadata_files = list(metadata_dir.glob('*.json'))

    # Calculate sizes
    tiles_size = sum(f.stat().st_size for f in tile_files) / (1024**2)
    metadata_size = sum(f.stat().st_size for f in metadata_files) / (1024**2)
    index_size = (output_dir / 'index.json').stat().st_size / 1024

    print(f"\n📦 Output Directory: {output_dir}")
    print(f"\n📊 Files Generated:")
    print(f"  • Tile files: {len(tile_files):,} ({tiles_size:.2f} MB)")
    print(f"  • Metadata files: {len(metadata_files)}")
    print(f"  • Index file: 1 ({index_size:.2f} KB)")

    print(f"\n📁 Directory Structure:")
    print(f"  {output_dir}/")
    print(f"  ├── tiles/")
    print(f"  │   ├── lat{37}_lon{-122}.json")
    print(f"  │   ├── ... ({len(tile_files):,} files)")
    print(f"  ├── metadata/")
    print(f"  │   ├── cluster_labels.json")
    print(f"  │   └── global_stats.json")
    print(f"  └── index.json")

    print(f"\n✅ EXPORT COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review exported data in app/public/data/")
    print("  2. Proceed to Task 4.1: Integrate map library")
    print("  3. Start building the frontend map visualization")
    print()


def main():
    """Run the complete data export pipeline."""
    try:
        # Define output directory (frontend public data)
        output_dir = Path('../app/public/data')

        # Run export pipeline
        df = load_ml_data()
        tile_index = generate_tiles(df, output_dir)
        export_cluster_labels(df, output_dir)
        export_global_stats(df, tile_index, output_dir)
        export_grid_index(tile_index, output_dir)

        # Print summary
        print_export_summary(output_dir)

    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        raise


if __name__ == "__main__":
    main()
