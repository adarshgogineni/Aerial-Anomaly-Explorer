"""
UAP Explorer - Data Cleaning Script

This script cleans and normalizes the raw UAP sighting dataset.

Steps:
1. Load raw data from data/raw/complete.csv
2. Parse and normalize dates to UTC
3. Validate and clean geographic coordinates
4. Clean and normalize text fields
5. Handle missing values
6. Export cleaned data to data/processed/cleaned_sightings.parquet

Usage:
    python scripts/clean_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dateutil import parser
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_raw_data():
    """Load the raw dataset with error handling."""
    print("=" * 60)
    print("UAP EXPLORER - DATA CLEANING PIPELINE")
    print("=" * 60)
    print("\n[1/6] Loading raw data...")

    data_path = Path('data/raw/complete.csv')

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load with error handling for malformed lines
    df = pd.read_csv(data_path, on_bad_lines='skip', engine='python')

    print(f"  ✓ Loaded {len(df):,} raw records")
    print(f"  ✓ Found {len(df.columns)} columns")

    return df


def clean_dates(df):
    """Parse and normalize dates to standard format."""
    print("\n[2/6] Cleaning and normalizing dates...")

    initial_count = len(df)

    # Parse datetime column
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Drop records with invalid dates
    df = df[df['datetime'].notna()].copy()

    # Extract useful date components
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour

    # Create year_month for aggregation
    df['year_month'] = df['datetime'].dt.to_period('M')

    dropped = initial_count - len(df)
    print(f"  ✓ Parsed dates successfully")
    print(f"  ✓ Dropped {dropped:,} records with invalid dates")
    print(f"  ✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def clean_locations(df):
    """Validate and clean geographic coordinates."""
    print("\n[3/6] Cleaning and validating coordinates...")

    initial_count = len(df)

    # Convert to numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Validate coordinate ranges
    # Latitude: -90 to 90
    # Longitude: -180 to 180
    df = df[
        (df['latitude'].notna()) &
        (df['longitude'].notna()) &
        (df['latitude'].between(-90, 90)) &
        (df['longitude'].between(-180, 180))
    ].copy()

    # Clean location text fields
    df['city'] = df['city'].fillna('Unknown').str.strip().str.title()
    df['state'] = df['state'].fillna('').str.strip().str.upper()
    df['country'] = df['country'].fillna('').str.strip().str.upper()

    # Create combined location string
    df['location'] = df.apply(
        lambda row: f"{row['city']}, {row['state']}, {row['country']}"
        if row['state'] else f"{row['city']}, {row['country']}",
        axis=1
    )

    # Create spatial grid for aggregation (1 degree grid)
    df['grid_lat'] = (df['latitude'] // 1.0) * 1.0
    df['grid_lon'] = (df['longitude'] // 1.0) * 1.0
    df['grid_id'] = df.apply(
        lambda row: f"lat{int(row['grid_lat'])}_lon{int(row['grid_lon'])}",
        axis=1
    )

    dropped = initial_count - len(df)
    print(f"  ✓ Validated coordinates")
    print(f"  ✓ Dropped {dropped:,} records with invalid coordinates")
    print(f"  ✓ Remaining records: {len(df):,}")

    return df


def clean_text(df):
    """Clean and normalize text fields."""
    print("\n[4/6] Cleaning text fields...")

    # Clean comments/description field
    df['description'] = df['comments'].fillna('')

    # Decode HTML entities and clean text
    df['description'] = (
        df['description']
        .str.replace(r'&#44', ',', regex=True)
        .str.replace(r'&amp;', '&', regex=True)
        .str.replace(r'&#\d+', ' ', regex=True)  # Remove other HTML entities
        .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
        .str.strip()
    )

    # Calculate text length for analysis
    df['description_length'] = df['description'].str.len()

    # Clean shape field
    df['shape'] = (
        df['shape']
        .fillna('unknown')
        .str.strip()
        .str.lower()
        .replace('', 'unknown')
    )

    # Normalize duration
    df['duration_seconds'] = pd.to_numeric(
        df['duration (seconds)'],
        errors='coerce'
    ).fillna(0)

    # Cap extreme durations (more than 24 hours is likely data error)
    max_duration = 24 * 3600  # 24 hours in seconds
    df.loc[df['duration_seconds'] > max_duration, 'duration_seconds'] = max_duration

    print(f"  ✓ Cleaned description field")
    print(f"  ✓ Normalized shape categories: {df['shape'].nunique()} unique shapes")
    print(f"  ✓ Processed duration data")

    return df


def handle_missing_values(df):
    """Handle remaining missing values with appropriate strategy."""
    print("\n[5/6] Handling missing values...")

    # Check missing values before
    missing_before = df.isnull().sum().sum()

    # Strategy:
    # - Critical fields already validated (datetime, lat, lon)
    # - Fill description with empty string (already done)
    # - Fill shape with 'unknown' (already done)
    # - Fill duration with 0 (already done)

    # Drop any rows with missing critical fields
    # (should be none at this point, but as a safety check)
    critical_fields = ['datetime', 'latitude', 'longitude', 'description']
    df = df.dropna(subset=critical_fields)

    missing_after = df.isnull().sum().sum()

    print(f"  ✓ Missing values before: {missing_before:,}")
    print(f"  ✓ Missing values after: {missing_after:,}")
    print(f"  ✓ Final dataset size: {len(df):,} records")

    return df


def export_cleaned_data(df):
    """Export cleaned data to processed directory."""
    print("\n[6/6] Exporting cleaned data...")

    # Create processed directory if it doesn't exist
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select final columns for export
    columns_to_export = [
        'datetime', 'year', 'month', 'day_of_week', 'hour', 'year_month',
        'city', 'state', 'country', 'location',
        'latitude', 'longitude', 'grid_lat', 'grid_lon', 'grid_id',
        'shape', 'duration_seconds',
        'description', 'description_length',
        'date posted'
    ]

    # Create unique ID for each sighting
    df['id'] = df.reset_index().index.astype(str)
    columns_to_export.insert(0, 'id')

    # Export to parquet (more efficient than CSV)
    output_path = output_dir / 'cleaned_sightings.parquet'
    df[columns_to_export].to_parquet(output_path, index=False)

    print(f"  ✓ Exported to: {output_path}")
    print(f"  ✓ File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"  ✓ Records exported: {len(df):,}")
    print(f"  ✓ Columns exported: {len(columns_to_export)}")

    return output_path


def print_summary_stats(df):
    """Print summary statistics of cleaned data."""
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(f"\n📊 Dataset Statistics:")
    print(f"  • Total sightings: {len(df):,}")
    print(f"  • Date range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"  • Countries: {df['country'].nunique()}")
    print(f"  • Unique shapes: {df['shape'].nunique()}")
    print(f"  • Grid cells covered: {df['grid_id'].nunique()}")

    print(f"\n📍 Geographic Coverage:")
    top_countries = df['country'].value_counts().head(5)
    for country, count in top_countries.items():
        print(f"  • {country}: {count:,} sightings")

    print(f"\n🔷 Top Shapes:")
    top_shapes = df['shape'].value_counts().head(5)
    for shape, count in top_shapes.items():
        print(f"  • {shape}: {count:,} sightings")

    print(f"\n📅 Temporal Distribution:")
    print(f"  • Peak year: {df.groupby('year').size().idxmax():.0f} ({df.groupby('year').size().max():,} sightings)")
    print(f"  • Average per year: {len(df) / df['year'].nunique():.0f}")

    print("\n" + "=" * 60)
    print("✅ DATA CLEANING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review cleaned data: data/processed/cleaned_sightings.parquet")
    print("  2. Proceed to Task 2.3: Spatiotemporal aggregation")
    print("  3. Or start exploring in a notebook")
    print()


def main():
    """Run the complete data cleaning pipeline."""
    try:
        # Run cleaning pipeline
        df = load_raw_data()
        df = clean_dates(df)
        df = clean_locations(df)
        df = clean_text(df)
        df = handle_missing_values(df)
        output_path = export_cleaned_data(df)

        # Print summary
        print_summary_stats(df)

        return df

    except Exception as e:
        print(f"\n❌ Error during cleaning: {e}")
        raise


if __name__ == "__main__":
    cleaned_df = main()
