import os
import sys

# Add the project root to path so we can import verus modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verus.data.extraction import DataExtractor

if __name__ == "__main__":
    print("Starting data extraction test...")

    # Set up proper base directory for outputs
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(base_dir, exist_ok=True)

    # Example 1: Extract data using a region name with timeout and proper directory structure
    extractor = DataExtractor(
        region="Porto, Portugal",
        buffer_distance=500,
        fetch_timeout=90,  # 90 second timeout per request
        output_dir=base_dir,  # Use the proper output directory
        verbose=True,
    )

    # Clear cache first if you suspect stale data is causing lag
    print("Clearing OSM cache...")
    extractor.clear_osm_cache()

    # Run extraction with progress indicators
    print("Running extraction...")
    df = extractor.run(save_dataset=True)

    # Create and save a map if data was retrieved
    if df is not None:
        print("Creating map...")
        map_obj = extractor.create_map(df)
        if map_obj:
            # Save to the proper maps directory
            maps_dir = os.path.join(base_dir, "maps")
            os.makedirs(maps_dir, exist_ok=True)
            map_path = os.path.join(maps_dir, "Porto_map.html")
            map_obj.save(map_path)
            print(f"Map saved to {map_path}")

    print("\nTesting with smaller batch sizes...")
    # Example 2: Extract data with smaller batch sizes to avoid lag
    custom_amenities = {
        # Limit to fewer categories for faster testing
        "school": {"amenity": "school"},
        "hospital": {"amenity": "hospital"},
        "university": {"amenity": "university"},
    }

    # Use a smaller buffer distance to reduce query size
    small_extractor = DataExtractor(
        region="Aveiro, Portugal",
        buffer_distance=300,  # Smaller buffer distance
        amenity_tags=custom_amenities,  # Fewer categories
        output_dir=base_dir,
        fetch_timeout=60,
        verbose=True,
    )

    print("Running extraction with smaller buffer...")
    aveiro_df = small_extractor.run(save_dataset=True)

    if aveiro_df is not None:
        print(f"Successfully extracted {len(aveiro_df)} points for Aveiro")
    else:
        print("Extraction for Aveiro failed")

    print("Tests completed.")
