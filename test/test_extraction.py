import os
import sys

# Add the project root to path so we can import verus modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verus.data.extraction import DataExtractor
from verus.utils.logger import Logger

if __name__ == "__main__":
    # Create a dedicated logger for the test
    logger = Logger(name="Test_Extraction", verbose=True)

    logger.log(
        "Starting data extraction test with flexible save paths...", level="info"
    )

    # Set up proper base directory for outputs
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(base_dir, exist_ok=True)
    logger.log(f"Using base directory: {base_dir}", level="info")

    # Example 1: Extract data using a region name with timeout
    logger.log("Example 1: Testing full extraction for Porto", level="info")
    extractor = DataExtractor(
        region="Porto, Portugal",
        buffer_distance=500,
        fetch_timeout=90,  # 90 second timeout per request
        verbose=True,
    )

    # Clear cache first if you suspect stale data is causing lag
    logger.log("Clearing OSM cache...", level="info")
    extractor.clear_osm_cache()

    # Run extraction with progress indicators
    logger.log("Running extraction...", level="info")
    df = extractor.run()  # Just extract data, don't save yet

    logger.log(f"Extracted {len(df)} PoTIs", level="success")

    # Now explicitly save the data with path parameter
    logger.log("Saving extracted data...", level="info")
    saved_files = extractor.save(dataset=df, path=base_dir)  # Directory path
    logger.log(f"Saved data to {len(saved_files)} files", level="success")

    # Create and save a map if data was retrieved
    if len(df) > 0:
        logger.log("Creating map visualization...", level="info")
        map_obj = extractor.view(df)
        if map_obj:
            # Save to the maps directory
            maps_dir = os.path.join(base_dir, "maps")
            os.makedirs(maps_dir, exist_ok=True)
            map_path = os.path.join(maps_dir, "Porto_PoTI_map.html")
            map_obj.save(map_path)
            logger.log(f"Map saved to {map_path}", level="success")

    logger.log("\nExample 2: Testing extraction with custom file path...", level="info")
    # Example 2: Extract data with fewer categories and save to a custom path
    custom_amenities = {
        # Limit to fewer categories for faster testing
        "school": {"amenity": "school"},
        "hospital": {"amenity": "hospital"},
        "university": {"amenity": "university"},
    }

    # Use a smaller buffer distance
    small_extractor = DataExtractor(
        region="Aveiro, Portugal",
        buffer_distance=300,  # Smaller buffer distance
        amenity_tags=custom_amenities,  # Fewer categories
        fetch_timeout=60,
        verbose=True,
    )

    logger.log("Running extraction with smaller buffer...", level="info")
    aveiro_df = small_extractor.run()  # Extract without saving

    if len(aveiro_df) > 0:
        logger.log(
            f"Successfully extracted {len(aveiro_df)} points for Aveiro",
            level="success",
        )

        # Example 2: Save to a specific file path
        custom_file_path = os.path.join(base_dir, "custom", "aveiro_dataset.csv")
        os.makedirs(os.path.dirname(custom_file_path), exist_ok=True)
        logger.log(
            f"Saving Aveiro data to specific file: {custom_file_path}", level="info"
        )

        aveiro_files = small_extractor.save(
            dataset=aveiro_df,
            path=custom_file_path,  # Specific file path
            save_individual_categories=True,
        )

        logger.log(
            f"Saved main dataset to: {aveiro_files.get('dataset')}", level="success"
        )
        if "categories" in aveiro_files:
            logger.log(
                f"Saved {len(aveiro_files['categories'])} individual category files",
                level="success",
            )
    else:
        logger.log("Extraction for Aveiro failed", level="error")

    logger.log(
        "\nExample 3: Testing loading from file with custom save path...", level="info"
    )
    # Example 3: Load from existing file and save to a different location
    try:
        # Path to the saved Porto dataset
        file_path = os.path.join(base_dir, "poti", "Porto_dataset_buffered.csv")
        logger.log(f"Loading data from: {file_path}", level="info")

        # Load data from file
        loaded_extractor, loaded_df = DataExtractor.from_file(
            file_path=file_path, verbose=True
        )

        logger.log(
            f"Successfully loaded {len(loaded_df)} POIs from file", level="success"
        )

        # Filter the data
        schools_df = loaded_df[loaded_df["category"] == "school"].copy()
        if len(schools_df) > 0:
            logger.log(
                f"Found {len(schools_df)} schools in the dataset", level="success"
            )

            # Save to a completely custom path with descriptive name
            schools_path = os.path.join(base_dir, "processed", "porto_schools_only.csv")
            os.makedirs(os.path.dirname(schools_path), exist_ok=True)

            logger.log(f"Saving filtered schools data to: {schools_path}", level="info")
            school_files = loaded_extractor.save(
                dataset=schools_df,
                path=schools_path,
            )
            logger.log(
                f"Saved schools data to: {school_files.get('dataset')}", level="success"
            )
        else:
            logger.log("No schools found in dataset", level="warning")

    except Exception as e:
        logger.log(f"Error loading from file: {str(e)}", level="error")

    logger.log("Tests completed.", level="success")
