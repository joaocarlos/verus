import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from osmnx import geocoder

from verus.grid.hexagon import HexagonGridGenerator
from verus.utils.logger import Logger

if __name__ == "__main__":
    # Create a logger for the test
    logger = Logger(name="Test_HexagonGrid", verbose=True)
    logger.log(
        "Starting hexagon grid test with DataFrame-first approach...", level="info"
    )

    # Set up proper base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    logger.log(f"Using base directory: {base_dir}", level="info")

    # Example 1: Basic grid generation with run() method
    logger.log("Example 1: Basic grid generation for Paris", level="info")

    # Create generator with edge length of 250 meters
    gen = HexagonGridGenerator(region="Paris, France", edge_length=250, verbose=True)

    # Generate the grid but don't save yet
    grid = gen.run(save_output=False)

    if grid is not None:
        logger.log(f"Generated {len(grid)} hexagons", level="success")

        # Now explicitly save the grid
        output_path = os.path.join(base_dir, "geojson", "Paris_hex_grid.geojson")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        saved_path = gen.save_to_geojson(grid, path=output_path)
        logger.log(f"Saved grid to: {saved_path}", level="success")

        # Create and save map
        logger.log("Creating map visualization...", level="info")
        map_obj = gen.create_map(grid)
        if map_obj:
            maps_dir = os.path.join(base_dir, "maps")
            os.makedirs(maps_dir, exist_ok=True)
            map_obj.save(os.path.join(maps_dir, "Paris_hex_grid_map.html"))
            logger.log("Saved map to maps directory", level="success")

    # Example 2: Step-by-step custom workflow
    logger.log("\nExample 2: Custom step-by-step workflow for Porto", level="info")
    custom_generator = HexagonGridGenerator(
        region="Porto, Portugal", edge_length=300, verbose=True
    )

    # Step 1: Get the area
    logger.log("Step 1: Getting region boundary", level="info")
    area_gdf = geocoder.geocode_to_gdf("Porto, Portugal")
    bounding_box = area_gdf.bounds.iloc[0]

    # Step 2: Generate the grid
    logger.log("Step 2: Generating grid", level="info")
    porto_grid = custom_generator.generate_hex_grid(bounding_box)
    logger.log(f"Generated {len(porto_grid)} hexagons", level="info")

    # Step 3: Add property values and colors
    logger.log("Step 3: Adding values and colors", level="info")
    porto_grid = custom_generator.assign_random_values(
        porto_grid, seed=123, min_val=0, max_val=10
    )
    porto_grid = custom_generator.assign_colors(porto_grid)

    # Step 4: Clip to region and save
    logger.log("Step 4: Clipping to region boundary", level="info")
    porto_grid_clipped = custom_generator.clip_to_region(porto_grid, area_gdf)
    logger.log(f"Clipped to {len(porto_grid_clipped)} hexagons", level="info")

    # Step 5: Save to a custom path
    custom_path = os.path.join(base_dir, "geojson", "Porto_hex_grid_custom.geojson")
    custom_generator.save_to_geojson(porto_grid_clipped, path=custom_path)

    # Step 6: Create and save visualization
    logger.log("Step 6: Creating visualization", level="info")
    map_obj = custom_generator.create_map(porto_grid_clipped, area_gdf)
    map_path = os.path.join(base_dir, "maps", "Porto_hex_grid_custom_map.html")
    map_obj.save(map_path)
    logger.log(f"Saved map to: {map_path}", level="success")

    # Example 3: Loading from an existing file
    logger.log("\nExample 3: Loading grid from file", level="info")
    try:
        # Use the file we just saved
        loaded_generator, loaded_grid = HexagonGridGenerator.from_file(
            file_path=custom_path, verbose=True
        )

        logger.log(
            f"Successfully loaded {len(loaded_grid)} hexagons from file",
            level="success",
        )

        # Demonstrate manipulating the loaded data
        logger.log("Filtering hexagons by value", level="info")
        if "value" in loaded_grid.columns:
            high_value = loaded_grid[loaded_grid["value"] > 7]
            logger.log(f"Found {len(high_value)} high-value hexagons", level="info")

            # Create visualization of high-value areas only
            high_value_map = loaded_generator.create_map(high_value, area_gdf)
            high_value_map.save(
                os.path.join(base_dir, "maps", "Porto_high_value_hexagons.html")
            )
            logger.log("Saved filtered map visualization", level="success")

    except Exception as e:
        logger.log(f"Error loading from file: {str(e)}", level="error")

    logger.log("Hexagon grid tests completed successfully!", level="success")
