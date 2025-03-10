import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from osmnx import geocoder

from verus.grid.hexagon import HexagonGridGenerator

if __name__ == "__main__":
    # Set up proper output directory using absolute path
    # This should be consistent across all components
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    print(f"Using base directory: {base_dir}")

    # Create generator with edge length of 250 meters
    gen = HexagonGridGenerator(
        region="Paris, France",
        edge_length=250,
        output_dir=base_dir,  # Use consistent base directory
        verbose=True,
    )

    # Generate hexagonal grid
    grid = gen.run(save_output=True)

    if grid is not None:
        print(f"Generated {len(grid)} hexagons")

    # Example 2: Custom usage with specific steps
    custom_generator = HexagonGridGenerator(
        region="Porto, Portugal", edge_length=300, verbose=True, output_dir=base_dir
    )

    # Get the area
    area_gdf = geocoder.geocode_to_gdf("Porto, Portugal")
    bounding_box = area_gdf.bounds.iloc[0]

    # Generate the grid
    porto_grid = custom_generator.generate_hex_grid(bounding_box)

    # Add property values and colors
    porto_grid = custom_generator.assign_random_values(
        porto_grid, seed=123, min_val=0, max_val=10
    )
    porto_grid = custom_generator.assign_colors(porto_grid)

    # Clip to region and save
    porto_grid_clipped = custom_generator.clip_to_region(porto_grid, area_gdf)
    custom_generator.save_to_geojson(
        porto_grid_clipped, "Porto_hex_grid_custom.geojson"
    )

    # Create map with the clipped grid
    map_obj = custom_generator.create_map(porto_grid_clipped, area_gdf)
    # Check if folder exists
    maps_dir = os.path.join(base_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    # Save to the proper maps directory
    map_obj.save(os.path.join(maps_dir, "Porto_hex_grid_custom_map.html"))
