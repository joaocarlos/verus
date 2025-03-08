import os
import sys

from verus.clustering.kmeans import KMeansClusterer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from osmnx import geocoder

from verus.grid.hexagon import HexagonGridGenerator

if __name__ == "__main__":
    # Example 1: Using OPTICS output as input for KMeans with boundary visualization
    import os

    import pandas as pd

    from verus.clustering.optics import OpticsClusterer
    from verus.data.extraction import DataExtractor

    # Set up proper output directory using absolute path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    print(f"Using base directory: {base_dir}")

    # Ensure directories exist
    poti_dir = os.path.join(base_dir, "poti")
    time_windows_dir = os.path.join(base_dir, "time_windows")
    geojson_dir = os.path.join(base_dir, "geojson")
    os.makedirs(poti_dir, exist_ok=True)
    os.makedirs(time_windows_dir, exist_ok=True)
    os.makedirs(geojson_dir, exist_ok=True)

    # First extract data and get boundary
    extractor = DataExtractor(
        region="Porto, Portugal", buffer_distance=500, verbose=True, output_dir=base_dir
    )

    # Check if we have a dataset already, otherwise extract it
    porto_dataset_path = os.path.join(poti_dir, "Porto_dataset_buffered.csv")
    if not os.path.exists(porto_dataset_path):
        # Extract the data and save it
        poi_df = extractor.run(save_dataset=True)
    else:
        # Just get the boundaries
        extractor.get_boundaries()

    # Get boundary paths for visualization
    boundary_path = extractor.get_boundary_path()
    buffered_boundary_path = extractor.get_buffered_boundary_path()

    print(f"Boundary path: {boundary_path}")
    print(f"Buffered boundary path: {buffered_boundary_path}")

    # Run OPTICS to get initial clustering and apply time window filtering
    optics = OpticsClusterer(min_samples=5, xi=0.05, output_dir=base_dir, verbose=True)

    optics_results = optics.run(
        data_source=porto_dataset_path,
        place_name="Porto",
        time_windows_path=time_windows_dir,
        evaluation_time="ET4",
        save_output=False,
    )

    # Check if OPTICS was successful
    if optics_results["centroids"] is None:
        print("WARNING: OPTICS returned no centroids. Using a test example instead.")
        # Create a simple test example with dummy centroids
        test_data = {
            "cluster_id": [0, 1, 2, 3, 4, 5, 6, 7],
            "latitude": [41.15, 41.16, 41.14, 41.13, 41.17, 41.18, 41.19, 41.12],
            "longitude": [-8.61, -8.62, -8.63, -8.64, -8.65, -8.66, -8.67, -8.68],
        }
        optics_centroids = pd.DataFrame(test_data)
        filtered_df = pd.read_csv(porto_dataset_path)
        # Add dummy VI values if needed
        if "vi" not in filtered_df.columns:
            filtered_df["vi"] = 1.0
    else:
        filtered_df = optics_results["input_data"]
        optics_centroids = optics_results["centroids"]

    print("Using centroids:")
    print(optics_centroids)

    # Create KMeans with standard initialization (not using predefined centers)
    # Include boundary visualization
    kmeans_standard = KMeansClusterer(
        n_clusters=8, init="k-means++", output_dir=base_dir
    )

    standard_results = kmeans_standard.run(
        data_source=filtered_df,
        place_name="Porto",
        evaluation_time="ET4",
        save_output=True,
        create_map_output=True,
        area_boundary_path=boundary_path,  # Use the boundary for visualization
        algorithm_suffix="KM-Standard",
    )

    # Run with predefined centers from OPTICS
    # Include buffered boundary visualization
    kmeans = KMeansClusterer(n_clusters=len(optics_centroids), output_dir=base_dir)

    kmeans_results = kmeans.run(
        data_source=filtered_df,
        place_name="Porto",
        evaluation_time="ET4",
        centers_input=optics_centroids,
        save_output=True,
        create_map_output=True,
        area_boundary_path=buffered_boundary_path,  # Use buffered boundary
        algorithm_suffix="KM-OPTICS",
    )

    # Compare results
    if (
        kmeans_results["inertia"] is not None
        and standard_results["inertia"] is not None
    ):
        print(f"OPTICS+KMeans inertia: {kmeans_results['inertia']}")
        print(f"Standard KMeans inertia: {standard_results['inertia']}")

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
    map_obj.save("./data/maps/Porto_hex_grid_custom_map.html")
