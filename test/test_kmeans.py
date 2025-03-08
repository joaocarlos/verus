from verus.clustering import KMeansHaversine

if __name__ == "__main__":
    # Example 1: Using OPTICS output as input for KMeans with boundary visualization
    import os

    import pandas as pd

    from verus.clustering import GeOPTICS
    from verus.data.extraction import DataExtractor

    # Set paths for testing
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    print("Base directory:", base_dir)
    poti_dir = os.path.join(base_dir, "poti")
    time_windows_dir = os.path.join(base_dir, "time_windows")
    geojson_dir = os.path.join(base_dir, "geojson")

    # Create directories if they don't exist
    for directory in [poti_dir, time_windows_dir, geojson_dir]:
        os.makedirs(directory, exist_ok=True)

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
    optics = GeOPTICS(min_samples=5, xi=0.05, output_dir=base_dir, verbose=True)
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
    kmeans_standard = KMeansHaversine(
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
    kmeans = KMeansHaversine(n_clusters=len(optics_centroids), output_dir=base_dir)
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

    # Example 2: Alternative approach using a custom GeoJSON boundary file
    # This example shows how to use a custom GeoJSON file if you have one
    custom_geojson = os.path.join(geojson_dir, "custom_boundary.geojson")
    if os.path.exists(custom_geojson):
        print(f"Using custom boundary file: {custom_geojson}")

        # Create custom KMeans with the custom boundary
        kmeans_custom = KMeansHaversine(
            n_clusters=len(optics_centroids), output_dir=base_dir
        )
        custom_results = kmeans_custom.run(
            data_source=filtered_df,
            place_name="Porto",
            evaluation_time="ET4-Custom",
            centers_input=optics_centroids,
            save_output=True,
            create_map_output=True,
            area_boundary_path=custom_geojson,  # Use custom boundary
            algorithm_suffix="KM-Custom",
        )
    else:
        print(f"No custom boundary file found at {custom_geojson}")
