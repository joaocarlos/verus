"""
Test script for the OPTICS clustering module.

This script demonstrates how to use the GeOPTICS class for clustering
points of interest with optional time-based vulnerability indexing.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from verus.clustering.optics import GeOPTICS
from verus.data.extraction import DataExtractor
from verus.data.timewindow import TimeWindowGenerator
from verus.utils.logger import Logger

if __name__ == "__main__":
    # Create a logger for the test
    logger = Logger(name="Test_OPTICS", verbose=True)
    logger.log(
        "Starting OPTICS clustering test with DataFrame-first approach...", level="info"
    )

    # Set up base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    logger.log(f"Using base directory: {base_dir}", level="info")

    # Example 1: Simple clustering test on pre-extracted data
    logger.log("\nExample 1: Basic clustering with existing data", level="info")

    # Define path to dataset
    dataset_path = os.path.join(base_dir, "poti", "Porto_dataset_buffered.csv")

    # Check if we need to extract data or if it already exists
    if not os.path.exists(dataset_path):
        logger.log("Dataset not found. Extracting POI data...", level="info")

        # Extract data using DataExtractor
        extractor = DataExtractor(
            region="Porto, Portugal",
            buffer_distance=500,
            verbose=True,
        )

        # Run extraction
        df = extractor.run()

        # Save results
        saved_files = extractor.save(df, path=os.path.join(base_dir, "poti"))
        dataset_path = saved_files.get("dataset", dataset_path)
        logger.log(f"Saved dataset to: {dataset_path}", level="success")
    else:
        logger.log(f"Using existing dataset: {dataset_path}", level="info")
        df = pd.read_csv(dataset_path)

    if df is not None and not df.empty:
        # Create OPTICS clusterer
        optics = GeOPTICS(
            min_samples=5,
            xi=0.05,
            min_cluster_size=5,
            verbose=True,
        )

        # Get boundary path if available
        boundary_path = os.path.join(base_dir, "geojson", "Porto_boundary.geojson")
        if not os.path.exists(boundary_path):
            boundary_path = None

        # Run clustering without time windows
        logger.log("Running OPTICS clustering without time windows", level="info")
        results = optics.run(
            data_source=df,
            area_boundary_path=boundary_path,
        )

        # Check results
        if results["clusters"] is not None:
            num_unique_clusters = len(results["clusters"]["cluster"].unique())
            logger.log(
                f"Created {num_unique_clusters} clusters with {len(results['clusters'])} total points",
                level="success",
            )

            # Save results
            logger.log("Saving clustering results", level="info")
            saved_files = optics.save(
                results["clusters"],
                results["centroids"],
                path=os.path.join(base_dir, "clusters", "Porto_optics"),
            )

            # Save map if created
            if results["map"]:
                maps_dir = os.path.join(base_dir, "maps")
                os.makedirs(maps_dir, exist_ok=True)
                map_path = os.path.join(maps_dir, "Porto_optics_map.html")
                results["map"].save(map_path)
                logger.log(f"Saved map to: {map_path}", level="success")
        else:
            logger.log("Clustering did not produce valid results", level="error")
    else:
        logger.log("No data available for clustering", level="error")

    # Example 2: Advanced clustering with time window filtering
    logger.log("\nExample 2: Clustering with time window filtering", level="info")

    # First ensure we have time window data
    time_windows_dir = os.path.join(base_dir, "time_windows")

    if not os.path.exists(time_windows_dir) or not os.listdir(time_windows_dir):
        logger.log("Time windows not found. Generating...", level="info")

        # Generate time windows
        tw_generator = TimeWindowGenerator(
            reference_date="2023-11-06",  # Monday
            verbose=True,
        )

        # Generate windows
        time_windows = tw_generator.generate_from_schedule()

        # Save time windows
        tw_generator.save(time_windows, path=time_windows_dir)
        logger.log(
            f"Generated and saved time windows to: {time_windows_dir}", level="success"
        )
    else:
        logger.log(f"Using existing time windows in: {time_windows_dir}", level="info")

    # Example 2.1: Using time windows directory path
    logger.log("\nExample 2.1: Using time windows directory path", level="info")
    evaluation_times = ["ET2", "ET4"]  # Morning peak, Evening peak

    for et in evaluation_times:
        logger.log(f"\nTesting time scenario: {et} with directory path", level="info")

        # Run clustering with time window filtering using directory path
        time_results = optics.run(
            data_source=df,
            time_windows=time_windows_dir,  # Using directory path
            evaluation_time=et,
            area_boundary_path=boundary_path,
        )

        # Check results
        if time_results["clusters"] is not None:
            num_unique_clusters = len(time_results["clusters"]["cluster"].unique())
            logger.log(
                f"Time scenario {et}: Created {num_unique_clusters} clusters with "
                f"{len(time_results['clusters'])} points (from {len(time_results['input_data'])} filtered POIs)",
                level="success",
            )

            # Save results
            saved_files = optics.save(
                time_results["clusters"],
                time_results["centroids"],
                path=os.path.join(base_dir, "clusters"),
                evaluation_time=et,
            )

    # Example 2.2: Using DataFrames for time windows
    logger.log("\nExample 2.2: Using time windows DataFrames", level="info")

    # Load time windows into DataFrames
    time_windows_dfs = {}
    for file in os.listdir(time_windows_dir):
        if file.endswith(".csv"):
            try:
                category = os.path.splitext(file)[0]
                file_path = os.path.join(time_windows_dir, file)
                time_windows_dfs[category] = pd.read_csv(file_path)
            except Exception as e:
                logger.log(f"Error loading {file}: {str(e)}", level="warning")

    if time_windows_dfs:
        # Run clustering with time window filtering using DataFrame
        et = "ET5"  # Weekend scenario
        logger.log(f"\nTesting time scenario: {et} with DataFrames", level="info")

        # Run clustering with time window filtering using DataFrame dictionary
        time_results = optics.run(
            data_source=df,
            time_windows=time_windows_dfs,  # Using DataFrame dictionary
            evaluation_time=et,
            area_boundary_path=boundary_path,
        )

        # Check results
        if time_results["clusters"] is not None:
            num_unique_clusters = len(time_results["clusters"]["cluster"].unique())
            logger.log(
                f"Time scenario {et} with DataFrame: Created {num_unique_clusters} clusters with "
                f"{len(time_results['clusters'])} points",
                level="success",
            )

            # Save results
            saved_files = optics.save(
                time_results["clusters"],
                time_results["centroids"],
                path=os.path.join(base_dir, "clusters", "Porto_df_method"),
                evaluation_time=et,
            )

    # Example 3: Customized clustering process with explicit steps
    logger.log("\nExample 3: Custom step-by-step clustering workflow", level="info")

    # Define custom parameters
    custom_optics = GeOPTICS(
        min_samples=3,
        xi=0.1,
        min_cluster_size=3,
        verbose=True,
    )

    # Combine all time windows into a single DataFrame
    if time_windows_dfs:
        all_time_windows = []
        for category, df_tw in time_windows_dfs.items():
            df_copy = df_tw.copy()
            df_copy["category"] = category
            all_time_windows.append(df_copy)

        combined_time_windows = pd.concat(all_time_windows, ignore_index=True)

        # Step 1: Load data with lunchtime filter using DataFrame
        logger.log(
            "Step 1: Loading data with lunchtime filter using DataFrame", level="info"
        )
        lunch_data = custom_optics.load(
            data_source=df,
            time_windows=combined_time_windows,  # Using combined DataFrame
            evaluation_time="ET3",  # Lunch time
        )

        if lunch_data is not None and not lunch_data.empty:
            logger.log(
                f"Loaded {len(lunch_data)} POIs for lunchtime scenario", level="info"
            )

            # Step 2: Run clustering on filtered data
            logger.log("Step 2: Running OPTICS clustering", level="info")
            lunch_clusters, lunch_centroids = custom_optics.fit(lunch_data)

            if lunch_clusters is not None:
                num_lunch_clusters = len(lunch_clusters["cluster"].unique())
                logger.log(
                    f"Found {num_lunch_clusters} lunchtime clusters", level="info"
                )

                # Step 3: Create map visualization and save it directly
                logger.log("Step 3: Creating map visualization", level="info")
                map_path = os.path.join(base_dir, "maps", "Porto_lunch_clusters.html")
                lunch_map = custom_optics.view(
                    lunch_clusters,
                    lunch_centroids,
                    boundary_path,
                    save_path=map_path,  # Save map directly
                )

                # Step 4: Save clustering results
                logger.log("Step 4: Saving results", level="info")
                saved_files = custom_optics.save(
                    lunch_clusters,
                    lunch_centroids,
                    path=os.path.join(base_dir, "clusters", "custom_lunch"),
                    evaluation_time="ET3",
                )

                logger.log(
                    "Custom clustering workflow completed successfully", level="success"
                )

    logger.log("OPTICS clustering tests completed successfully!", level="success")
