"""
Test script for the K-means clustering module with Haversine distance.

This script demonstrates how to use the KMeansHaversine class for clustering
geographic points of interest.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from verus.clustering.kmeans import KMeansHaversine
from verus.clustering.optics import GeOPTICS
from verus.data.extraction import DataExtractor
from verus.utils.logger import Logger

if __name__ == "__main__":
    # Create a logger for the test
    logger = Logger(name="Test_KMeans", verbose=True)
    logger.log(
        "Starting K-means clustering test with DataFrame-first approach...",
        level="info",
    )

    # Set up proper base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    logger.log(f"Using base directory: {base_dir}", level="info")

    # Example 1: Basic K-means clustering
    logger.log("\nExample 1: Basic K-means clustering with existing data", level="info")

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
        saved_files = extractor.save(df, path=os.path.join(base_dir, "poti"))
        dataset_path = saved_files.get("dataset", dataset_path)
        logger.log(f"Saved dataset to: {dataset_path}", level="success")
    else:
        logger.log(f"Using existing dataset: {dataset_path}", level="info")
        df = pd.read_csv(dataset_path)

    if df is not None and not df.empty:
        # Create KMeans clusterer
        kmeans = KMeansHaversine(
            n_clusters=8,  # 8 clusters
            init="k-means++",
            max_iter=300,
            verbose=True,
        )

        # Get boundary path if available
        boundary_path = os.path.join(base_dir, "geojson", "Porto_boundary.geojson")
        if not os.path.exists(boundary_path):
            boundary_path = None

        # Run clustering
        results = kmeans.run(
            data_source=df,
            area_boundary_path=boundary_path,
        )

        # Check results
        if results["clusters"] is not None:
            logger.log(
                f"Created {kmeans.n_clusters} clusters with {len(results['clusters'])} points",
                level="success",
            )
            logger.log(
                f"K-means converged in {results['n_iter']} iterations with inertia {results['inertia']:.2f}",
                level="info",
            )

            # Save results
            saved_files = kmeans.save(
                results["clusters"],
                results["centroids"],
                path=os.path.join(base_dir, "clusters", "Porto_kmeans"),
            )

            # Save map if created
            if results["map"]:
                maps_dir = os.path.join(base_dir, "maps")
                os.makedirs(maps_dir, exist_ok=True)
                map_path = os.path.join(maps_dir, "Porto_kmeans_map.html")
                results["map"].save(map_path)
                logger.log(f"Saved map to: {map_path}", level="success")
        else:
            logger.log("Clustering did not produce valid results", level="error")
    else:
        logger.log("No data available for clustering", level="error")

    # Example 2: K-means clustering using OPTICS centroids as a starting point
    logger.log(
        "\nExample 2: K-means with OPTICS centroids initialization", level="info"
    )

    # First run OPTICS to get initial centroids
    optics = GeOPTICS(
        min_samples=5,
        xi=0.05,
        min_cluster_size=5,
        verbose=True,
    )

    optics_results = optics.run(
        data_source=df,
        area_boundary_path=boundary_path,
    )

    if optics_results["centroids"] is not None:
        n_optics_clusters = len(optics_results["centroids"])
        logger.log(
            f"OPTICS found {n_optics_clusters} clusters to use as seeds", level="info"
        )

        # Create KMeans with predefined centers from OPTICS
        kmeans_from_optics = KMeansHaversine(
            n_clusters=n_optics_clusters,
            init="predefined",
            predefined_centers=optics_results["centroids"][
                ["latitude", "longitude"]
            ].values,
            verbose=True,
        )

        # Run K-means using OPTICS centroids
        logger.log("Running K-means with OPTICS centroids", level="info")
        kmeans_results = kmeans_from_optics.run(
            data_source=df,
            area_boundary_path=boundary_path,
        )

        if kmeans_results["clusters"] is not None:
            logger.log(
                f"Created {n_optics_clusters} clusters with {len(kmeans_results['clusters'])} points",
                level="success",
            )

            # Save results
            saved_files = kmeans_from_optics.save(
                kmeans_results["clusters"],
                kmeans_results["centroids"],
                path=os.path.join(base_dir, "clusters", "Porto_kmeans_from_optics"),
            )

            # Save map
            if kmeans_results["map"]:
                map_path = os.path.join(
                    base_dir, "maps", "Porto_kmeans_from_optics.html"
                )
                kmeans_results["map"].save(map_path)
                logger.log(f"Saved map to: {map_path}", level="success")
    else:
        logger.log("OPTICS did not produce valid centroids", level="warning")

    # Example 3: K-means clustering with vulnerability weights from OPTICS time filtering
    logger.log(
        "\nExample 3: K-means with time-based vulnerability weights", level="info"
    )

    # Ensure time_windows_dir exists
    time_windows_dir = os.path.join(base_dir, "time_windows")
    if not os.path.exists(time_windows_dir) or not os.listdir(time_windows_dir):
        logger.log("Time windows not found. Skipping Example 3.", level="warning")
    else:
        # Run OPTICS with time window for ET4 (evening peak)
        logger.log(
            "Running OPTICS with time window filter for Evening Peak (ET4)",
            level="info",
        )

        # Use the correct parameter name 'time_windows' instead of 'time_windows_path'
        time_optics_results = optics.run(
            data_source=df,
            time_windows=time_windows_dir,  # Changed from time_windows_path
            evaluation_time="ET4",
        )

        if (
            time_optics_results["input_data"] is not None
            and not time_optics_results["input_data"].empty
        ):
            filtered_data = time_optics_results["input_data"]
            logger.log(
                f"Got {len(filtered_data)} time-filtered points with vulnerability indices",
                level="info",
            )

            # Run K-means on filtered data
            weighted_kmeans = KMeansHaversine(
                n_clusters=5,
                init="k-means++",
                verbose=True,
            )

            weighted_results = weighted_kmeans.run(
                data_source=filtered_data,  # This has VI values that will be used as weights
                area_boundary_path=boundary_path,
            )

            if weighted_results["clusters"] is not None:
                logger.log(
                    f"Created {weighted_kmeans.n_clusters} weighted clusters with {len(weighted_results['clusters'])} points",
                    level="success",
                )

                # Save results
                saved_files = weighted_kmeans.save(
                    weighted_results["clusters"],
                    weighted_results["centroids"],
                    path=os.path.join(base_dir, "clusters", "Porto_weighted_kmeans"),
                    evaluation_time="ET4",
                )

                # Save map
                if weighted_results["map"]:
                    map_path = os.path.join(
                        base_dir, "maps", "Porto_weighted_kmeans_ET4.html"
                    )
                    weighted_results["map"].save(map_path)
                    logger.log(f"Saved map to: {map_path}", level="success")
        else:
            logger.log("Time-filtered data is not available", level="warning")
