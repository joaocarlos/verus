from verus.clustering.optics import GeOPTICS
from verus.data.extraction import DataExtractor

if __name__ == "__main__":
    import os

    import pandas as pd

    from verus.data.extraction import DataExtractor

    # Set up proper output directory using absolute path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    print(f"Using base directory: {base_dir}")

    # First extract POI data
    extractor = DataExtractor(
        region="Porto, Portugal",
        buffer_distance=500,
        output_dir=base_dir,  # Use the same base_dir
        verbose=True,
    )

    # Define path to dataset
    dataset_path = os.path.join(base_dir, "poti", "Porto_dataset_buffered.csv")

    # Check if we need to extract data or if it already exists
    if not os.path.exists(dataset_path):
        print("Extracting POI data...")
        df = extractor.run(save_dataset=True)
    else:
        print(f"Using existing dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)

    if df is not None and not df.empty:
        # Define the time windows path
        time_windows_dir = os.path.join(base_dir, "time_windows")
        os.makedirs(time_windows_dir, exist_ok=True)

        # Create OPTICS clusterer with same base_dir
        optics = GeOPTICS(
            min_samples=5,
            xi=0.05,
            min_cluster_size=5,
            output_dir=base_dir,  # Use the same base_dir
            verbose=True,
        )

        # Run clustering with time window filtering
        results = optics.run(
            data_source=df,
            place_name="Porto",
            time_windows_path=time_windows_dir,
            evaluation_time="ET4",  # Evening peak
            save_output=True,
            create_map=True,
        )

        # Check results
        if results["clusters"] is not None:
            # Get unique cluster count
            num_unique_clusters = len(results["clusters"]["cluster_id"].unique())
            print(
                f"Created {num_unique_clusters} clusters with {len(results['clusters'])} total points"
            )
            print(f"Found {len(results['centroids'])} centroids")
            if results["map"]:
                print("Created interactive map")
        else:
            print("Clustering did not produce valid results")
    else:
        print("No data available for clustering")
