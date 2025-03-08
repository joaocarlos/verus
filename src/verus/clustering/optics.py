import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.neighbors import KNeighborsClassifier

from verus.clustering import Cluster


class GeOPTICS(Cluster):
    """
    Class for performing OPTICS clustering on geospatial data with time-based vulnerability indexing.

    This class implements OPTICS (Ordering Points To Identify the Clustering Structure) algorithm
    to cluster points of interest based on their spatial proximity, with the option to filter
    points based on time-specific vulnerability indices.
    """

    def __init__(
        self,
        min_samples=5,
        xi=0.05,
        min_cluster_size=5,
        output_dir=None,
        verbose=True,
        et_scenarios=None,
    ):
        """
        Initialize the OpticsClusterer with clustering parameters.

        Args:
            min_samples (int): The number of samples in a neighborhood for a point
                              to be considered a core point.
            xi (float): Determines the minimum steepness on the reachability plot
                       that constitutes a cluster boundary.
            min_cluster_size (int): Minimum number of points to form a cluster.
            output_dir (str, optional): Base directory for output files
            verbose (bool): Whether to print informational messages.
            et_scenarios (dict, optional): Custom evaluation time scenarios dictionary.
                                          Used primarily for debugging.
        """
        # Initialize the base clusterer
        super().__init__(output_dir=output_dir, verbose=verbose)

        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")

        if not isinstance(xi, float) or xi <= 0 or xi >= 1:
            raise ValueError("xi must be a float between 0 and 1")

        if not isinstance(min_cluster_size, int) or min_cluster_size < 1:
            raise ValueError("min_cluster_size must be a positive integer")

        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size

        # Evaluation time scenarios (optional, primarily for debugging)
        self.et_scenarios = et_scenarios or {
            "ET1": {
                "name": "Evaluation Time Scenario 1",
                "description": "Weekend to demonstrate low activity",
                "datetime": int(datetime(2023, 11, 11, 10, 20, 0).timestamp()),
            },
            "ET2": {
                "name": "Evaluation Time Scenario 2",
                "description": "Weekday morning peak - Schools, universities, and transportation hubs at high activity",
                "datetime": int(datetime(2023, 11, 6, 8, 40, 0).timestamp()),
            },
            "ET3": {
                "name": "Evaluation Time Scenario 3",
                "description": "Weekday midday - Lunchtime rush in shopping centres and transport hubs",
                "datetime": int(datetime(2023, 11, 6, 12, 30, 0).timestamp()),
            },
            "ET4": {
                "name": "Evaluation Time Scenario 4",
                "description": "Weekday evening peak - High activity in transportation hubs, shopping centres, and educational institutions",
                "datetime": int(datetime(2023, 11, 6, 17, 30, 0).timestamp()),
            },
            "ET5": {
                "name": "Evaluation Time Scenario 5",
                "description": "Weekend midday - High activity in tourist attractions and shopping centres",
                "datetime": int(datetime(2023, 11, 12, 13, 0, 0).timestamp()),
            },
        }

    def load_data(self, data_source, time_windows_path=None, evaluation_time=None):
        """
        Load POI data and optionally time window data.

        Args:
            data_source (str or pd.DataFrame): Path to POI CSV file or DataFrame from extraction.py
            time_windows_path (str, optional): Path to the time windows directory.
            evaluation_time (str or int, optional): Time scenario key (e.g., "ET4") or
                                                  epoch timestamp directly.

        Returns:
            pd.DataFrame: DataFrame containing points of interest, filtered by time if applicable.
        """
        try:
            # Use the base class method to load the initial data
            df = super().load_data(data_source)

            # Basic validation
            required_columns = ["latitude", "longitude", "category"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                self.log(f"Data missing required columns: {missing_columns}", "error")
                raise ValueError(f"Data missing required columns: {missing_columns}")

            self.log(f"Loaded {len(df)} points of interest")

            # Process time windows if provided
            if time_windows_path and evaluation_time:
                df = self._apply_time_window_filter(
                    df, time_windows_path, evaluation_time
                )

            return df

        except Exception as e:
            self.log(f"Failed to load data: {e}", "error")
            raise ValueError(f"Failed to load data: {e}")

    def _apply_time_window_filter(self, df, time_windows_path, evaluation_time):
        """
        Apply time window filters to the dataset based on the evaluation time.

        Args:
            df (pd.DataFrame): Original POI dataframe.
            time_windows_path (str): Path to the time windows directory.
            evaluation_time (str or int): Evaluation time scenario key or epoch timestamp.

        Returns:
            pd.DataFrame: Filtered dataframe based on time window activity.
        """
        try:
            # Handle either scenario key or direct timestamp
            if (
                isinstance(evaluation_time, str)
                and evaluation_time in self.et_scenarios
            ):
                scenario_info = self.et_scenarios[evaluation_time]
                self.log(
                    f"Evaluating {scenario_info['name']}:\n{scenario_info['description']}"
                )
                epoch_time = scenario_info["datetime"]
            elif isinstance(evaluation_time, (int, float)):
                epoch_time = int(evaluation_time)
                self.log(f"Using direct epoch timestamp: {epoch_time}")
            else:
                # Try to convert string representation of timestamp to int
                try:
                    epoch_time = int(evaluation_time)
                    self.log(f"Using parsed epoch timestamp: {epoch_time}")
                except (ValueError, TypeError):
                    self.log(
                        f"Unknown evaluation time format: {evaluation_time}", "error"
                    )
                    raise ValueError(
                        f"Unknown evaluation time format: {evaluation_time}"
                    )

            self.log(f"Epoch Time: {epoch_time} UTC")

            # Load time windows data
            self.log(f"Loading time windows from: {time_windows_path}")
            time_windows_df = []

            for file in os.listdir(time_windows_path):
                if file.endswith(".csv"):
                    time_window = pd.read_csv(os.path.join(time_windows_path, file))
                    time_window["category"] = os.path.splitext(file)[0]
                    time_windows_df.append(time_window)

            if not time_windows_df:
                self.log("No time window files found", "warning")
                return df

            time_windows = pd.concat(time_windows_df, ignore_index=True)
            self.log(f"Loaded {len(time_windows)} time window entries")

            # Filter time windows for the selected time
            filtered_time_windows = time_windows[
                (time_windows["ts"] <= epoch_time) & (time_windows["te"] >= epoch_time)
            ]

            if filtered_time_windows.empty:
                self.log("No active categories for the selected time window", "warning")
                return df

            self.log(
                f"Found {len(filtered_time_windows)} active categories for the selected time"
            )

            # Apply vulnerability index to points based on their category
            category_vi_map = filtered_time_windows.set_index("category")[
                "vi"
            ].to_dict()

            # Add the VI values to the POI dataframe
            df["vi"] = df["category"].map(category_vi_map).fillna(0)

            # Keep only points with non-zero vulnerability index
            original_count = len(df)
            df = df[df["vi"] > 0]

            self.log(
                f"Filtered from {original_count} to {len(df)} points based on time window"
            )
            return df

        except Exception as e:
            self.log(f"Failed to apply time window filter: {e}", "error")
            raise ValueError(f"Failed to apply time window filter: {e}")

    def run_clustering(
        self, df, place_name=None, evaluation_time=None, save_output=False
    ):
        """
        Run OPTICS clustering on the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing points of interest with lat/lon coordinates.
            place_name (str, optional): Name of the location for output files.
            evaluation_time (str or int, optional): Time scenario key or timestamp for output files.
            save_output (bool): Whether to save the results to files.

        Returns:
            tuple: (cluster_df, centroids_df) - DataFrames with cluster assignments and centroids.
        """
        try:
            # Extract coordinates for clustering
            coords = df[["latitude", "longitude"]].to_numpy()

            if len(coords) < self.min_samples:
                self.log(
                    f"Not enough points for clustering: {len(coords)} < {self.min_samples}",
                    "warning",
                )
                return None, None

            self.log(f"Running OPTICS clustering on {len(coords)} points")

            # Convert to radians for haversine distance
            kms_per_radian = 6371.0088
            epsilon = 1.2 / kms_per_radian
            self.log(f"Using epsilon: {epsilon} radians")

            # Initialize and fit OPTICS model
            optics = OPTICS(
                min_samples=self.min_samples,
                metric="haversine",
                cluster_method="xi",
                xi=self.xi,
                min_cluster_size=self.min_cluster_size,
            )

            optics.fit(np.radians(coords))
            cluster_labels = optics.labels_

            # Assign noise points to nearest clusters using KNN
            self.log("Assigning noise points to nearest clusters using KNN")
            core_mask = cluster_labels != -1
            knn = KNeighborsClassifier(n_neighbors=5)

            # Only train KNN if there are core points
            if np.any(core_mask):
                knn.fit(coords[core_mask], cluster_labels[core_mask])

                # Predict cluster labels for noise points
                noise_mask = cluster_labels == -1
                if np.any(noise_mask):
                    cluster_labels[noise_mask] = knn.predict(coords[noise_mask])

            # Count unique clusters (excluding noise points labeled -1)
            unique_clusters = np.unique(cluster_labels)
            num_clusters = len([c for c in unique_clusters if c != -1])
            self.log(f"Found {num_clusters} clusters")

            # Create cluster dataframe
            cluster_data = []
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Skip noise points if any remain
                    cluster_data.append([coords[i][0], coords[i][1], int(label)])

            if not cluster_data:
                self.log("No clusters found after filtering noise points", "warning")
                return None, None

            # Rename 'cluster' to 'cluster_id' for consistency with the base class
            cluster_df = pd.DataFrame(
                cluster_data, columns=["latitude", "longitude", "cluster_id"]
            )

            # Merge with original data to include category, name, and vi
            columns_to_include = ["latitude", "longitude"]
            for col in ["category", "name", "vi"]:
                if col in df.columns:
                    columns_to_include.append(col)

            cluster_df = cluster_df.merge(
                df[columns_to_include],
                on=["latitude", "longitude"],
                how="left",
            )

            # Calculate cluster centroids
            centroids = []
            cluster_ids = []
            cluster_sizes = []

            for label in unique_clusters:
                if label != -1:  # Skip noise points
                    cluster_points = coords[cluster_labels == label]
                    if len(cluster_points) > 0:
                        centroid = cluster_points.mean(axis=0)
                        centroids.append(centroid)
                        cluster_ids.append(int(label))
                        cluster_sizes.append(len(cluster_points))

            if not centroids:
                self.log("No valid centroids found", "warning")
                return cluster_df, None

            centroids_df = pd.DataFrame(centroids, columns=["latitude", "longitude"])
            centroids_df["cluster_id"] = cluster_ids
            # Add size for consistency with base class
            centroids_df["size"] = cluster_sizes
            centroids_df = centroids_df[["cluster_id", "latitude", "longitude", "size"]]

            # Save results if requested - using the base class method
            if save_output and place_name:
                self.save_results(
                    cluster_df, centroids_df, place_name, evaluation_time, "OPTICS"
                )

            return cluster_df, centroids_df

        except Exception as e:
            self.log(f"Error in clustering: {e}", "error")
            return None, None

    def create_optics_map(self, cluster_df, centroids_df=None, area_boundary_path=None):
        """
        Create a specialized OPTICS map with custom coloring.

        Args:
            cluster_df (pd.DataFrame): DataFrame with cluster assignments.
            centroids_df (pd.DataFrame, optional): DataFrame with cluster centroids.
            area_boundary_path (str, optional): Path to area boundary GeoJSON file.

        Returns:
            folium.Map: Interactive map with clusters and centroids.
        """
        try:
            import folium
            from branca.colormap import linear

            if cluster_df is None or len(cluster_df) == 0:
                self.log("No data available to create map", "warning")
                return None

            # Get center of the map
            center_lat = cluster_df["latitude"].mean()
            center_lon = cluster_df["longitude"].mean()

            # Create map
            self.log("Creating interactive map")
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="cartodbpositron",
            )

            # Add boundary if provided
            if area_boundary_path and os.path.exists(area_boundary_path):
                folium.GeoJson(
                    area_boundary_path,
                    name="Boundary",
                    style_function=lambda x: {
                        "fillColor": "transparent",
                        "color": "blue",
                        "weight": 2,
                    },
                ).add_to(m)

            # Add cluster points - create a colormap for clusters
            self.log("Adding cluster points to map")
            num_clusters = len(cluster_df["cluster_id"].unique())
            colors = linear.viridis.scale(0, max(1, num_clusters - 1))

            # Add points by cluster with consistent colors but without clustering
            cluster_groups = {}  # Store feature groups for each cluster

            # Create feature groups for each cluster
            for cluster_id in sorted(cluster_df["cluster_id"].unique()):
                # Create a feature group for this cluster
                cluster_groups[cluster_id] = folium.FeatureGroup(
                    name=f"Cluster {cluster_id}"
                )

            # Add points to their respective feature groups
            for i, row in cluster_df.iterrows():
                cluster_id = row["cluster_id"]
                color = colors(cluster_id)

                # Create popup content with vulnerability index
                popup_content = f"""
                <b>{row.get('name', 'Unknown')}</b><br>
                Category: {row.get('category', 'Unknown')}<br>
                VI: {row.get('vi', 'N/A')}
                """

                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Cluster {cluster_id}: {row.get('name', 'Unknown')}",
                ).add_to(cluster_groups[cluster_id])

            # Add all feature groups to the map
            for cluster_id, feature_group in cluster_groups.items():
                feature_group.add_to(m)

            # Add centroids if provided
            if centroids_df is not None and not centroids_df.empty:
                centroid_group = folium.FeatureGroup(name="Cluster Centroids")

                for i, row in centroids_df.iterrows():
                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        icon=folium.Icon(color="red", icon="info-sign"),
                        popup=f"Centroid {row['cluster_id']}: {row.get('size', 'N/A')} points",
                    ).add_to(centroid_group)

                centroid_group.add_to(m)

            # Add layer control to toggle visibility
            folium.LayerControl().add_to(m)

            return m

        except Exception as e:
            self.log(f"Error creating map: {e}", "error")
            return None

    def run(
        self,
        data_source,
        place_name=None,
        time_windows_path=None,
        evaluation_time=None,
        save_output=False,
        create_map=False,
        area_boundary_path=None,  # Add boundary path parameter for consistency
    ):
        """
        Run the complete clustering workflow.

        Args:
            data_source (str or pd.DataFrame): Path to POI CSV file or DataFrame from extraction.py
            place_name (str, optional): Name of the location for output files.
            time_windows_path (str, optional): Path to the time windows directory.
            evaluation_time (str or int, optional): Time scenario key (e.g., "ET4") or timestamp.
            save_output (bool): Whether to save the results to files.
            create_map (bool): Whether to create and return a map.
            area_boundary_path (str, optional): Path to area boundary GeoJSON file.

        Returns:
            dict: Dictionary containing clustering results:
                - 'clusters': DataFrame with cluster assignments
                - 'centroids': DataFrame with cluster centroids
                - 'map': Folium map object (if create_map is True)
                - 'input_data': Filtered input data used for clustering
        """
        try:
            # Extract place name from path if not provided
            if place_name is None:
                if isinstance(data_source, str):
                    import os

                    filename = os.path.basename(data_source)
                    if "_dataset" in filename:
                        place_name = filename.split("_dataset")[0]
                    else:
                        place_name = "unknown"
                else:
                    place_name = "unknown"

            # Load and filter data
            df = self.load_data(data_source, time_windows_path, evaluation_time)

            # Run clustering
            cluster_df, centroids_df = self.run_clustering(
                df, place_name, evaluation_time, save_output
            )

            # Create map if requested
            map_obj = None
            if create_map and cluster_df is not None:
                # Use either the specialized OPTICS map or the base class map
                map_obj = self.create_optics_map(
                    cluster_df, centroids_df, area_boundary_path
                )

                # Save map if requested
                if save_output and map_obj:
                    # Format the evaluation time for filename
                    suffix = evaluation_time if evaluation_time else ""

                    # Use the paths manager from base class
                    map_path = self.paths.get_path(
                        "maps", f"{place_name}_Map_OPTICS_{suffix}.html"
                    )

                    map_obj.save(map_path)
                    self.log(f"Saved map to: {map_path}", "success")

            # Return filtered input data as well for analysis
            return {
                "clusters": cluster_df,
                "centroids": centroids_df,
                "map": map_obj,
                "input_data": df,
            }

        except Exception as e:
            self.log(f"Error in clustering workflow: {e}", "error")
            return {
                "clusters": None,
                "centroids": None,
                "map": None,
                "input_data": None,
            }
