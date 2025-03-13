"""
OPTICS clustering module for the Verus project.

This module provides functionality for applying OPTICS (Ordering Points To Identify
the Clustering Structure) algorithm to geospatial data, with optional time-based
vulnerability indexing.
"""

import os
from datetime import datetime

import folium
import numpy as np
import pandas as pd
from branca.colormap import linear
from sklearn.cluster import OPTICS
from sklearn.neighbors import KNeighborsClassifier

from verus.utils.logger import Logger


class GeOPTICS(Logger):
    """
    Perform OPTICS clustering on geospatial data with time-based vulnerability indexing.

    This class implements OPTICS (Ordering Points To Identify the Clustering Structure)
    algorithm to cluster points of interest based on their spatial proximity, with
    the option to filter points based on time-specific vulnerability indices.

    Attributes:
        min_samples (int): The minimum number of samples in a neighborhood for a point
                           to be considered a core point.
        xi (float): Determines the minimum steepness on the reachability plot that
                    constitutes a cluster boundary.
        min_cluster_size (int): Minimum number of points to form a cluster.
        et_scenarios (dict): Dictionary of evaluation time scenarios for time-based analysis.

    Examples:
        >>> optics = GeOPTICS(min_samples=5, xi=0.05)
        >>> results = optics.run(data_source=poi_df, time_windows_path="./time_windows")
        >>> clusters_df = results["clusters"]
        >>> map_obj = results["map"]
    """

    def __init__(
        self,
        min_samples=5,
        xi=0.05,
        min_cluster_size=5,
        verbose=True,
        et_scenarios=None,
    ):
        """
        Initialize the GeOPTICS clusterer with clustering parameters.

        Parameters
        ----------
        min_samples : int
            The number of samples in a neighborhood for a point to be considered a core point.
        xi : float
            Determines the minimum steepness on the reachability plot that constitutes
            a cluster boundary. Must be between 0 and 1.
        min_cluster_size : int
            Minimum number of points to form a cluster.
        verbose : bool
            Whether to print informational messages.
        et_scenarios : dict, optional
            Custom evaluation time scenarios dictionary.

        Raises
        ------
        ValueError
            If any parameter values are invalid.
        """
        # Initialize the Logger
        super().__init__(verbose=verbose)

        # Validate parameters
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

    def load(self, data_source, time_windows=None, evaluation_time=None):
        """
        Load POI data and apply optional time window filtering.

        Parameters
        ----------
        data_source : str or pd.DataFrame
            Path to POI CSV file or DataFrame containing point data.
        time_windows : str or dict or pd.DataFrame, optional
            Time windows data. Can be:
            - Path to time windows directory
            - Dictionary of category-based time window DataFrames
            - Single DataFrame with 'category', 'ts', 'te', 'vi' columns
        evaluation_time : str or int, optional
            Time scenario key (e.g., "ET4") or epoch timestamp.

        Returns
        -------
        pd.DataFrame
            DataFrame containing points of interest, filtered by time if applicable.

        Raises
        ------
        ValueError
            If the data cannot be loaded or processed.
        """
        try:
            # Load data from path or use provided DataFrame
            if isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
                self.log("Using provided DataFrame")
            elif isinstance(data_source, str) and os.path.exists(data_source):
                self.log(f"Loading data from file: {data_source}")
                df = pd.read_csv(data_source)
            else:
                raise ValueError(f"Invalid data source: {data_source}")

            # Basic validation
            required_columns = ["latitude", "longitude", "category"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                self.log(f"Data missing required columns: {missing_columns}", "error")
                raise ValueError(f"Data missing required columns: {missing_columns}")

            self.log(f"Loaded {len(df)} points of interest")

            # Process time windows if provided
            if time_windows is not None and evaluation_time:
                df = self._apply_time_window_filter(df, time_windows, evaluation_time)

            return df

        except Exception as e:
            self.log(f"Failed to load data: {str(e)}", "error")
            raise ValueError(f"Failed to load data: {str(e)}")

    def _apply_time_window_filter(self, df, time_windows, evaluation_time):
        """
        Apply time window filters to the dataset based on the evaluation time.

        Parameters
        ----------
        df : pd.DataFrame
            Original POI dataframe.
        time_windows : str or dict or pd.DataFrame
            Time windows data. Can be:
            - Path to time windows directory
            - Dictionary of category-based time window DataFrames
            - Single DataFrame with 'category', 'ts', 'te', 'vi' columns
        evaluation_time : str or int
            Evaluation time scenario key or epoch timestamp.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe based on time window activity.

        Raises
        ------
        ValueError
            If the time window processing fails.
        """
        try:
            # Resolve evaluation time to epoch timestamp
            epoch_time = self._resolve_evaluation_time(evaluation_time)

            # Format the time for logging
            time_str = datetime.fromtimestamp(epoch_time).strftime("%Y-%m-%d %H:%M:%S")
            self.log(f"Epoch Time: {epoch_time} ({time_str})")

            # Load time windows data based on input type
            time_windows_df = self._load_time_windows(time_windows)

            if time_windows_df is None or time_windows_df.empty:
                self.log("No time window data available", "warning")
                return df

            # Filter time windows for the selected time
            filtered_time_windows = time_windows_df[
                (time_windows_df["ts"] <= epoch_time)
                & (time_windows_df["te"] >= epoch_time)
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
            self.log(f"Failed to apply time window filter: {str(e)}", "error")
            raise ValueError(f"Failed to apply time window filter: {str(e)}")

    def _resolve_evaluation_time(self, evaluation_time):
        """
        Convert evaluation time to epoch timestamp.

        Parameters
        ----------
        evaluation_time : str or int or float
            Evaluation time identifier or timestamp.

        Returns
        -------
        int
            Epoch timestamp.
        """
        # Handle either scenario key or direct timestamp
        if isinstance(evaluation_time, str) and evaluation_time in self.et_scenarios:
            scenario_info = self.et_scenarios[evaluation_time]
            self.log(
                f"Evaluating {scenario_info['name']}:\n{scenario_info['description']}"
            )
            return scenario_info["datetime"]
        elif isinstance(evaluation_time, (int, float)):
            return int(evaluation_time)
        else:
            # Try to convert string representation of timestamp to int
            try:
                return int(evaluation_time)
            except (ValueError, TypeError):
                self.log(f"Unknown evaluation time format: {evaluation_time}", "error")
                raise ValueError(f"Unknown evaluation time format: {evaluation_time}")

    def _load_time_windows(self, time_windows):
        """
        Load time windows from various input formats.

        Parameters
        ----------
        time_windows : str or dict or pd.DataFrame
            Time windows data in various formats.

        Returns
        -------
        pd.DataFrame or None
            Processed time windows DataFrame or None if loading fails.
        """
        # Case 1: DataFrame already provided
        if isinstance(time_windows, pd.DataFrame):
            # Validate required columns
            required_cols = ["category", "ts", "te", "vi"]
            missing_cols = [
                col for col in required_cols if col not in time_windows.columns
            ]

            if missing_cols:
                self.log(
                    f"Time windows DataFrame missing columns: {missing_cols}", "warning"
                )
                return None

            self.log(
                f"Using provided time windows DataFrame with {len(time_windows)} entries"
            )
            return time_windows

        # Case 2: Dictionary of DataFrames by category
        elif isinstance(time_windows, dict):
            dfs = []
            for category, df in time_windows.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Add category column if not present
                    if "category" not in df.columns:
                        df = df.copy()
                        df["category"] = category
                    dfs.append(df)

            if not dfs:
                self.log("No valid DataFrames in time windows dictionary", "warning")
                return None

            combined_df = pd.concat(dfs, ignore_index=True)
            self.log(
                f"Using provided dictionary with {len(combined_df)} time window entries"
            )
            return combined_df

        # Case 3: Path to directory with CSV files
        elif isinstance(time_windows, str) and os.path.exists(time_windows):
            self.log(f"Loading time windows from directory: {time_windows}")
            time_windows_df = []

            for file in os.listdir(time_windows):
                if file.endswith(".csv"):
                    try:
                        file_path = os.path.join(time_windows, file)
                        time_window = pd.read_csv(file_path)
                        # Extract category from filename
                        category = os.path.splitext(file)[0]
                        # Add category if not present
                        if "category" not in time_window.columns:
                            time_window["category"] = category
                        time_windows_df.append(time_window)
                    except Exception as e:
                        self.log(f"Error loading file {file}: {str(e)}", "warning")

            if not time_windows_df:
                self.log("No time window files found", "warning")
                return None

            combined_df = pd.concat(time_windows_df, ignore_index=True)
            self.log(f"Loaded {len(combined_df)} time window entries from files")
            return combined_df

        # Invalid input
        else:
            self.log(
                f"Invalid time windows input format: {type(time_windows)}", "warning"
            )
            return None

    def fit(self, df):
        """
        Run OPTICS clustering on the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing points of interest with lat/lon coordinates.

        Returns
        -------
        tuple
            (cluster_df, centroids_df) - DataFrames with cluster assignments and centroids.
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

            cluster_df = pd.DataFrame(
                cluster_data, columns=["latitude", "longitude", "cluster"]
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
            clusters = []
            cluster_sizes = []

            for label in unique_clusters:
                if label != -1:  # Skip noise points
                    cluster_points = coords[cluster_labels == label]
                    if len(cluster_points) > 0:
                        centroid = cluster_points.mean(axis=0)
                        centroids.append(centroid)
                        clusters.append(int(label))
                        cluster_sizes.append(len(cluster_points))

            if not centroids:
                self.log("No valid centroids found", "warning")
                return cluster_df, None

            centroids_df = pd.DataFrame(centroids, columns=["latitude", "longitude"])
            centroids_df["cluster"] = clusters
            centroids_df["size"] = cluster_sizes
            centroids_df = centroids_df[["cluster", "latitude", "longitude", "size"]]

            return cluster_df, centroids_df

        except Exception as e:
            self.log(f"Error in clustering: {str(e)}", "error")
            return None, None

    def view(
        self, cluster_df, centroids_df=None, area_boundary_path=None, save_path=None
    ):
        """
        Create an interactive map visualizing the clustering results.

        Parameters
        ----------
        cluster_df : pd.DataFrame
            DataFrame with cluster assignments.
        centroids_df : pd.DataFrame, optional
            DataFrame with cluster centroids.
        area_boundary_path : str, optional
            Path to area boundary GeoJSON file.
        save_path : str, optional
            Path to save the map HTML file. If None, the map is just returned.

        Returns
        -------
        folium.Map or None
            Interactive map with clusters and centroids, or None on error.
        """
        try:
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

            # Create a colormap for clusters
            self.log("Adding cluster points to map")
            num_clusters = len(cluster_df["cluster"].unique())
            colors = linear.viridis.scale(0, max(1, num_clusters - 1))

            # Create feature groups for each cluster
            cluster_groups = {}
            for cluster in sorted(cluster_df["cluster"].unique()):
                cluster_groups[cluster] = folium.FeatureGroup(name=f"Cluster {cluster}")

            # Add points to their respective feature groups
            for i, row in cluster_df.iterrows():
                cluster = row["cluster"]
                color = colors(cluster)

                # Create popup content
                popup_content = f"""
                <b>{row.get('name', 'Unknown')}</b><br>
                Category: {row.get('category', 'Unknown')}
                """
                if "vi" in row:
                    popup_content += f"<br>Vulnerability Index: {row['vi']}"

                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3,
                    color=color,
                    stroke=False,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Cluster {cluster}: {row.get('category', 'Unknown')}",
                ).add_to(cluster_groups[cluster])

            # Add all feature groups to the map
            for cluster, feature_group in cluster_groups.items():
                feature_group.add_to(m)

            # Add centroids if provided
            if centroids_df is not None and not centroids_df.empty:
                centroid_group = folium.FeatureGroup(name="Cluster Centroids")

                for i, row in centroids_df.iterrows():
                    cluster_num = int(row["cluster"])
                    cluster_size = row.get("size", "N/A")

                    # Create a DivIcon with the cluster number inside
                    icon = folium.DivIcon(
                        icon_size=(40, 40),
                        icon_anchor=(20, 20),
                        html=f"""
                            <div style="
                                background-color: #A52A2A;
                                width: 20px;
                                height: 20px;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                border-radius: 50%;
                                color: white;
                                font-weight: normal;
                                font-size: 10px;
                            ">
                                {cluster_num}
                            </div>
                        """,
                    )

                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        icon=icon,
                        popup=f"Centroid {cluster_num}: {cluster_size} points",
                        tooltip=f"Centroid {cluster_num}: {cluster_size} points",
                    ).add_to(centroid_group)

                centroid_group.add_to(m)

            # Add layer control to toggle visibility
            folium.LayerControl().add_to(m)

            # Save map if path provided
            if save_path:
                try:
                    # Create directory if it doesn't exist
                    save_dir = os.path.dirname(os.path.abspath(save_path))
                    os.makedirs(save_dir, exist_ok=True)

                    # Save the map
                    m.save(save_path)
                    self.log(f"Saved map to: {save_path}", "success")
                except Exception as e:
                    self.log(f"Failed to save map: {str(e)}", "error")

            return m

        except Exception as e:
            self.log(f"Error creating map: {str(e)}", "error")
            return None

    def save(self, cluster_df, centroids_df, path=None, evaluation_time=None):
        """
        Save clustering results to CSV files.

        Parameters
        ----------
        cluster_df : pd.DataFrame
            DataFrame with cluster assignments.
        centroids_df : pd.DataFrame
            DataFrame with cluster centroids.
        path : str, optional
            Directory or file path to save results.
            If None, uses "./data/clusters".
        evaluation_time : str, optional
            Time scenario identifier to include in filenames.

        Returns
        -------
        dict
            Dictionary with paths to saved files.
        """
        if cluster_df is None:
            self.log("No cluster data to save", "warning")
            return {}

        try:
            # Determine base directory and file names
            if path is None:
                save_dir = os.path.abspath("./data/clusters")
                filename_prefix = "optics"
            elif os.path.isdir(path) or not path.endswith(".csv"):
                save_dir = os.path.abspath(path)
                filename_prefix = "optics"
            else:
                save_dir = os.path.dirname(os.path.abspath(path))
                filename_prefix = os.path.splitext(os.path.basename(path))[0]

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Format evaluation time for filename
            time_suffix = f"_{evaluation_time}" if evaluation_time else ""

            # Save cluster assignments
            clusters_path = os.path.join(
                save_dir, f"{filename_prefix}_clusters{time_suffix}.csv"
            )
            cluster_df.to_csv(clusters_path, index=False)
            self.log(f"Saved clusters to: {clusters_path}", "success")

            saved_files = {"clusters": clusters_path}

            # Save centroids if available
            if centroids_df is not None and not centroids_df.empty:
                centroids_path = os.path.join(
                    save_dir, f"{filename_prefix}_centroids{time_suffix}.csv"
                )
                centroids_df.to_csv(centroids_path, index=False)
                self.log(f"Saved centroids to: {centroids_path}", "success")
                saved_files["centroids"] = centroids_path

            return saved_files

        except Exception as e:
            self.log(f"Failed to save results: {str(e)}", "error")
            return {}

    def run(
        self,
        data_source,
        time_windows=None,
        evaluation_time=None,
        area_boundary_path=None,
    ):
        """
        Run the complete clustering workflow.

        Parameters
        ----------
        data_source : str or pd.DataFrame
            Path to POI CSV file or DataFrame from extraction.py
        time_windows : str or dict or pd.DataFrame, optional
            Time windows data. Can be:
            - Path to time windows directory
            - Dictionary of category-based time window DataFrames
            - Single DataFrame with 'category', 'ts', 'te', 'vi' columns
        evaluation_time : str or int, optional
            Time scenario key (e.g., "ET4") or timestamp.
        area_boundary_path : str, optional
            Path to area boundary GeoJSON file for map visualization.

        Returns
        -------
        dict
            Dictionary containing clustering results:
                - 'clusters': DataFrame with cluster assignments
                - 'centroids': DataFrame with cluster centroids
                - 'map': Folium map object if area_boundary_path was provided
                - 'input_data': Filtered input data used for clustering
                - 'place_name': Extracted place name from source (if available)
        """
        try:
            # Extract place name from path if possible
            place_name = self._extract_place_name(data_source)

            # Load and filter data
            df = self.load(data_source, time_windows, evaluation_time)

            # Run clustering
            cluster_df, centroids_df = self.fit(df)

            # Create map if boundary path was provided
            map_obj = None
            if area_boundary_path and cluster_df is not None:
                map_obj = self.view(cluster_df, centroids_df, area_boundary_path)

            # Return results
            return {
                "clusters": cluster_df,
                "centroids": centroids_df,
                "map": map_obj,
                "input_data": df,
                "place_name": place_name,
            }

        except Exception as e:
            self.log(f"Error in clustering workflow: {str(e)}", "error")
            return {
                "clusters": None,
                "centroids": None,
                "map": None,
                "input_data": None,
                "place_name": None,
            }

    def _extract_place_name(self, data_source):
        """
        Extract place name from data source if possible.

        Parameters
        ----------
        data_source : str or pd.DataFrame
            Data source to extract place name from.

        Returns
        -------
        str
            Extracted place name or "unknown".
        """
        if isinstance(data_source, str):
            filename = os.path.basename(data_source)
            if "_dataset" in filename:
                return filename.split("_dataset")[0]
        return "unknown"
