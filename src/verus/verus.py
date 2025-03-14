import os

import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import linear
from haversine import Unit, haversine

from verus.utils.logger import Logger


class VERUS(Logger):
    """
    Main class for assessing urban vulnerability using POI clustering and spatial analysis.

    This class handles the entire vulnerability assessment workflow including:
    - Accepting data as DataFrames or GeoDataFrames
    - Computing vulnerability levels using various distance methods
    - Visualizing and exporting results

    The workflow consists of these main steps:
    1. Load data (POIs, centroids, zones) using the `load()` method
    2. Update vulnerability indices based on time windows
    3. Run clustering using OPTICS followed by KMeans
    4. Calculate vulnerability for each zone
    5. Apply smoothing to improve spatial continuity
    6. Export or visualize results

    Attributes:
        place_name (str): Name of the place to analyze
        method (str): Clustering method used (e.g., "KM-OPTICS")
        distance_method (str): Method for calculating vulnerability based on distance
            Options: "gaussian", "inverse_weighted"
        sigma (float): Bandwidth parameter for Gaussian methods (in meters)
        config (dict): Configuration parameters
        verbose (bool): Whether to print verbose logging
    """

    # Define default configuration
    DEFAULT_CONFIG = {
        "max_vulnerability": {},  # Map of place_name -> max vulnerability value
        "map_center": {},  # Map of place_name -> {"lat": float, "lon": float}
        "boundary_paths": {},  # Map of place_name -> boundary file path
        "clustering": {
            "optics": {"min_samples": 5, "xi": 0.05, "min_cluster_size": 5},
            "kmeans": {"n_clusters": 8, "init": "k-means++", "random_state": 42},
        },
    }

    DISTANCE_METHODS = {
        "gaussian": "_gaussian_weighted_vulnerability",
        "inverse_weighted": "_inversely_weighted_distance",
    }

    def __init__(
        self,
        place_name,
        method="KM-OPTICS",
        distance_method="gaussian",
        sigma=1000,
        config=None,
        verbose=True,
    ):
        super().__init__()
        if distance_method not in self.DISTANCE_METHODS:
            valid_methods = ", ".join(self.DISTANCE_METHODS.keys())
            raise ValueError(
                f"Invalid distance method: {distance_method}. Choose from: {valid_methods}"
            )
        self.place_name = place_name
        self.method = method
        self.distance_method = distance_method
        self.sigma = sigma
        self.config = config
        self.verbose = verbose  # Initialize verbose attribute

        # Initialize configuration with defaults, then update with user-provided config
        self.config = self._init_config(config)

        # Data containers (to be loaded from DataFrames)
        self.poti_df = None
        self.cluster_centers = None
        self.vulnerability_zones = None
        self.time_windows = None

        # Results storage
        self.results = {}

        self.log(
            f"VulnerabilityAssessor initialized for {place_name} using {distance_method} method"
        )

    def _init_config(self, user_config=None):
        """
        Initialize configuration with defaults and user overrides.

        Args:
            user_config (dict, optional): User-provided configuration that overrides defaults

        Returns:
            dict: Complete configuration dictionary
        """
        # Start with a deep copy of the default config
        import copy

        config = copy.deepcopy(self.DEFAULT_CONFIG)

        # Update with user-provided config if available
        if user_config:
            self._deep_update(config, user_config)

        return config

    def _deep_update(self, d, u):
        """
        Recursively update a nested dictionary with another dictionary.

        Args:
            d (dict): Dictionary to update
            u (dict): Dictionary with updates

        Returns:
            dict: Updated dictionary
        """
        import collections.abc

        for k, v in u.items():
            if (
                isinstance(v, collections.abc.Mapping)
                and k in d
                and isinstance(d[k], dict)
            ):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def _get_config(self, *keys, default=None):
        """
        Safely get a configuration value by path.

        Args:
            *keys: Sequence of keys to navigate the config dictionary
            default: Default value to return if path doesn't exist

        Returns:
            The configuration value or default
        """
        current = self.config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def load(
        self,
        potis_df=None,
        centroids_df=None,
        zones_gdf=None,
        time_windows_df=None,
        time_windows_dict=None,
    ):
        """
        Load POI, cluster centroids, vulnerability zones, and time windows data.

        This method accepts pre-constructed DataFrames rather than reading from files,
        providing a more flexible workflow for data integration.

        Args:
            potis_df (pd.DataFrame): DataFrame with points of interest. Must contain at least
                "latitude", "longitude", "category" and optionally "vi" and "cluster".
            centroids_df (pd.DataFrame): DataFrame with cluster centroids. Must contain
                "latitude" and "longitude".
            zones_gdf (GeoDataFrame): GeoDataFrame with vulnerability zones geometry.
            time_windows_df (pd.DataFrame, optional): DataFrame with time window data.
                Must have "category" and "vi" columns.
            time_windows_dict (dict, optional): Dictionary of time windows where keys are
                categories and values are DataFrames with "vi" column.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If required columns are missing or data is invalid.
        """
        # Load POTIs DataFrame
        if potis_df is not None:
            # Validate POTIs DataFrame
            required_poti_columns = ["latitude", "longitude", "category"]
            missing_columns = [
                col for col in required_poti_columns if col not in potis_df.columns
            ]
            if missing_columns:
                self.log(
                    f"POTI DataFrame missing required columns: {', '.join(missing_columns)}",
                    level="error",
                )
                raise ValueError(
                    f"POTI DataFrame missing required columns: {', '.join(missing_columns)}"
                )
            self.poti_df = potis_df.copy()

            # Ensure vulnerability index column exists
            if "vi" not in self.poti_df.columns:
                self.log(
                    "POTI DataFrame doesn't have 'vi' column. Using default vi=1.",
                    level="warning",
                )
                self.poti_df["vi"] = 0.0

            # Set default cluster if not present
            if "cluster" not in self.poti_df.columns:
                self.log(
                    "POTI DataFrame doesn't have 'cluster' column. Setting all to 0.",
                    level="warning",
                )
                self.poti_df["cluster"] = 0

            # Log summary info
            self.log(f"Loaded {len(self.poti_df)} POTIs", level="info")

        # Load centroids DataFrame
        if centroids_df is not None:
            # Validate centroids DataFrame
            required_centroid_columns = ["latitude", "longitude"]
            missing_columns = [
                col
                for col in required_centroid_columns
                if col not in centroids_df.columns
            ]
            if missing_columns:
                self.log(
                    f"Centroids DataFrame missing required columns: {', '.join(missing_columns)}",
                    level="error",
                )
                raise ValueError(
                    f"Centroids DataFrame missing required columns: {', '.join(missing_columns)}"
                )
            self.cluster_centers = centroids_df.copy()
            self.log(
                f"Loaded {len(self.cluster_centers)} cluster centers", level="info"
            )

        # Load vulnerability zones GeoDataFrame
        if zones_gdf is not None:
            # Validate vulnerability zones GeoDataFrame
            if not isinstance(zones_gdf, gpd.GeoDataFrame):
                self.log("Input zones data must be a GeoDataFrame", level="error")
                raise ValueError("Zones data must be a GeoDataFrame")
            self.vulnerability_zones = zones_gdf.copy()
            if not all(self.vulnerability_zones.geometry.is_valid):
                self.log(
                    "Vulnerability zones contain invalid geometries", level="error"
                )
                raise ValueError("Vulnerability zones contain invalid geometries")
            self.log(
                f"Loaded {len(self.vulnerability_zones)} vulnerability zones",
                level="info",
            )

        # Load time windows
        if time_windows_df is not None:
            self._load_time_windows_from_df(time_windows_df)
        elif time_windows_dict is not None:
            self._load_time_windows_from_dict(time_windows_dict)

        # Log summary info if all data is loaded
        if self.poti_df is not None and self.cluster_centers is not None:
            n_clusters = len(np.unique(self.poti_df["cluster"]))
            self.log(
                f"Successfully loaded {len(self.poti_df)} POTIs across {n_clusters} clusters",
                level="success",
            )

        return self

    def _load_time_windows_from_df(self, time_windows_df):
        """
        Load time windows from a DataFrame.

        Args:
            time_windows_df (pd.DataFrame): DataFrame with time window data.
                Must have "category" and "vi" columns.

        Returns:
            bool: Whether time windows were successfully loaded
        """
        if not isinstance(time_windows_df, pd.DataFrame):
            self.log("Time windows must be a DataFrame", level="error")
            return False

        if (
            "category" not in time_windows_df.columns
            or "vi" not in time_windows_df.columns
        ):
            self.log(
                "Time windows DataFrame must have 'category' and 'vi' columns",
                level="error",
            )
            return False

        self.time_windows = time_windows_df.copy()
        self.log(f"Loaded {len(self.time_windows)} time window entries", level="info")
        return True

    def _load_time_windows_from_dict(self, time_windows_dict):
        """
        Load time windows from a dictionary.

        Args:
            time_windows_dict (dict): Dictionary where keys are categories
                and values are DataFrames with "vi" column.

        Returns:
            bool: Whether time windows were successfully loaded
        """
        if not isinstance(time_windows_dict, dict):
            self.log("Time windows dict must be a dictionary", level="error")
            return False

        dfs = []
        for category, tw_df in time_windows_dict.items():
            if not isinstance(tw_df, pd.DataFrame) or "vi" not in tw_df.columns:
                self.log(
                    f"Time window for category '{category}' must have 'vi' column",
                    level="warning",
                )
                continue

            # Create a new DataFrame with category and vi columns
            category_df = pd.DataFrame(
                {"category": [category] * len(tw_df), "vi": tw_df["vi"].values}
            )

            # Add timestamps if available
            for col in ["ts", "te"]:
                if col in tw_df.columns:
                    category_df[col] = tw_df[col].values

            dfs.append(category_df)

        if dfs:
            self.time_windows = pd.concat(dfs, ignore_index=True)
            self.log(f"Loaded time windows for {len(dfs)} categories", level="info")
            return True
        else:
            self.log("No valid time windows found in dictionary", level="warning")
            return False

    def _apply_time_windows_to_potis(self, evaluation_time=None):
        """
        Apply time windows to update POTIs vulnerability indices based on evaluation time.

        Args:
            evaluation_time (str, optional): Time scenario to evaluate.
                If None, will use all time windows.

        Returns:
            pd.DataFrame: Updated POTIs DataFrame
        """
        if self.poti_df is None:
            self.log("No POTIs loaded to apply time windows", level="warning")
            return None

        if self.time_windows is None:
            self.log("No time windows loaded, using original vi values", level="info")
            return self.poti_df.copy()

        # Make a copy of the POTIs DataFrame to modify
        updated_df = self.poti_df.copy()

        # Filter time windows by evaluation time if provided
        filtered_tw = self.time_windows
        if evaluation_time is not None and "ts" in self.time_windows.columns:
            filtered_tw = self.time_windows[
                (self.time_windows["ts"] <= evaluation_time)
                & (self.time_windows["te"] >= evaluation_time)
            ]
            self.log(
                f"Filtered to {len(filtered_tw)} time windows for {evaluation_time}",
                level="info",
            )

        # Get unique category-vi pairs
        unique_vis = filtered_tw[["category", "vi"]].drop_duplicates()

        # Update vi values based on category
        updated_df = updated_df.merge(
            unique_vis, on="category", how="left", suffixes=("", "_tw")
        )

        # Use new vi if available, otherwise keep original
        if "vi_tw" in updated_df.columns:
            updated_df["vi"] = updated_df["vi_tw"].fillna(updated_df["vi"])
            updated_df.drop(columns=["vi_tw"], inplace=True)

        self.log(f"Applied time windows to {len(updated_df)} POTIs", level="info")
        return updated_df

    def _gaussian_weighted_vulnerability(self, x_c_z, y_c_z, potis):
        """
        Calculate the vulnerability level (VL) of a zone using a Gaussian approach.

        Parameters:
            x_c_z: latitude of the zone center
            y_c_z: longitude of the zone center
            potis: DataFrame with points of interest (latitude, longitude, vi)

        Returns:
            Vulnerability level (VL) of the zone
        """
        gaussian_kernel_sum = 0
        potis = potis.reset_index(drop=True)
        n = len(potis)

        if n == 0:
            return 0

        # Check if 'vi' column exists, add default if not
        if "vi" not in potis.columns:
            self.log(
                "POTIs DataFrame doesn't have 'vi' column. Using default vi=1.",
                level="warning",
            )
            potis["vi"] = 1.0

        for _, row in potis.iterrows():
            # Calculate the distance in meters
            distance = haversine(
                (x_c_z, y_c_z), (row["latitude"], row["longitude"]), unit=Unit.METERS
            )

            # Calculate the Gaussian influence
            influence = row["vi"] * np.exp(-0.5 * (distance / self.sigma) ** 2)
            gaussian_kernel_sum += influence

        # Normalize by the sum of influences to obtain the vulnerability level
        VL = gaussian_kernel_sum / (n * self.sigma * np.sqrt(2 * np.pi))
        return VL

    def _inversely_weighted_distance(self, x_c_z, y_c_z, potis):
        """
        Calculate the vulnerability level (VL) of a zone using modified IDW.

        Parameters:
            x_c_z: latitude of the zone center
            y_c_z: longitude of the zone center
            potis: DataFrame with points of interest (latitude, longitude, vi)

        Returns:
            Normalized vulnerability level (VL)
        """
        VL = 0
        potis = potis.reset_index(drop=True)

        if len(potis) == 0:
            return 0

        # Check if 'vi' column exists, add default if not
        if "vi" not in potis.columns:
            self.log(
                "POTIs DataFrame doesn't have 'vi' column. Using default vi=1.",
                level="warning",
            )
            potis["vi"] = 1.0

        for _, row in potis.iterrows():
            # Calculate the distance
            distance = haversine(
                (x_c_z, y_c_z), (row["latitude"], row["longitude"]), unit=Unit.METERS
            )

            # Avoid division by zero or negative values in log
            if distance < 1:
                distance = 1

            # Calculate the inverse weight
            inverse_weight = 1 / np.log(distance) ** 2

            # Accumulate weighted vulnerability
            VL += row["vi"] * inverse_weight

        return VL

    def _nearest_cluster(self, x_c_z, y_c_z, centroids, labels):
        """
        Find the nearest cluster to a given point based on centroid distances.

        Args:
            x_c_z: latitude of the zone center
            y_c_z: longitude of the zone center
            centroids: DataFrame of cluster centroids
            labels: Array of cluster labels

        Returns:
            int: Index of the nearest cluster
        """
        nearest_cluster = None
        nearest_distance = float("inf")
        for j in range(len(np.unique(labels))):
            # Calculate the distance between the VZ centre and the cluster centre
            distance = haversine(
                (x_c_z, y_c_z),
                (centroids.iloc[j]["latitude"], centroids.iloc[j]["longitude"]),
                unit=Unit.METERS,
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_cluster = j

        return nearest_cluster

    def _find_neighboring_zones_other_clusters(self, vulnerability_zones):
        """
        For each vulnerability zone, find neighboring zones that belong to different clusters.

        Args:
            vulnerability_zones (GeoDataFrame): GeoDataFrame with vulnerability zones

        Returns:
            GeoDataFrame: Updated GeoDataFrame with neighboring zones column
        """
        # Ensure the GeoDataFrame has a unique identifier
        if "zone_id" not in vulnerability_zones.columns:
            vulnerability_zones = vulnerability_zones.reset_index().rename(
                columns={"index": "zone_id"}
            )

        # Create spatial index for efficient spatial queries
        spatial_index = vulnerability_zones.sindex

        # Function to find neighbors in different clusters for a single zone
        def get_neighbors_other_clusters(zone, zones, spatial_index):
            potential_matches_index = list(
                spatial_index.intersection(zone.geometry.bounds)
            )
            potential_matches = zones.iloc[potential_matches_index]

            # Exclude the zone itself
            potential_matches = potential_matches[
                potential_matches["zone_id"] != zone["zone_id"]
            ]

            # Find actual neighbors (touching polygons)
            neighbors = potential_matches[
                potential_matches.geometry.touches(zone.geometry)
            ]

            # Select neighbors from different clusters
            neighbors_diff_cluster = neighbors[neighbors["cluster"] != zone["cluster"]]

            # Return list of neighboring zone_ids
            return neighbors_diff_cluster["zone_id"].tolist()

        # Apply the function to each zone
        vulnerability_zones["neighbors_other_clusters"] = vulnerability_zones.apply(
            lambda zone: get_neighbors_other_clusters(
                zone, vulnerability_zones, spatial_index
            ),
            axis=1,
        )

        return vulnerability_zones

    def _compute_relative_influence(self, vulnerability_zones):
        """
        Compute the relative influence of each zone based on its vulnerability level.

        Args:
            vulnerability_zones (GeoDataFrame): GeoDataFrame with vulnerability zones

        Returns:
            GeoDataFrame: Updated GeoDataFrame with relative influence column
        """
        # Initialize the relative influence column
        vulnerability_zones["relative_influence"] = 0.0

        # Normalize VL_normalized to ensure it's between 0 and 1
        vl_min = vulnerability_zones["VL_normalized"].min()
        vl_max = 0.0003476593149558199  # vulnerability_zones["VL_normalized"].max()
        vulnerability_zones["VL_norm"] = (
            (vulnerability_zones["VL_normalized"] - vl_min) / (vl_max - vl_min)
            if vl_max != vl_min
            else 1.0
        )

        # Function to distribute influence to neighbors
        def distribute_influence(zone, zones_dict):
            influence = zone["VL_norm"]
            neighbors = zone["neighbors_other_clusters"]
            for neighbor_id in neighbors:
                if neighbor_id in zones_dict:
                    zones_dict[neighbor_id]["relative_influence"] += influence

        # Create a dictionary for quick zone lookup
        zones_dict = vulnerability_zones.set_index("zone_id").to_dict("index")

        # Distribute influence from each zone to its neighbors
        for zone_id, zone_data in zones_dict.items():
            distribute_influence(zone_data, zones_dict)

        # Update the GeoDataFrame with the computed relative influence
        vulnerability_zones["relative_influence"] = vulnerability_zones["zone_id"].map(
            lambda zid: zones_dict[zid]["relative_influence"]
        )

        # Drop the temporary normalized VL column
        vulnerability_zones.drop(columns=["VL_norm"], inplace=True)

        return vulnerability_zones

    def calculate_vulnerability(self, x_c_z, y_c_z, potis):
        """
        Calculate vulnerability using the selected distance method.

        Args:
            x_c_z: latitude of the zone center
            y_c_z: longitude of the zone center
            potis: DataFrame with points of interest

        Returns:
            float: Calculated vulnerability level
        """
        # Get the appropriate method based on the selected distance_method
        method_name = self.DISTANCE_METHODS[self.distance_method]
        method = getattr(self, method_name)

        # Call the method with the provided parameters
        return method(x_c_z, y_c_z, potis)

    def calculate_vulnerability_zones(self):
        """
        Calculate vulnerability for each zone based on the nearest cluster's POTIs.
        Assumes the 'vi' values are already present in the POTI DataFrame.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If required data has not been loaded.
        """
        if self.poti_df is None or self.vulnerability_zones is None:
            self.log(
                "Data must be loaded before calculating vulnerability zones",
                level="error",
            )
            raise ValueError(
                "Data must be loaded before calculating vulnerability zones."
            )

        zones = self.vulnerability_zones.copy()
        self.log("Assigning zones to nearest clusters...", level="info")

        zones["cluster"] = zones.apply(
            lambda x: self._nearest_cluster(
                x["geometry"].centroid.y,
                x["geometry"].centroid.x,
                self.cluster_centers,
                self.poti_df["cluster"].values,
            ),
            axis=1,
        )

        self.log(
            f"Calculating vulnerability using {self.distance_method} method...",
            level="info",
        )
        zones["value"] = zones.apply(
            lambda x: self.calculate_vulnerability(
                x["geometry"].centroid.y,
                x["geometry"].centroid.x,
                self.poti_df[self.poti_df["cluster"] == x["cluster"]],
            ),
            axis=1,
        )

        min_vl = zones["value"].min()
        # Use the new config helper method
        max_vl = self._get_config("max_vulnerability", self.place_name)
        if max_vl is not None:
            self.log(f"Using configured max vulnerability: {max_vl}", level="info")
        else:
            max_vl = zones["value"].max()
            self.log(f"Using calculated max vulnerability: {max_vl}", level="info")

        self.log(f"Min VL: {min_vl}, Max VL: {max_vl}", level="info")
        zones["VL_normalized"] = zones["value"].apply(
            lambda x: (x - min_vl) / (max_vl - min_vl) if max_vl > min_vl else 0.5
        )

        self.vulnerability_zones = zones
        self.results["vulnerability_zones"] = zones

        self.log(f"Vulnerability calculated for {len(zones)} zones", level="success")
        self.log("Vulnerability statistics:", level="info")
        self.log(str(zones["VL_normalized"].describe()), level="info")
        return self

    def smooth_vulnerability(self, influence_threshold=0.3):
        """
        Apply smoothing across cluster boundaries to create more realistic transitions.

        Args:
            influence_threshold (float): Threshold to determine significant influence

        Returns:
            self: For method chaining

        Raises:
            ValueError: If vulnerability zones have not been calculated
        """
        if "VL_normalized" not in self.vulnerability_zones.columns:
            self.log(
                "Vulnerability zones must be calculated before smoothing", level="error"
            )
            raise ValueError("Vulnerability zones must be calculated before smoothing.")

        self.log(
            f"Smoothing vulnerability levels with influence threshold {influence_threshold}...",
            level="info",
        )

        # Find neighboring zones in different clusters
        zones = self._find_neighboring_zones_other_clusters(self.vulnerability_zones)

        # Compute relative influence between zones
        zones = self._compute_relative_influence(zones)

        # Create a copy to avoid modifying the original GeoDataFrame
        zones["VL_normalized_smoothed"] = zones["VL_normalized"]

        # Dictionary for quick lookup of zone data by zone_id
        zones_dict = zones.set_index("zone_id").to_dict("index")

        # Function to determine smoothed VL_normalized for a single zone
        def compute_smoothed_vl(zone_id, zone_data, zones_dict, threshold):
            neighbors = zone_data["neighbors_other_clusters"]
            if not neighbors:
                return zone_data["VL_normalized"]  # No neighbors from other clusters

            # Dictionary to accumulate influence from each neighboring cluster
            cluster_influence = {}

            for neighbor_id in neighbors:
                neighbor = zones_dict.get(neighbor_id)
                if neighbor:
                    neighbor_cluster = neighbor["cluster"]
                    neighbor_vl = neighbor["VL_normalized"]
                    if neighbor_cluster not in cluster_influence:
                        cluster_influence[neighbor_cluster] = 0.0
                    cluster_influence[neighbor_cluster] += neighbor_vl

            if not cluster_influence:
                return zone_data["VL_normalized"]  # No valid neighbors found

            # Identify the most influential neighboring cluster
            most_influential_cluster = max(cluster_influence, key=cluster_influence.get)
            total_influence = cluster_influence[most_influential_cluster]

            # Calculate relative influence
            total_cluster_influence = sum(cluster_influence.values())
            if total_cluster_influence == 0:
                return zone_data["VL_normalized"]  # Avoid division by zero
            relative_influence = total_influence / total_cluster_influence

            # Only apply smoothing if the relative influence exceeds the threshold
            if relative_influence >= threshold:
                # Get the average VL_normalized of the most influential cluster
                influencing_zones = zones[zones["cluster"] == most_influential_cluster]
                if len(influencing_zones) == 0:
                    return zone_data["VL_normalized"]  # Avoid division by zero
                average_influencing_vl = influencing_zones["VL_normalized"].mean()

                # Define smoothing factor (alpha) between 0 and 1
                alpha = 0.6  # Adjust this value to control smoothing intensity

                # Compute the smoothed VL_normalized
                smoothed_vl = (
                    alpha * zone_data["VL_normalized"]
                    + (1 - alpha) * average_influencing_vl
                )

                # Ensure the smoothed value stays within [0,1]
                smoothed_vl = min(max(smoothed_vl, 0), 1)

                return smoothed_vl
            else:
                # Influence from neighbors is not significant enough to adjust
                return zone_data["VL_normalized"]

        # Apply the smoothing function to each zone
        for zone_id, zone_data in zones_dict.items():
            smoothed_vl = compute_smoothed_vl(
                zone_id, zone_data, zones_dict, influence_threshold
            )
            zones.loc[
                zones[zones["zone_id"] == zone_id].index, "VL_normalized_smoothed"
            ] = smoothed_vl

        # Create color scale for visualization
        color_scale = linear.YlOrRd_09.scale(0, 1)
        color_scale.caption = "Vulnerability Level"
        zones["color"] = zones["VL_normalized_smoothed"].apply(color_scale)

        # Update the vulnerability zones
        self.vulnerability_zones = zones
        self.results["smoothed_vulnerability_zones"] = zones

        self.log("Vulnerability smoothing complete", level="success")
        self.log("Smoothed vulnerability statistics:", level="info")
        self.log(str(zones["VL_normalized_smoothed"].describe()), level="info")

        return self

    def visualize(self, output_file=None):
        """
        Create visualization of vulnerability zones.

        Args:
            output_file (str, optional): File path to save the visualization

        Returns:
            folium.Map: Interactive map with vulnerability zones

        Raises:
            ValueError: If vulnerability has not been calculated
        """
        # Use the existing _create_interactive_map method
        return self._create_interactive_map(output_file)

    def save(self, output_dir=None):
        """
        Export results to files.

        Args:
            output_dir (str, optional): Directory to save output files.

        Returns:
            self: For method chaining
        """
        if output_dir is None:
            output_dir = f"./results/{self.place_name}/"

        os.makedirs(output_dir, exist_ok=True)

        # Save vulnerability zones
        if self.vulnerability_zones is not None:
            vz_path = os.path.join(
                output_dir, f"{self.place_name}_vulnerability_zones.geojson"
            )
            self.vulnerability_zones.to_file(vz_path, driver="GeoJSON")
            self.log(f"Vulnerability zones saved to {vz_path}", level="success")

            # Also save as CSV for easier analysis
            csv_path = os.path.join(
                output_dir, f"{self.place_name}_vulnerability_zones.csv"
            )
            # Save only non-geometry columns to CSV
            non_geom_cols = [
                col for col in self.vulnerability_zones.columns if col != "geometry"
            ]
            self.vulnerability_zones[non_geom_cols].to_csv(csv_path, index=False)
            self.log(f"Vulnerability data saved to {csv_path}", level="success")

        # Save POTI data with cluster assignments
        if self.poti_df is not None:
            poti_path = os.path.join(
                output_dir, f"{self.place_name}_poti_clustered.csv"
            )
            self.poti_df.to_csv(poti_path, index=False)
            self.log(f"Clustered POTIs saved to {poti_path}", level="success")

        # Save cluster centers
        if self.cluster_centers is not None:
            centers_path = os.path.join(
                output_dir, f"{self.place_name}_cluster_centers.csv"
            )
            self.cluster_centers.to_csv(centers_path, index=False)
            self.log(f"Cluster centers saved to {centers_path}", level="success")

        return self

    def _run_clustering_pipeline(self, df, evaluation_time):
        """
        Run a clustering pipeline that first obtains clustering centers via OPTICS
        and then initializes KMeans with those centers.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with POTI data (including 'vi').
        evaluation_time : str or int
            Evaluation time identifier (used for naming but not passed to clustering methods).

        Returns
        -------
        dict
            Dictionary of clustering results.
        """
        from verus.clustering.kmeans import KMeansHaversine
        from verus.clustering.optics import GeOPTICS

        # Run OPTICS to obtain centers
        self.log("Running OPTICS clustering to obtain centres...", level="info")
        optics_params = self._get_config(
            "clustering",
            "optics",
            default={"min_samples": 5, "xi": 0.05, "min_cluster_size": 5},
        )
        optics = GeOPTICS(
            min_samples=optics_params.get("min_samples", 5),
            xi=optics_params.get("xi", 0.05),
            min_cluster_size=optics_params.get("min_cluster_size", 5),
            verbose=self.verbose,
        )
        optics_results = optics.run(data_source=df)

        # Check if OPTICS produced enough centroids
        if (
            optics_results["centroids"] is not None
            and len(optics_results["centroids"]) > 1
        ):
            centers = optics_results["centroids"]
            self.log(f"Running KMeans with {len(centers)} OPTICS centres", level="info")
            kmeans_params = self._get_config(
                "clustering",
                "kmeans",
                default={"n_clusters": 8, "init": "predefined", "random_state": 42},
            )
            kmeans = KMeansHaversine(
                n_clusters=kmeans_params.get("n_clusters", 8),
                init=kmeans_params.get("init", "predefined"),
                random_state=kmeans_params.get("random_state", 42),
                verbose=self.verbose,
                predefined_centers=centers,
            )
            # Only pass what's needed
            kmeans_results = kmeans.run(
                data_source=df,
                centers_input=centers,
            )
            # Add suffix information after the fact
            kmeans_results["algorithm_suffix"] = f"KM-OPTICS_{evaluation_time}"
        else:
            # Fallback on default KMeans
            self.log(
                "OPTICS returned insufficient centres → defaulting to standard KMeans",
                level="warning",
            )
            kmeans = KMeansHaversine(
                n_clusters=8, init="k-means++", verbose=self.verbose, random_state=42
            )
            # Remove all problematic parameters
            kmeans_results = kmeans.run(
                data_source=df,
            )
            # Add suffix information after the fact
            kmeans_results["algorithm_suffix"] = f"KM-Standard_{evaluation_time}"

        # Add evaluation_time to results directly
        kmeans_results["evaluation_time"] = evaluation_time
        return kmeans_results

    def _extract_place_name(self, data_source):
        """
        Extract place name from data source or return the configured place name.

        Args:
            data_source: Source of data (DataFrame or path)

        Returns:
            str: Extracted place name
        """
        # If data_source is a DataFrame, just return the configured place_name
        if isinstance(data_source, (pd.DataFrame, gpd.GeoDataFrame)):
            return self.place_name

        # If it's a string (file path), try to extract place name from it
        elif isinstance(data_source, str):
            try:
                basename = os.path.basename(data_source)
                # Try to extract place name from file name (before first underscore)
                place = basename.split("_")[0]
                if place:
                    return place
            except (IndexError, AttributeError):
                pass

        # Default to configured place_name
        return self.place_name

    def _create_interactive_map(self, output_file=None):
        """
        Create an interactive folium map of vulnerability zones.

        Args:
            output_file (str, optional): File path to save the map. If None, doesn't save.

        Returns:
            folium.Map: Interactive map object
        """
        import folium
        from branca.colormap import linear

        # Get map center (use config if available, otherwise calculate from POTIs)
        map_center_config = self._get_config("map_center", self.place_name)
        if map_center_config:
            map_center = [map_center_config["lat"], map_center_config["lon"]]
        else:
            coords = self.poti_df[["latitude", "longitude"]].to_numpy()
            map_center = [coords[:, 0].mean(), coords[:, 1].mean()]

        # Create map
        m = folium.Map(location=map_center, zoom_start=13, tiles="cartodbpositron")

        # Add vulnerability zones
        folium.GeoJson(
            data=self.vulnerability_zones,
            style_function=lambda feature: {
                "fillColor": feature["properties"]["color"],
                "color": feature["properties"]["color"],
                "weight": 0.1,
                "fillOpacity": 0.7,
            },
            popup=folium.GeoJsonPopup(
                fields=["VL_normalized_smoothed", "cluster"],
                aliases=["VL:", "Cluster:"],
                localize=True,
            ),
        ).add_to(m)

        # Add boundary if available
        boundary_path = self._get_config("boundary_paths", self.place_name)
        if not boundary_path:
            # Try standard path
            potential_path = f"./geojson/{self.place_name}_boundaries.geojson"
            if os.path.exists(potential_path):
                boundary_path = potential_path

        if boundary_path and os.path.exists(boundary_path):
            folium.GeoJson(
                boundary_path,
                name="boundary",
                style_function=lambda feature: {
                    "color": "#B2BEB5",
                    "weight": 2,
                    "fillOpacity": 0,
                },
            ).add_to(m)

        # Add color scale
        color_scale = linear.YlOrRd_09.scale(0, 1)
        color_scale.caption = "Vulnerability Level"
        color_scale.caption_font_size = "14pt"
        color_scale.add_to(m)

        # Save map if output file is specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            m.save(output_file)

        return m

    def run(
        self,
        data_source=None,
        evaluation_time=None,
        centers_input=None,
        area_boundary_path=None,
    ):
        """
        Run the complete vulnerability assessment workflow.

        This method executes the full vulnerability assessment pipeline:
        1. Updates POTI vulnerability indices based on external time window data (if provided)
        2. Executes the clustering pipeline (OPTICS → KMeans)
        3. Computes vulnerability zones
        4. Applies smoothing to improve spatial continuity
        5. Creates map visualization (if area_boundary_path is provided)

        Args:
            data_source (pd.DataFrame or str, optional): POTI DataFrame or path to POTI data.
                If None, uses data previously loaded via load().
            evaluation_time (str, optional): Time scenario to evaluate.
                Used to filter time windows and in result naming.
            centers_input (pd.DataFrame, optional): Predefined centers for clustering.
                If None, centers are determined by the clustering pipeline.
            area_boundary_path (str, optional): Path to boundary GeoJSON for visualization.

        Returns:
            dict: Dictionary with results of the vulnerability assessment, containing:
                - "clusters": DataFrame with cluster assignments
                - "centroids": DataFrame with cluster centers
                - "map": Interactive map (if area_boundary_path was provided)
                - "input_data": POTI DataFrame with updated vulnerability indices
                - "place_name": Name of the analyzed place
                - "labels", "inertia", "n_iter": Additional clustering information
                - "vulnerability_zones": GeoDataFrame with vulnerability zones
                - "error": Error message if an exception occurred
                - "evaluation_time": The evaluation time used
        """
        try:
            # Extract place name from data source if possible
            place_name = self._extract_place_name(data_source)

            # Use the already loaded data or load it if a string path is provided
            if isinstance(data_source, str) and self.poti_df is None:
                # If data_source is a file path and data isn't loaded yet, load it
                self.log(f"Loading data from {data_source}", level="info")
                # This would require implementing a load_from_file method, which isn't shown here
                # For now, just use pandas/geopandas directly
                if data_source.endswith(".csv"):
                    df = pd.read_csv(data_source)
                elif data_source.endswith((".geojson", ".shp")):
                    df = gpd.read_file(data_source)
                else:
                    raise ValueError(f"Unsupported file format for {data_source}")
            elif isinstance(data_source, (pd.DataFrame, gpd.GeoDataFrame)):
                # If data_source is already a DataFrame, use it directly
                df = data_source.copy()
            else:
                # Use the already loaded data
                df = self.poti_df.copy() if self.poti_df is not None else None

            if df is None:
                raise ValueError(
                    "No data source provided and no data previously loaded"
                )

            # Ensure 'vi' column exists with default value 1.0 if missing
            if "vi" not in df.columns:
                self.log("Adding default vulnerability index (vi=1.0)", level="info")
                df["vi"] = 1.0

            # --- Apply time windows to update vulnerability indices ---
            if self.time_windows is not None:
                self.log(
                    f"Applying time windows for evaluation time: {evaluation_time}",
                    level="info",
                )
                df = self._apply_time_windows_to_potis(evaluation_time)

            # --- Execute the clustering pipeline (OPTICS -> KMeans) ---
            clusters_results = self._run_clustering_pipeline(
                df, evaluation_time or "default"
            )

            # Check cluster distribution
            # Add debug code to check cluster distribution
            if "clusters" in clusters_results:
                cluster_counts = clusters_results["clusters"]["cluster"].value_counts()
                self.log(f"Cluster distribution: {dict(cluster_counts)}", level="info")
                if len(cluster_counts) <= 1:
                    self.log(
                        "WARNING: All POIs assigned to a single cluster!",
                        level="warning",
                    )

            # Update internal state with clustering results
            self.poti_df = clusters_results.get("clusters", df)
            # Fix: Use "centroids" key instead of "clusters"
            self.cluster_centers = clusters_results.get("centroids", None)

            # Proceed to calculate vulnerability zones based on updated poti_df
            self.calculate_vulnerability_zones()
            self.smooth_vulnerability()

            map_obj = None
            if area_boundary_path and self.poti_df is not None:
                # Use the _create_interactive_map method directly
                map_obj = self._create_interactive_map(area_boundary_path)

            return {
                "clusters": clusters_results.get("clusters"),
                "centroids": clusters_results.get("centroids"),
                "map": map_obj,
                "input_data": self.poti_df,
                "place_name": place_name,
                "labels": clusters_results.get("labels"),
                "inertia": clusters_results.get("inertia"),
                "n_iter": clusters_results.get("n_iter"),
                "vulnerability_zones": self.vulnerability_zones,
                "evaluation_time": evaluation_time or "default",
            }

        except Exception as e:
            import traceback

            self.log(f"Error in vulnerability assessment workflow: {e}", "error")
            self.log(traceback.format_exc(), "error")  # Print stack trace for debugging
            return {
                "clusters": None,
                "centroids": None,
                "map": None,
                "input_data": None,
                "place_name": None,
                "labels": None,
                "inertia": None,
                "n_iter": None,
                "vulnerability_zones": None,
                "error": str(e),  # Include error message in results
                "evaluation_time": evaluation_time or "default",
            }

    # def _create_interactive_map(self, output_file=None):
    #     """
    #     Create an interactive folium map of vulnerability zones.

    #     Args:
    #         output_file (str, optional): File path to save the map. If None, doesn't save.

    #     Returns:
    #         folium.Map: Interactive map object
    #     """

    #     import folium
    #     from branca.colormap import linear

    #     # Get map center (use config if available, otherwise calculate from POTIs)
    #     map_center_config = self._get_config("map_center", self.place_name)
    #     if map_center_config:
    #         map_center = [map_center_config["lat"], map_center_config["lon"]]
    #     else:
    #         coords = self.poti_df[["latitude", "longitude"]].to_numpy()
    #         map_center = [coords[:, 0].mean(), coords[:, 1].mean()]

    #     # Create map
    #     m = folium.Map(location=map_center, zoom_start=13, tiles="cartodbpositron")

    #     # Add vulnerability zones
    #     folium.GeoJson(
    #         data=self.vulnerability_zones,
    #         style_function=lambda feature: {
    #             "fillColor": feature["properties"]["color"],
    #             "color": feature["properties"]["color"],
    #             "weight": 0.1,
    #             "fillOpacity": 0.7,
    #         },
    #         popup=folium.GeoJsonPopup(
    #             fields=["VL_normalized_smoothed", "cluster"],
    #             aliases=["VL:", "Cluster:"],
    #             localize=True,
    #         ),
    #     ).add_to(m)

    #     # Add boundary if available
    #     boundary_path = self._get_config("boundary_paths", self.place_name)
    #     if not boundary_path:
    #         # Try standard path
    #         potential_path = f"./geojson/{self.place_name}_boundaries.geojson"
    #         if os.path.exists(potential_path):
    #             boundary_path = potential_path

    #     if boundary_path and os.path.exists(boundary_path):
    #         folium.GeoJson(
    #             boundary_path,
    #             name="boundary",
    #             style_function=lambda feature: {
    #                 "color": "#B2BEB5",
    #                 "weight": 2,
    #                 "fillOpacity": 0,
    #             },
    #         ).add_to(m)

    #     # Add color scale
    #     color_scale = linear.YlOrRd_09.scale(0, 1)
    #     color_scale.caption = "Vulnerability Level"
    #     color_scale.caption_font_size = "14pt"
    #     color_scale.add_to(m)

    #     # Save map if output file is specified
    #     if output_file:
    #         os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #         m.save(output_file)

    #     return m
