import os

import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import linear
from haversine import Unit, haversine
from sklearn.neighbors import BallTree

from verus.utils.logger import Logger


class VERUS(Logger):
    """
    Main class for assessing urban vulnerability using POI clustering and spatial analysis.

    This class handles the entire vulnerability assessment workflow including:
    - Loading POI and cluster data
    - Processing time windows
    - Computing vulnerability levels using various distance methods
    - Visualizing and exporting results

    Attributes:
        place_name (str): Name of the place to analyze
        method (str): Clustering method used (e.g., "KM-OPTICS", "AP")
        evaluation_time (str): Time scenario to evaluate
        distance_method (str): Method for calculating vulnerability based on distance
            Options: "gaussian", "adaptive", "inverse_weighted"
        sigma (float): Bandwidth parameter for Gaussian methods (in meters)
        adaptive_radius (float): Radius for adaptive density calculation (in meters)
        config (dict): Configuration parameters
    """

    # Available distance calculation methods
    DISTANCE_METHODS = {
        "gaussian": "_gaussian_weighted_vulnerability",
        "adaptive": "_gaussian_weighted_vulnerability_adaptive_density",
        "inverse_weighted": "_inversely_weighted_distance",
    }

    def __init__(
        self,
        place_name,
        method="KM-OPTICS",
        evaluation_time="ET4",
        distance_method="gaussian",
        sigma=1000,
        adaptive_radius=1500,
        config=None,
    ):
        """
        Initialize the VulnerabilityAssessor with the given parameters.

        Args:
            place_name (str): Name of the place to analyze
            method (str, optional): Clustering method. Defaults to "KM-OPTICS".
            evaluation_time (str, optional): Time scenario. Defaults to "ET4".
            distance_method (str, optional): Distance calculation method. Defaults to "gaussian".
            sigma (float, optional): Bandwidth for Gaussian methods. Defaults to 1000.
            adaptive_radius (float, optional): Radius for adaptive density. Defaults to 1500.
            config (dict, optional): Configuration parameters. Defaults to None.

        Raises:
            ValueError: If an invalid distance_method is provided.
        """
        super().__init__()

        # Validate distance method
        if distance_method not in self.DISTANCE_METHODS:
            valid_methods = ", ".join(self.DISTANCE_METHODS.keys())
            raise ValueError(
                f"Invalid distance method: {distance_method}. Choose from: {valid_methods}"
            )

        # Initialize instance variables
        self.place_name = place_name
        self.method = method
        self.evaluation_time = evaluation_time
        self.distance_method = distance_method
        self.sigma = sigma
        self.adaptive_radius = adaptive_radius
        self.config = config

        # Data containers to be populated later
        self.poti_df = None
        self.cluster_centers = None
        self.vulnerability_zones = None
        self.time_windows = None
        self.epoch_time = None
        self.filtered_time_windows = None

        # Results
        self.results = {}

        self.log(
            f"VulnerabilityAssessor initialized for {place_name} using {distance_method} method"
        )

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

    def _gaussian_weighted_vulnerability_adaptive_density(self, x_c_z, y_c_z, potis):
        """
        Calculate the vulnerability level (VL) using a Gaussian approach with adaptive bandwidth
        based on local point density.

        Parameters:
            x_c_z: latitude of the zone center
            y_c_z: longitude of the zone center
            potis: DataFrame with points of interest (latitude, longitude, vi)

        Returns:
            Vulnerability level (VL) of the zone
        """
        # Reset index to ensure indices align with sigmas
        potis = potis.reset_index(drop=True)

        # Prepare PoTI coordinates
        poti_coords = potis[["latitude", "longitude"]].to_numpy()
        n = len(poti_coords)

        if n == 0:
            return 0

        # Check if 'vi' column exists, add default if not
        if "vi" not in potis.columns:
            self.log(
                "POTIs DataFrame doesn't have 'vi' column. Using default vi=1.",
                level="warning",
            )
            potis["vi"] = 1.0

        # Earth's radius in meters
        earth_radius = 6371000

        # Convert degrees to radians for haversine metric
        poti_coords_rad = np.radians(poti_coords)

        # Build a BallTree for efficient neighbor queries
        tree = BallTree(poti_coords_rad, metric="haversine")

        # Compute local point density for each PoI
        densities = []
        for coord in poti_coords_rad:
            # Find neighbors within the given radius (convert radius to radians)
            indices = tree.query_radius([coord], r=self.adaptive_radius / earth_radius)
            local_density = len(indices[0]) - 1  # Subtract the point itself
            densities.append(local_density)

        densities = np.array(densities)

        # Prevent division by zero
        epsilon = 1e-6

        # Calculate adaptive sigma as inversely proportional to local density
        sigmas = 1 / (densities + epsilon)

        # Normalize sigmas to have a reasonable scale
        sigmas = sigmas / sigmas.max() * self.adaptive_radius

        # Compute vulnerability level
        gaussian_kernel_sum = 0

        for idx, row in potis.iterrows():
            sigma = sigmas[idx]
            sigma = max(sigma, 1e-6)  # Ensure sigma is positive

            # Calculate the distance in meters to the zone center
            distance = haversine(
                (x_c_z, y_c_z), (row["latitude"], row["longitude"]), unit=Unit.METERS
            )

            # Calculate the Gaussian influence
            influence = row["vi"] * np.exp(-0.5 * (distance / sigma) ** 2)
            gaussian_kernel_sum += influence

        # Average sigma for normalization
        avg_sigma = sigmas.mean() if len(sigmas) > 0 else self.adaptive_radius

        # Normalize to obtain the vulnerability level
        VL = gaussian_kernel_sum / (n * avg_sigma * np.sqrt(2 * np.pi))

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
        vl_max = vulnerability_zones["VL_normalized"].max()
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

    def load(
        self,
        poti_file,
        centroids_file,
        zones_file,
        time_windows_dir,
    ):
        """
        Load POI and cluster data from explicitly provided files.

        Args:
            poti_file (str): Full path to the POI cluster file
            centroids_file (str): Full path to the centroids file
            zones_file (str): Full path to the vulnerability zones geojson file
            time_windows_dir (str): Directory containing time window files

        Returns:
            self: For method chaining

        Raises:
            FileNotFoundError: If required files are not found
            ValueError: If files don't have the required columns
        """
        # Load POI cluster data
        if not os.path.exists(poti_file):
            self.log(f"POI cluster file not found: {poti_file}", level="error")
            raise FileNotFoundError(f"POI cluster file not found: {poti_file}")

        if not os.path.exists(centroids_file):
            self.log(
                f"Cluster centroids file not found: {centroids_file}", level="error"
            )
            raise FileNotFoundError(
                f"Cluster centroids file not found: {centroids_file}"
            )

        # Load POI data
        self.poti_df = pd.read_csv(poti_file, index_col=False)

        # Check for required columns and handle missing columns
        required_poti_columns = ["latitude", "longitude", "category"]
        missing_columns = [
            col for col in required_poti_columns if col not in self.poti_df.columns
        ]
        if missing_columns:
            self.log(
                f"POI file missing required columns: {', '.join(missing_columns)}",
                level="error",
            )
            raise ValueError(
                f"POI file missing required columns: {', '.join(missing_columns)}"
            )

        # Check if cluster column exists, add default if not
        if "cluster" not in self.poti_df.columns:
            self.log(
                "POI file doesn't have 'cluster' column. Adding default cluster 0.",
                level="warning",
            )
            self.poti_df["cluster"] = 0

        # Load centroids data
        self.cluster_centers = pd.read_csv(centroids_file, index_col=False)
        required_centroid_columns = ["latitude", "longitude"]
        missing_columns = [
            col
            for col in required_centroid_columns
            if col not in self.cluster_centers.columns
        ]
        if missing_columns:
            self.log(
                f"Centroids file missing required columns: {', '.join(missing_columns)}",
                level="error",
            )
            raise ValueError(
                f"Centroids file missing required columns: {', '.join(missing_columns)}"
            )

        # Load time windows
        if not os.path.exists(time_windows_dir):
            self.log(
                f"Time windows directory not found: {time_windows_dir}", level="error"
            )
            raise FileNotFoundError(
                f"Time windows directory not found: {time_windows_dir}"
            )

        time_windows_df = []
        for file in os.listdir(time_windows_dir):
            if file.endswith(".csv"):
                try:
                    time_window = pd.read_csv(os.path.join(time_windows_dir, file))
                    # Validate time window columns
                    required_tw_columns = ["ts", "te", "vi"]
                    missing_columns = [
                        col
                        for col in required_tw_columns
                        if col not in time_window.columns
                    ]
                    if missing_columns:
                        self.log(
                            f"Time window file {file} missing required columns: {', '.join(missing_columns)}. Skipping.",
                            level="warning",
                        )
                        continue

                    time_window["category"] = os.path.splitext(file)[0]
                    time_windows_df.append(time_window)
                except Exception as e:
                    self.log(
                        f"Error loading time window file {file}: {str(e)}",
                        level="warning",
                    )
                    continue

        if not time_windows_df:
            self.log(
                f"No valid time window files found in: {time_windows_dir}",
                level="error",
            )
            raise FileNotFoundError(
                f"No valid time window files found in: {time_windows_dir}"
            )

        self.time_windows = pd.concat(time_windows_df, ignore_index=True)

        # Load vulnerability zones geometry
        if not os.path.exists(zones_file):
            self.log(f"Vulnerability zones file not found: {zones_file}", level="error")
            raise FileNotFoundError(f"Vulnerability zones file not found: {zones_file}")

        try:
            self.vulnerability_zones = gpd.read_file(zones_file)
            # Check if the GeoDataFrame has valid geometry
            if not all(self.vulnerability_zones.geometry.is_valid):
                self.log(
                    "Vulnerability zones file contains invalid geometries",
                    level="error",
                )
                raise ValueError("Vulnerability zones file contains invalid geometries")
        except Exception as e:
            self.log(f"Error loading vulnerability zones file: {str(e)}", level="error")
            raise ValueError(f"Error loading vulnerability zones file: {str(e)}")

        # Get unique clusters (safely handling the case where it might be a new column)
        n_clusters = 1
        if "cluster" in self.poti_df.columns:
            n_clusters = len(np.unique(self.poti_df["cluster"]))

        # Log summary info
        self.log(
            f"Loaded {len(self.poti_df)} POTIs across {n_clusters} clusters",
            level="success",
        )
        self.log(
            f"Loaded {len(self.vulnerability_zones)} vulnerability zones", level="info"
        )
        self.log(f"Loaded {len(self.time_windows)} time window entries", level="info")

        return self

    def calculate_vulnerability_zones(self):
        """
        Calculate vulnerability for each zone based on the nearest cluster's POIs.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If required data has not been loaded
        """
        if (
            self.poti_df is None
            or self.cluster_centers is None
            or self.vulnerability_zones is None
        ):
            self.log(
                "Data must be loaded before calculating vulnerability zones",
                level="error",
            )
            raise ValueError(
                "Data must be loaded before calculating vulnerability zones."
            )

        # Make a copy of vulnerability zones to avoid modifying the original
        zones = self.vulnerability_zones.copy()

        self.log("Assigning zones to nearest clusters...", level="info")

        # Find the nearest cluster for each vulnerability zone
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

        # Compute the vulnerability level for each zone using the selected distance method
        zones["value"] = zones.apply(
            lambda x: self.calculate_vulnerability(
                x["geometry"].centroid.y,
                x["geometry"].centroid.x,
                self.poti_df[self.poti_df["cluster"] == x["cluster"]],
            ),
            axis=1,
        )

        # Normalize the vulnerability levels
        min_vl = zones["value"].min()

        # Use max vulnerability from config if available, otherwise use the maximum value
        if (
            self.config
            and hasattr(self.config, "max_vulnerability")
            and self.place_name in self.config.max_vulnerability
        ):
            max_vl = self.config.max_vulnerability[self.place_name]
            self.log(f"Using configured max vulnerability: {max_vl}", level="info")
        else:
            max_vl = zones["value"].max()
            self.log(f"Using calculated max vulnerability: {max_vl}", level="info")

        self.log(f"Min VL: {min_vl}, Max VL: {max_vl}", level="info")

        # Normalize values between 0 and 1
        zones["VL_normalized"] = zones["value"].apply(
            lambda x: (x - min_vl) / (max_vl - min_vl) if max_vl > min_vl else 0.5
        )

        # Update the vulnerability zones
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
        # To be implemented
        pass

    def export_results(self, output_dir=None):
        """
        Export results to files.

        Args:
            output_dir (str, optional): Directory to save output files.

        Returns:
            self: For method chaining
        """
        # To be implemented
        pass

    def run(
        self,
        evaluation_time=None,
        save_output=True,
        output_dir=None,
        run_clustering=True,
        optics_params=None,
        kmeans_params=None,
    ):
        """
        Run the vulnerability assessment workflow.

        Args:
            evaluation_time (str, optional): Time scenario to evaluate. If None, uses the one set during initialization.
            save_output (bool, optional): Whether to save results to files. Defaults to True.
            output_dir (str, optional): Directory to save output files. If None, uses "./results/{place_name}/".
            run_clustering (bool, optional): Whether to run OPTICS+KMeans clustering. Defaults to True.
            optics_params (dict, optional): Parameters for OPTICS clustering algorithm.
            kmeans_params (dict, optional): Parameters for K-means clustering algorithm.

        Returns:
            dict: Results dictionary containing:
                - vulnerability_zones: GeoDataFrame with vulnerability assessment
                - potis: DataFrame with points of interest
                - clusters: DataFrame with cluster centers
                - epoch_time: Epoch time used for evaluation

        Raises:
            ValueError: If required data has not been loaded
        """
        # Check if data is loaded
        if (
            self.poti_df is None
            or self.time_windows is None
            or self.vulnerability_zones is None
        ):
            self.log(
                "Data must be loaded before running vulnerability assessment",
                level="error",
            )
            raise ValueError(
                "Data must be loaded before running vulnerability assessment."
            )

        # Update evaluation time if provided
        if evaluation_time is not None:
            self.log(f"Updating evaluation time to {evaluation_time}", level="info")
            self.evaluation_time = evaluation_time

        # Set default output directory if not provided
        if output_dir is None:
            output_dir = f"./results/{self.place_name}/"

        os.makedirs(output_dir, exist_ok=True)

        # Run clustering if requested
        if run_clustering:
            self.log("Running clustering pipeline (OPTICS + KMeans)", level="info")

            # Import clustering modules
            from verus.clustering import GeOPTICS, KMeansHaversine

            # Default OPTICS parameters if not provided
            if optics_params is None:
                optics_params = {"min_samples": 5, "xi": 0.05, "verbose": True}

            # Run OPTICS
            self.log("Running OPTICS clustering...", level="info")
            optics = GeOPTICS(**optics_params, output_dir=output_dir)
            optics_results = optics.run(
                data_source=self.poti_df,
                place_name=self.place_name,
                time_windows_path=None,  # We already have time windows loaded
                evaluation_time=self.evaluation_time,
                save_output=False,
            )

            # Prepare KMeans parameters
            if kmeans_params is None:
                kmeans_params = {}

            # Check if OPTICS was successful
            if (
                optics_results["centroids"] is None
                or len(optics_results["centroids"]) < 2
            ):
                self.log(
                    "OPTICS clustering failed or returned too few clusters. Using default KMeans.",
                    level="warning",
                )

                # Set default number of clusters if not specified
                if "n_clusters" not in kmeans_params:
                    kmeans_params["n_clusters"] = 8

                # Set default initialization method if not specified
                if "init" not in kmeans_params:
                    kmeans_params["init"] = "k-means++"

                # Run standard KMeans without predefined centers
                kmeans = KMeansHaversine(**kmeans_params, output_dir=output_dir)
                kmeans_results = kmeans.run(
                    data_source=self.poti_df,
                    place_name=self.place_name,
                    evaluation_time=self.evaluation_time,
                    save_output=False,
                    algorithm_suffix="KM-Standard",
                )
            else:
                # Run KMeans with OPTICS centers
                self.log(
                    f"Running KMeans with {len(optics_results['centroids'])} OPTICS centers...",
                    level="info",
                )

                # Set n_clusters to match OPTICS centroids count if not explicitly specified
                if "n_clusters" not in kmeans_params:
                    kmeans_params["n_clusters"] = len(optics_results["centroids"])

                kmeans = KMeansHaversine(**kmeans_params, output_dir=output_dir)
                kmeans_results = kmeans.run(
                    data_source=self.poti_df,
                    place_name=self.place_name,
                    evaluation_time=self.evaluation_time,
                    centers_input=optics_results[
                        "centroids"
                    ],  # Always use OPTICS centroids when available
                    save_output=False,
                    algorithm_suffix="KM-OPTICS",
                )

            # Update POTIs and centroids from KMeans results
            self.log("Updating POI data with clustering results", level="info")
            self.poti_df = kmeans_results["poti_df"]
            self.cluster_centers = kmeans_results["clusters"]

        # Calculate vulnerability zones
        self.log(
            f"Calculating vulnerability using {self.distance_method} method",
            level="info",
        )
        self.calculate_vulnerability_zones()

        # Apply smoothing
        self.log("Applying smoothing across cluster boundaries", level="info")
        self.smooth_vulnerability()

        # Save results if requested
        if save_output:
            # Save vulnerability zones
            output_geojson = os.path.join(
                output_dir,
                f"{self.place_name}_vulnerability_zones_{self.distance_method}_{self.evaluation_time}.geojson",
            )
            self.vulnerability_zones.to_file(output_geojson, driver="GeoJSON")
            self.log(f"Saved vulnerability zones to {output_geojson}", level="success")

            # Create and save visualization
            output_html = os.path.join(
                output_dir,
                f"{self.place_name}_vulnerability_map_{self.distance_method}_{self.evaluation_time}.html",
            )
            self._create_interactive_map(output_html)
            self.log(f"Saved interactive map to {output_html}", level="success")

        # Return results
        return {
            "vulnerability_zones": self.vulnerability_zones,
            "potis": self.poti_df,
            "clusters": self.cluster_centers,
            "epoch_time": self.epoch_time,
        }

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
        if (
            self.config
            and hasattr(self.config, "map_center")
            and self.place_name in self.config.map_center
        ):
            map_center = [
                self.config.map_center[self.place_name]["lat"],
                self.config.map_center[self.place_name]["lon"],
            ]
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
        boundary_path = None
        if self.config and hasattr(self.config, "boundary_paths"):
            boundary_path = self.config.boundary_paths.get(self.place_name)
        else:
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
