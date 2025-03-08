import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from verus.clustering.base import Cluster


class KMeansHaversine(Cluster):
    """
    Class for performing K-means clustering on geospatial data using Haversine distance.

    This class implements K-means with Haversine distance to correctly cluster
    geographic coordinates, with support for vulnerability index weighting and
    predefined centers from other clustering methods like OPTICS.
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        predefined_centers=None,
        output_dir=None,
        verbose=True,
    ):
        """
        Initialize the KMeansClusterer with clustering parameters.

        Args:
            n_clusters (int): Number of clusters to form.
            init (str): Initialization method ('k-means++', 'random', or 'predefined').
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.
            random_state (int, optional): Random seed for reproducibility.
            predefined_centers (ndarray, optional): Predefined initial centroids.
            output_dir (str, optional): Base directory for output files.
            verbose (bool): Whether to print informational messages.
        """
        # Initialize the base clusterer
        super().__init__(output_dir=output_dir, verbose=verbose)

        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")

        if init not in ["k-means++", "random", "predefined"]:
            raise ValueError("init must be 'k-means++', 'random', or 'predefined'")

        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("tol must be a positive number")

        if init == "predefined" and predefined_centers is None:
            raise ValueError(
                "Predefined centers must be provided when init='predefined'"
            )

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.predefined_centers = predefined_centers

        # Results attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points.

        Args:
            lat1, lon1, lat2, lon2: Coordinates in degrees

        Returns:
            float: Distance in kilometers
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        # Earth radius in kilometers
        r = 6371.0
        return r * c

    def _haversine_matrix(self, X, Y):
        """
        Compute a distance matrix between X and Y using Haversine distance.

        Args:
            X, Y: Arrays of shape (N, 2) and (M, 2) with lat/lon coordinates

        Returns:
            ndarray: Matrix of distances with shape (N, M)
        """
        lat1 = X[:, 0][:, np.newaxis]
        lon1 = X[:, 1][:, np.newaxis]
        lat2 = Y[:, 0]
        lon2 = Y[:, 1]

        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        r = 6371.0
        dist = r * c
        return dist

    def _init_centroids_kmeans_pp(self, X):
        """
        Initialize centroids using k-means++ method.
        """
        self.log("Initializing centroids using k-means++")
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Choose one center uniformly at random
        centers = []
        first_center_idx = rng.integers(0, n_samples)
        centers.append(X[first_center_idx])

        # Compute D(x): distances to the nearest chosen center
        dist = self._haversine_matrix(X, np.array([centers[0]]))
        dist = dist.reshape(-1)

        for i in range(1, self.n_clusters):
            # Choose a new center weighted by D(x)^2
            dist_sq = dist**2
            probabilities = dist_sq / dist_sq.sum()
            new_center_idx = rng.choice(n_samples, p=probabilities)
            centers.append(X[new_center_idx])
            self.log(f"Initialized center {i+1}/{self.n_clusters}")

            # Update D(x)
            new_dist = self._haversine_matrix(X, np.array([X[new_center_idx]]))
            new_dist = new_dist.reshape(-1)
            dist = np.minimum(dist, new_dist)

        return np.array(centers)

    def _init_centroids_random(self, X):
        """
        Initialize centroids by randomly selecting points from the dataset.
        """
        self.log("Initializing centroids randomly")
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _init_centroids_predefined(self, X):
        """
        Use predefined centroids as starting points.
        """
        self.log("Using predefined centroids")
        if self.predefined_centers is None:
            raise ValueError(
                "Predefined centers must be provided for this initialization method."
            )
        return np.array(self.predefined_centers)

    @staticmethod
    def _centroid_on_sphere(points, weights=None):
        """
        Calculate the centroid of points on a sphere (Earth's surface).
        This properly accounts for the curvature of the Earth.

        Args:
            points: Array of shape (N, 2) with lat/lon in degrees
            weights: Optional weights for each point

        Returns:
            ndarray: Centroid coordinates [lat, lon]
        """
        # Convert to cartesian coordinates
        lat_rad = np.radians(points[:, 0])
        lon_rad = np.radians(points[:, 1])
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        if weights is not None:
            total_weight = np.sum(weights)
            if total_weight != 0:
                x_mean = np.average(x, weights=weights)
                y_mean = np.average(y, weights=weights)
                z_mean = np.average(z, weights=weights)
            else:
                x_mean = x.mean()
                y_mean = y.mean()
                z_mean = z.mean()
        else:
            x_mean = x.mean()
            y_mean = y.mean()
            z_mean = z.mean()

        # Convert back to lat/lon
        hyp = np.sqrt(x_mean**2 + y_mean**2)
        new_lat = np.degrees(np.arctan2(z_mean, hyp))
        new_lon = np.degrees(np.arctan2(y_mean, x_mean))

        return np.array([new_lat, new_lon])

    def _compute_inertia(self, X, labels, centers, sample_weights):
        """
        Compute the sum of squared distances of samples to their closest centroid.
        """
        # Inertia: sum of squared distances to the closest centroid
        dist = self._haversine_matrix(X, centers)
        # dist[i, labels[i]] gives the distance to assigned center
        assigned_dist = dist[np.arange(X.shape[0]), labels]
        inertia = np.sum(sample_weights * (assigned_dist**2))
        return inertia

    def fit(self, X, sample_weights=None):
        """
        Fit the K-means model to the data.

        Args:
            X: Array of shape (n_samples, 2) with lat/lon coordinates
            sample_weights: Optional weights for each sample

        Returns:
            self: The fitted model
        """
        try:
            self.log(f"Starting K-means clustering with {self.n_clusters} clusters")

            if sample_weights is None:
                sample_weights = np.ones(X.shape[0])

            self.log(f"Using {self.init} initialization method")
            # Initialization
            if self.init == "k-means++":
                centers = self._init_centroids_kmeans_pp(X)
            elif self.init == "random":
                centers = self._init_centroids_random(X)
            elif self.init == "predefined":
                centers = self._init_centroids_predefined(X)
            else:
                raise ValueError(f"Unknown initialization method: {self.init}")

            # Iterative refinement
            for i in range(self.max_iter):
                self.log(f"K-means iteration {i+1}/{self.max_iter}")

                # Assignment step
                dist = self._haversine_matrix(X, centers)
                labels = np.argmin(dist, axis=1)

                # Update step - compute new centers
                new_centers = []
                for k in range(self.n_clusters):
                    cluster_points = X[labels == k]
                    cluster_weights = sample_weights[labels == k]
                    if len(cluster_points) == 0:
                        # If a cluster is empty, reinitialize its center randomly
                        rng = np.random.default_rng(self.random_state)
                        new_centers.append(X[rng.integers(0, X.shape[0])])
                        self.log(
                            f"Cluster {k} is empty, reinitializing center randomly",
                            "warning",
                        )
                    else:
                        # Compute centroid on a sphere with weights
                        new_centers.append(
                            self._centroid_on_sphere(cluster_points, cluster_weights)
                        )
                new_centers = np.array(new_centers)

                # Check for convergence
                shift = self._haversine_matrix(centers, new_centers)
                # Max cluster center shift
                max_shift = np.max(np.diag(shift))
                self.log(f"Maximum centroid shift: {max_shift:.6f} km")

                centers = new_centers

                if max_shift < self.tol:
                    self.log(f"Converged after {i+1} iterations", "success")
                    break

            # Store the results
            self.cluster_centers_ = centers
            self.labels_ = labels
            self.inertia_ = self._compute_inertia(X, labels, centers, sample_weights)
            self.n_iter_ = i + 1

            self.log(f"K-means completed with inertia: {self.inertia_:.4f}", "success")
            return self

        except Exception as e:
            self.log(f"Error in K-means clustering: {e}", "error")
            raise ValueError(f"K-means clustering failed: {e}")

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Args:
            X: Array of shape (n_samples, 2) with lat/lon coordinates

        Returns:
            ndarray: Cluster labels for each point
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted, call fit first")

        dist = self._haversine_matrix(X, self.cluster_centers_)
        labels = np.argmin(dist, axis=1)
        return labels

    def load_data(self, data_source):
        """
        Load POI data. Simplified to just load data without time window processing.

        Args:
            data_source (str or pd.DataFrame): Path to POI CSV file or DataFrame

        Returns:
            tuple: (DataFrame, ndarray) - DataFrame with POI data and coordinate array
        """
        try:
            # Use base class method to load data
            df = super().load_data(data_source)

            # Basic validation
            required_columns = ["latitude", "longitude"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.log(f"Data missing required columns: {missing_columns}", "error")
                raise ValueError(f"Data missing required columns: {missing_columns}")

            self.log(f"Loaded {len(df)} points of interest")

            # Extract coordinates
            coords = df[["latitude", "longitude"]].to_numpy()
            return df, coords

        except Exception as e:
            self.log(f"Failed to load data: {e}", "error")
            raise ValueError(f"Failed to load data: {e}")

    def load_predefined_centers(self, centers_path=None, centers_df=None):
        """
        Load predefined centers from a file or DataFrame.

        Args:
            centers_path (str, optional): Path to centroids CSV file
            centers_df (pd.DataFrame, optional): DataFrame with centroids

        Returns:
            ndarray: Array of centroids
        """
        try:
            if centers_df is not None:
                self.log("Using provided DataFrame for centroids")
                predefined_centers = centers_df[["latitude", "longitude"]].to_numpy()
            elif centers_path is not None:
                self.log(f"Loading centroids from: {centers_path}")
                predefined_centers = pd.read_csv(centers_path)[
                    ["latitude", "longitude"]
                ].to_numpy()
            else:
                self.log("No centroids provided", "error")
                raise ValueError("Either centers_path or centers_df must be provided")

            self.n_clusters = len(predefined_centers)
            self.predefined_centers = predefined_centers
            self.log(f"Loaded {self.n_clusters} predefined centroids")

            return predefined_centers

        except Exception as e:
            self.log(f"Failed to load predefined centers: {e}", "error")
            raise ValueError(f"Failed to load predefined centers: {e}")

    def create_result_dataframes(self, coords, labels, centers, original_df):
        """
        Create DataFrames for clusters and centroids.

        Args:
            coords (ndarray): Coordinate array used for clustering
            labels (ndarray): Cluster labels for each point
            centers (ndarray): Cluster centers
            original_df (pd.DataFrame): Original dataframe with POI data

        Returns:
            tuple: (cluster_df, centroids_df) - DataFrames for clusters and centroids
        """
        try:
            # Create cluster DataFrame
            cluster_data = []
            for i in range(len(centers)):
                for point_idx in np.where(labels == i)[0]:
                    point = coords[point_idx]
                    cluster_data.append([point[0], point[1], i])

            cluster_df = pd.DataFrame(
                cluster_data, columns=["latitude", "longitude", "cluster_id"]
            )

            # Merge with original data to include category, name, and vi
            columns_to_include = ["latitude", "longitude"]
            for col in ["category", "name", "vi"]:
                if col in original_df.columns:
                    columns_to_include.append(col)

            cluster_df = cluster_df.merge(
                original_df[columns_to_include],
                on=["latitude", "longitude"],
                how="left",
            )

            # Create centroids DataFrame
            centroids_df = pd.DataFrame(centers, columns=["latitude", "longitude"])
            centroids_df["cluster_id"] = range(len(centers))

            # Calculate size of each cluster
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            centroids_df["size"] = [
                cluster_sizes.get(i, 0) for i in range(len(centers))
            ]

            # Reorder columns
            centroids_df = centroids_df[["cluster_id", "latitude", "longitude", "size"]]

            return cluster_df, centroids_df

        except Exception as e:
            self.log(f"Error creating result DataFrames: {e}", "error")
            return None, None

    def create_kmeans_map(self, cluster_df, centroids_df=None, area_boundary_path=None):
        """
        Create a specialized KMeans map with custom coloring.

        Args:
            cluster_df (pd.DataFrame): DataFrame with cluster assignments
            centroids_df (pd.DataFrame, optional): DataFrame with cluster centroids
            area_boundary_path (str, optional): Path to area boundary GeoJSON

        Returns:
            folium.Map: Interactive map
        """
        try:
            import folium

            if cluster_df is None or len(cluster_df) == 0:
                self.log("No data available to create map", "warning")
                return None

            # Get center of the map
            map_center = [cluster_df["latitude"].mean(), cluster_df["longitude"].mean()]

            # Create map
            self.log("Creating interactive map")
            m = folium.Map(location=map_center, zoom_start=13, tiles="Cartodb Positron")

            # Create a colormap for clusters
            cmap = plt.get_cmap("turbo")

            # Get unique cluster values and create a mapping to indices
            unique_clusters = sorted(cluster_df["cluster_id"].unique())
            cluster_to_index = {cluster: i for i, cluster in enumerate(unique_clusters)}
            n_clusters = len(unique_clusters)

            self.log(f"Creating map with {n_clusters} clusters")
            colors = [cmap(i / max(1, n_clusters - 1)) for i in range(n_clusters)]

            # Add the boundary area to the map if provided
            if area_boundary_path:
                self.log(f"Adding area boundary from: {area_boundary_path}")
                try:
                    folium.GeoJson(
                        area_boundary_path,
                        name="boundary",
                        style_function=lambda feature: {
                            "color": "#808080",
                            "weight": 2,
                            "fillOpacity": 0,
                        },
                    ).add_to(m)
                except Exception as e:
                    self.log(f"Error adding boundary: {e}", "warning")

            # Create feature groups for each cluster
            for cluster_val in unique_clusters:
                # Get the color index from our mapping
                k = cluster_to_index[cluster_val]
                color = colors[k]

                cluster_points = cluster_df[cluster_df["cluster_id"] == cluster_val]
                if len(cluster_points) == 0:
                    continue

                # Create a feature group for this cluster
                fg = folium.FeatureGroup(name=f"Cluster {cluster_val}")
                hex_color = mpl.colors.rgb2hex(color)

                # Add points to the feature group
                for _, row in cluster_points.iterrows():
                    # Create popup content
                    popup_content = f"""
                    <b>{row.get('name', 'Unknown')}</b><br>
                    Category: {row.get('category', 'Unknown')}<br>
                    VI: {row.get('vi', 'N/A')}
                    """

                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=3,
                        color=hex_color,
                        stroke=False,
                        fill=True,
                        fill_color=hex_color,
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=f"Cluster {cluster_val}: {row.get('name', 'Unknown')}",
                    ).add_to(fg)

                fg.add_to(m)

            # Add centroids if provided
            if centroids_df is not None and not centroids_df.empty:
                centroid_group = folium.FeatureGroup(name="Cluster Centroids")

                for _, row in centroids_df.iterrows():
                    cluster_id = row["cluster_id"]

                    # Use the mapping to get the correct color index
                    if cluster_id in cluster_to_index and cluster_to_index[
                        cluster_id
                    ] < len(colors):
                        color_idx = cluster_to_index[cluster_id]
                        hex_color = mpl.colors.rgb2hex(colors[color_idx])
                    else:
                        hex_color = "#FF0000"  # Default to red if out of range

                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        icon=folium.Icon(color="red", icon="info-sign"),
                        popup=f"Centroid {cluster_id}: {row.get('size', 'N/A')} points",
                    ).add_to(centroid_group)

                centroid_group.add_to(m)

            # Add layer control
            folium.LayerControl().add_to(m)

            return m

        except Exception as e:
            self.log(f"Error creating map: {e}", "error")
            # Print more detailed error information for debugging
            import traceback

            self.log(f"Traceback: {traceback.format_exc()}", "error")
            return None

    def run(
        self,
        data_source,
        place_name=None,
        evaluation_time=None,
        centers_input=None,
        save_output=False,
        create_map_output=False,
        area_boundary_path=None,
        algorithm_suffix="KMeans",
    ):
        """
        Run the complete K-means clustering workflow.

        Args:
            data_source (str or pd.DataFrame): Path to POI CSV or DataFrame
            place_name (str, optional): Name of the location for output files
            evaluation_time (str or int, optional): Evaluation time (used only for file naming)
            centers_input (str or pd.DataFrame, optional): Predefined centers file or DataFrame
            save_output (bool): Whether to save results to files
            create_map_output (bool): Whether to create an interactive map
            area_boundary_path (str, optional): Path to area boundary GeoJSON
            algorithm_suffix (str): Suffix for output filenames

        Returns:
            dict: Dictionary with clustering results
        """
        try:
            # Extract place name from path if not provided
            if place_name is None and isinstance(data_source, str):
                filename = os.path.basename(data_source)
                if "_dataset" in filename:
                    place_name = filename.split("_dataset")[0]
                else:
                    place_name = "unknown"

            # Load data
            self.log("Loading and preparing data")
            df, coords = self.load_data(data_source)

            # Handle predefined centers
            if centers_input is not None:
                # Extract predefined centers
                if isinstance(centers_input, pd.DataFrame):
                    self.log(
                        f"Processing centers from DataFrame with shape {centers_input.shape}"
                    )
                    predefined_centers = centers_input[
                        ["latitude", "longitude"]
                    ].to_numpy()
                    self.log(f"Extracted centers with shape {predefined_centers.shape}")
                else:  # Assume it's a path
                    predefined_centers = self.load_predefined_centers(
                        centers_path=centers_input
                    )

                # Update class attributes
                self.predefined_centers = predefined_centers
                self.n_clusters = len(predefined_centers)

                # Always use predefined initialization method when centers are provided
                self.init = "predefined"
                self.log(f"Using {self.n_clusters} predefined centers")

            # Check if we're set up correctly when init="predefined"
            if self.init == "predefined" and self.predefined_centers is None:
                self.log(
                    "Error: predefined initialization selected but no centers provided",
                    "error",
                )
                raise ValueError(
                    "Predefined centers must be provided when init='predefined'"
                )

            # Set up sample weights from vulnerability index if available
            sample_weights = None
            if "vi" in df.columns:
                self.log("Using vulnerability indices as sample weights")
                sample_weights = np.array(df["vi"].to_list())

            # Run K-means
            self.fit(coords, sample_weights)

            # Create result dataframes
            cluster_df, centroids_df = self.create_result_dataframes(
                coords, self.labels_, self.cluster_centers_, df
            )

            # Save results if requested
            if save_output and place_name:
                # Use the base class method with our custom suffix
                self.save_results(
                    cluster_df,
                    centroids_df,
                    place_name,
                    evaluation_time,
                    algorithm_suffix,
                )

            # Create map if requested
            map_obj = None
            if create_map_output and cluster_df is not None:
                # Use either the specialized KMeans map or the base class map
                map_obj = self.create_kmeans_map(
                    cluster_df, centroids_df, area_boundary_path
                )

                # Save map if requested
                if save_output and map_obj and place_name:
                    # Format the evaluation time suffix
                    suffix = f"_{evaluation_time}" if evaluation_time else ""

                    # Use the paths manager from base class
                    map_path = self.paths.get_path(
                        "maps", f"{place_name}_Map_{algorithm_suffix}{suffix}.html"
                    )

                    map_obj.save(map_path)
                    self.log(f"Saved map to: {map_path}", "success")

            # Return results
            return {
                "clusters": cluster_df,
                "centroids": centroids_df,
                "map": map_obj,
                "input_data": df,
                "labels": self.labels_,
                "inertia": self.inertia_,
                "n_iter": self.n_iter_,
            }

        except Exception as e:
            self.log(f"Error in K-means workflow: {e}", "error")
            return {
                "clusters": None,
                "centroids": None,
                "map": None,
                "input_data": None,
                "labels": None,
                "inertia": None,
                "n_iter": None,
            }
