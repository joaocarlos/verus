"""
K-means clustering module for the Verus project.

This module provides functionality for applying K-means clustering to geospatial data
using the Haversine distance metric, which correctly accounts for the Earth's curvature.
"""

import os

import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from verus.utils.logger import Logger


class KMeansHaversine(Logger):
    """
    Perform K-means clustering on geospatial data using Haversine distance.

    This class implements K-means with Haversine distance to correctly cluster
    geographic coordinates, with support for vulnerability index weighting and
    predefined centers from other clustering methods like OPTICS.

    Attributes:
        n_clusters (int): Number of clusters to form.
        init (str): Initialization method ('k-means++', 'random', or 'predefined').
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        random_state (int): Random seed for reproducibility.
        predefined_centers (ndarray): Predefined initial centroids.

    Examples:
        >>> kmeans = KMeansHaversine(n_clusters=8)
        >>> results = kmeans.run(data_source=poi_df)
        >>> clusters_df = results["clusters"]
        >>> centroids_df = results["centroids"]
        >>> saved_files = kmeans.save(clusters_df, centroids_df)
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        predefined_centers=None,
        verbose=True,
    ):
        """
        Initialize the KMeansHaversine clusterer with clustering parameters.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form.
        init : str
            Initialization method ('k-means++', 'random', or 'predefined').
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        random_state : int, optional
            Random seed for reproducibility.
        predefined_centers : ndarray, optional
            Predefined initial centroids, required if init='predefined'.
        verbose : bool
            Whether to print informational messages.

        Raises
        ------
        ValueError
            If parameters are invalid or incompatible.
        """
        # Initialize the Logger
        super().__init__(verbose=verbose)

        # Validate parameters
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

        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float
            Coordinates in degrees

        Returns
        -------
        float
            Distance in kilometers
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

        Parameters
        ----------
        X, Y : ndarray
            Arrays of shape (N, 2) and (M, 2) with lat/lon coordinates

        Returns
        -------
        ndarray
            Matrix of distances with shape (N, M)
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

        This method properly accounts for the curvature of the Earth.

        Parameters
        ----------
        points : ndarray
            Array of shape (N, 2) with lat/lon in degrees
        weights : ndarray, optional
            Optional weights for each point

        Returns
        -------
        ndarray
            Centroid coordinates [lat, lon]
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

        Parameters
        ----------
        X : ndarray
            Array of shape (n_samples, 2) with lat/lon coordinates
        sample_weights : ndarray, optional
            Optional weights for each sample

        Returns
        -------
        self
            The fitted model

        Raises
        ------
        ValueError
            If clustering fails for any reason
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

        Parameters
        ----------
        X : ndarray
            Array of shape (n_samples, 2) with lat/lon coordinates

        Returns
        -------
        ndarray
            Cluster labels for each point
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted, call fit first")

        dist = self._haversine_matrix(X, self.cluster_centers_)
        labels = np.argmin(dist, axis=1)
        return labels

    def load(self, data_source):
        """
        Load POI data for clustering.

        Parameters
        ----------
        data_source : str or pd.DataFrame
            Path to POI CSV file or DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame containing points of interest

        Raises
        ------
        ValueError
            If the data cannot be loaded or processed
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
            required_columns = ["latitude", "longitude"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                self.log(f"Data missing required columns: {missing_columns}", "error")
                raise ValueError(f"Data missing required columns: {missing_columns}")

            self.log(f"Loaded {len(df)} points of interest")

            return df

        except Exception as e:
            self.log(f"Failed to load data: {str(e)}", "error")
            raise ValueError(f"Failed to load data: {str(e)}")

    def _load_predefined_centers(self, centers_path=None, centers_df=None):
        """
        Load predefined centers from a file or DataFrame.

        Private method for internal use.

        Parameters
        ----------
        centers_path : str, optional
            Path to centroids CSV file
        centers_df : pd.DataFrame, optional
            DataFrame with centroids

        Returns
        -------
        ndarray
            Array of centroids

        Raises
        ------
        ValueError
            If neither path nor DataFrame is provided
        """
        if centers_df is not None:
            # Use provided DataFrame
            if (
                "latitude" not in centers_df.columns
                or "longitude" not in centers_df.columns
            ):
                self.log("Centers DataFrame missing required columns", "error")
                raise ValueError(
                    "Centers DataFrame must have latitude and longitude columns"
                )

            centers = centers_df[["latitude", "longitude"]].values
            self.log(f"Using {len(centers)} predefined centers from DataFrame")
            return centers

        elif centers_path is not None and os.path.exists(centers_path):
            # Load from file
            centers_df = pd.read_csv(centers_path)

            if (
                "latitude" not in centers_df.columns
                or "longitude" not in centers_df.columns
            ):
                self.log("Centers file missing required columns", "error")
                raise ValueError(
                    "Centers file must have latitude and longitude columns"
                )

            centers = centers_df[["latitude", "longitude"]].values
            self.log(f"Loaded {len(centers)} predefined centers from {centers_path}")
            return centers
        else:
            raise ValueError("Either centers_path or centers_df must be provided")

    def _create_result_dataframes(self, df, coords, labels, centers):
        """
        Create DataFrames for clusters and centroids.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe with POI data
        coords : ndarray
            Coordinate array used for clustering
        labels : ndarray
            Cluster labels for each point
        centers : ndarray
            Cluster centers

        Returns
        -------
        tuple
            (cluster_df, centroids_df) - DataFrames for clusters and centroids
        """
        try:
            # Create cluster DataFrame
            cluster_data = []
            for i, label in enumerate(labels):
                point = coords[i]
                cluster_data.append([point[0], point[1], int(label)])

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

            # Create centroids DataFrame
            centroids_df = pd.DataFrame(centers, columns=["latitude", "longitude"])
            centroids_df["cluster"] = range(len(centers))

            # Calculate size of each cluster
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            centroids_df["size"] = [
                cluster_sizes.get(i, 0) for i in range(len(centers))
            ]

            # Reorder columns
            centroids_df = centroids_df[["cluster", "latitude", "longitude", "size"]]

            return cluster_df, centroids_df

        except Exception as e:
            self.log(f"Error creating result DataFrames: {e}", "error")
            return None, None

    def create_map(
        self, cluster_df, centroids_df=None, area_boundary_path=None, save_path=None
    ):
        """
        Create an interactive map visualizing the clustering results.

        Parameters
        ----------
        cluster_df : pd.DataFrame
            DataFrame with cluster assignments
        centroids_df : pd.DataFrame, optional
            DataFrame with cluster centroids
        area_boundary_path : str, optional
            Path to area boundary GeoJSON
        save_path : str, optional
            Path to save map HTML file. If None, the map is just returned

        Returns
        -------
        folium.Map or None
            Interactive map or None if creation fails
        """
        try:
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
            unique_clusters = sorted(cluster_df["cluster"].unique())
            cluster_to_index = {cluster: i for i, cluster in enumerate(unique_clusters)}
            n_clusters = len(unique_clusters)

            self.log(f"Creating map with {n_clusters} clusters")
            colors = [cmap(i / max(1, n_clusters - 1)) for i in range(n_clusters)]

            # Add the boundary area to the map if provided
            if area_boundary_path and os.path.exists(area_boundary_path):
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

                cluster_points = cluster_df[cluster_df["cluster"] == cluster_val]
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
                    Category: {row.get('category', 'Unknown')}
                    """
                    if "vi" in row:
                        popup_content += f"<br>VI: {row['vi']}"

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
                    cluster_id = row["cluster"]

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
            DataFrame with cluster assignments
        centroids_df : pd.DataFrame
            DataFrame with cluster centroids
        path : str, optional
            Directory or file path to save results.
            If None, uses "./data/clusters".
        evaluation_time : str, optional
            Time scenario identifier to include in filenames.

        Returns
        -------
        dict
            Dictionary with paths to saved files
        """
        if cluster_df is None:
            self.log("No cluster data to save", "warning")
            return {}

        try:
            # Determine base directory and file names
            if path is None:
                save_dir = os.path.abspath("./data/clusters")
                filename_prefix = "kmeans"
            elif os.path.isdir(path) or not path.endswith(".csv"):
                save_dir = os.path.abspath(path)
                filename_prefix = "kmeans"
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
        centers_input=None,
        area_boundary_path=None,
    ):
        """
        Run the complete K-means clustering workflow.

        Parameters
        ----------
        data_source : str or pd.DataFrame
            Path to POI CSV file or DataFrame containing points to cluster.
            Can be the output of GeOPTICS's time-filtered data.
        centers_input : str or pd.DataFrame, optional
            Predefined centers file path or DataFrame.
            Can be the output of GeOPTICS's centroids.
        area_boundary_path : str, optional
            Path to area boundary GeoJSON file for map visualization.

        Returns
        -------
        dict
            Dictionary containing clustering results:
                - 'clusters': DataFrame with cluster assignments
                - 'centroids': DataFrame with cluster centroids
                - 'map': Folium map object if area_boundary_path was provided
                - 'input_data': Input data used for clustering
                - 'place_name': Extracted place name (if available)
                - 'labels': Cluster labels array
                - 'inertia': K-means inertia value
                - 'n_iter': Number of iterations run
        """
        try:
            # Extract place name from path if possible
            place_name = self._extract_place_name(data_source)

            # Load data
            df = self.load(data_source)

            # Extract coordinates for clustering
            coords = df[["latitude", "longitude"]].to_numpy()

            # Handle predefined centers if provided
            if centers_input is not None:
                # Extract predefined centers
                if isinstance(centers_input, pd.DataFrame):
                    predefined_centers = centers_input[
                        ["latitude", "longitude"]
                    ].to_numpy()
                    self.log(
                        f"Using provided DataFrame with {len(predefined_centers)} centers"
                    )
                else:  # Assume it's a path
                    predefined_centers = self._load_predefined_centers(
                        centers_path=centers_input
                    )

                # Update class attributes
                self.predefined_centers = predefined_centers
                self.n_clusters = len(predefined_centers)
                self.init = "predefined"  # Always use predefined method when centers are provided

            # Set up sample weights from vulnerability index if available
            sample_weights = None
            if "vi" in df.columns:
                self.log("Using vulnerability indices as sample weights")
                sample_weights = df["vi"].values

            # Run K-means
            self.fit(
                coords, sample_weights
            )  # Mark as private by adding underscore prefix

            # Create result dataframes
            cluster_df, centroids_df = (
                self._create_result_dataframes(  # Mark as private
                    df, coords, self.labels_, self.cluster_centers_
                )
            )

            # Create map if boundary path was provided
            map_obj = None
            if area_boundary_path and cluster_df is not None:
                map_obj = self.create_map(cluster_df, centroids_df, area_boundary_path)

            # Return results
            return {
                "clusters": cluster_df,
                "centroids": centroids_df,
                "map": map_obj,
                "input_data": df,
                "place_name": place_name,
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
                "place_name": None,
                "labels": None,
                "inertia": None,
                "n_iter": None,
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
