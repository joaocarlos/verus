import os

import folium
import pandas as pd
from folium.plugins import MarkerCluster

from verus.utils.logger import Logger
from verus.utils.paths import PathManager


class Cluster(Logger):
    """Base class for clustering algorithms with common functionality."""

    def __init__(self, output_dir=None, verbose=True):
        """
        Initialize the base clusterer.

        Args:
            output_dir (str, optional): Base directory for output files
            verbose (bool): Whether to print informational messages
        """
        super().__init__(verbose=verbose)

        # Initialize path manager
        self.paths = PathManager(output_dir=output_dir)

    def load(self, data_source):
        """
        Load data from a file or DataFrame.

        Args:
            data_source (str or pd.DataFrame): Path to CSV file or DataFrame

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if isinstance(data_source, pd.DataFrame):
                self.log("Using provided DataFrame")
                return data_source
            elif isinstance(data_source, str) and os.path.exists(data_source):
                self.log(f"Loading data from file: {data_source}")
                return pd.read_csv(data_source)
            else:
                raise ValueError(f"Invalid data source: {data_source}")
        except Exception as e:
            self.log(f"Error loading data: {e}", "error")
            return None

    def save(self, cluster_df, centroids_df, place_name, suffix="", algorithm=""):
        """
        Save clustering results to CSV files.

        Args:
            cluster_df (pd.DataFrame): DataFrame with cluster assignments
            centroids_df (pd.DataFrame): DataFrame with cluster centroids
            place_name (str): Name of the location
            suffix (str): Additional suffix for filenames
            algorithm (str): Algorithm name

        Returns:
            tuple: (clusters_path, centroids_path) - Paths to the saved files
        """
        try:
            # Format suffix
            suffix = f"_{suffix}" if suffix else ""

            # Save clusters
            clusters_path = self.paths.get_path(
                "clusters", f"{place_name}_Clusters_{algorithm}{suffix}.csv"
            )
            cluster_df.to_csv(clusters_path, index=False)
            self.log(f"Saved clusters to: {clusters_path}", "success")

            # Save centroids if available
            if centroids_df is not None:
                centroids_path = self.paths.get_path(
                    "clusters", f"{place_name}_Centroids_{algorithm}{suffix}.csv"
                )
                centroids_df.to_csv(centroids_path, index=False)
                self.log(f"Saved centroids to: {centroids_path}", "success")
                return clusters_path, centroids_path
            else:
                return clusters_path, None

        except Exception as e:
            self.log(f"Failed to save results: {e}", "error")
            return None, None

    def create_map(self, cluster_df, centroids_df=None, boundary_path=None):
        """
        Create an interactive map with clusters.

        Args:
            cluster_df (pd.DataFrame): DataFrame with cluster assignments
            centroids_df (pd.DataFrame, optional): DataFrame with cluster centroids
            boundary_path (str, optional): Path to boundary GeoJSON

        Returns:
            folium.Map: Interactive map object
        """
        try:
            # Calculate map center
            center_lat = cluster_df["latitude"].mean()
            center_lon = cluster_df["longitude"].mean()

            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles="CartoDB positron",
            )

            # Add boundary if provided
            if boundary_path and os.path.exists(boundary_path):
                folium.GeoJson(
                    boundary_path,
                    name="Boundary",
                    style_function=lambda x: {
                        "fillColor": "transparent",
                        "color": "blue",
                        "weight": 2,
                    },
                ).add_to(m)

            # Add clusters
            marker_cluster = MarkerCluster(name="POIs").add_to(m)

            # Add POIs
            for _, row in cluster_df.iterrows():
                popup_text = f"""
                    <b>ID:</b> {row.get('id', 'N/A')}<br>
                    <b>Category:</b> {row.get('category', 'N/A')}<br>
                    <b>Cluster:</b> {row.get('cluster_id', 'N/A')}
                """
                if "vi" in row:
                    popup_text += f"<br><b>VI:</b> {row['vi']}"

                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=5,
                    color=self._get_cluster_color(row.get("cluster_id", -1)),
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300),
                ).add_to(marker_cluster)

            # Add centroids if available
            if centroids_df is not None:
                for _, row in centroids_df.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=10,
                        color="black",
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"Cluster {row['cluster_id']}: {row.get('size', 'N/A')} points",
                    ).add_to(m)

            # Add layer control
            folium.LayerControl().add_to(m)

            return m

        except Exception as e:
            self.log(f"Error creating map: {e}", "error")
            return None

    def _get_cluster_color(self, cluster_id):
        """Get color for cluster visualization."""
        # List of colors for different clusters
        colors = [
            "#e6194B",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabed4",
            "#469990",
            "#dcbeff",
            "#9A6324",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#808000",
            "#ffd8b1",
            "#000075",
            "#a9a9a9",
        ]

        # Handle noise points (cluster_id = -1)
        if cluster_id == -1:
            return "#000000"  # Black for noise

        # Use modulo to cycle through colors for many clusters
        return colors[cluster_id % len(colors)]
