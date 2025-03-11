"""
Data extraction module for the Verus project.

This module provides functionality to extract Points of Interest (POIs) from
OpenStreetMap and process them into a structured format for further analysis.
"""

import os

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from verus.utils.logger import Logger
from verus.utils.timer import TimeoutException, with_timeout


class DataExtractor(Logger):
    """
    Extract and process Points of Temporal Influence (PoTIs) from OpenStreetMap.

    To streamline the development process and aligh with other projects we will call
    PoTIs as Points of Interest (POIs).

    This class fetches POIs from OpenStreetMap based on specified tags,
    processes them into a structured format, and returns them as DataFrames/GeoDataFrames.
    Optional saving to disk is supported.

    Attributes:
        region (str): The region name to extract data from.
        buffer_distance (float): Buffer distance in meters around the region.
        amenity_tags (dict): Dictionary of OSM tags to extract.
        place_name (str): Simplified name derived from region for file naming.

    Examples:
        >>> extractor = DataExtractor(region="Porto, Portugal")
        >>> df = extractor.run()
        >>> extractor.save(df, path="./data/porto_data.csv")
    """

    def __init__(
        self,
        region="Porto, Portugal",
        buffer_distance=500,
        amenity_tags=None,
        boundary_file=None,
        fetch_timeout=60,
        verbose=True,
    ):
        """
        Initialize the DataExtractor with input validation.

        Parameters
        ----------
        region : str or None
            Region to extract data from (e.g., "Porto, Portugal").
            Required if boundary_file is not provided.
        buffer_distance : int
            Buffer distance in meters around the region boundary.
        amenity_tags : dict, optional
            Custom amenity tags to extract. If None, defaults are used.
        boundary_file : str, optional
            Path to a boundary shapefile or GeoJSON. Used instead of region if provided.
        fetch_timeout : int
            Timeout in seconds for fetching data from OSM.
        verbose : bool
            Whether to print informational messages.

        Raises
        ------
        ValueError
            If neither region nor boundary_file is provided.
        """
        # Initialize the Logger
        super().__init__(verbose=verbose)

        # Check if at least one of region or boundary_file is provided
        if not region and not boundary_file:
            raise ValueError("Either region or boundary_file must be provided")

        self.region = region
        self.buffer_distance = buffer_distance
        self.fetch_timeout = fetch_timeout

        # Set up amenity tags
        self.amenity_tags = amenity_tags or self._get_default_amenity_tags()

        # Initialize state attributes
        self.boundary = None
        self.buffered_boundary = None
        self.boundary_file = boundary_file

        # Extract place name from region for file naming
        self.place_name = region.split(",")[0].strip() if region else "custom"

    def _get_default_amenity_tags(self):
        """
        Get the default amenity tags dictionary.

        Returns
        -------
        dict
            Default amenity tags for POI extraction.
        """
        # Default amenity categories and their OSM tags
        return {
            # Education
            "school": {"amenity": "school"},
            "university": {"amenity": "university"},
            # Healthcare
            "hospital": {"amenity": "hospital"},
            # Transportation
            "station": {"public_transport": "station"},
            "bus_station": {"amenity": "bus_station"},
            "aerodrome": {"aeroway": "aerodrome"},
            # Commercial & Industrial
            "shopping_mall": {"shop": "mall"},
            "industrial": {"landuse": "industrial"},
            # Recreation
            "park": {"leisure": "park"},
            "attraction": {"tourism": "attraction"},
        }

    def get_boundaries(self):
        """
        Get or fetch the geographical boundaries of the region.

        This method either returns cached boundaries, loads them from a file,
        or fetches them from OpenStreetMap.

        Returns
        -------
        tuple of geopandas.GeoDataFrame
            (Region boundary, Buffered boundary)

        Raises
        ------
        ValueError
            If the region boundary cannot be fetched or the file cannot be read.
        """
        # Return cached boundaries if available
        if self.boundary is not None and self.buffered_boundary is not None:
            return self.boundary, self.buffered_boundary

        # Load from boundary file if specified
        if self.boundary_file and os.path.exists(self.boundary_file):
            try:
                self.log(f"Loading boundary from file: {self.boundary_file}")
                self.boundary = gpd.read_file(self.boundary_file)

                # Check if the file contained valid geometry
                if self.boundary is None or self.boundary.empty:
                    raise ValueError(
                        f"Boundary file contains no valid geometries: {self.boundary_file}"
                    )

                # Create buffered boundary - first project to a projected CRS for accurate buffering
                projected_boundary = self.boundary.to_crs(
                    "EPSG:3857"
                )  # Web Mercator projection
                buffered_projected = projected_boundary.copy()
                buffered_projected.geometry = projected_boundary.buffer(
                    self.buffer_distance
                )

                # Project back to original CRS
                self.buffered_boundary = buffered_projected.to_crs(self.boundary.crs)

                return self.boundary, self.buffered_boundary

            except Exception as e:
                self.log(f"Error loading boundary file: {str(e)}", level="error")
                self.log("Falling back to OpenStreetMap for boundary", level="warning")
                # Continue to OSM fetching as fallback

        # Otherwise fetch from OSM
        try:
            self.log(f"Fetching boundaries for {self.region} from OpenStreetMap...")
            # Fetch boundary geometry using OSMNX with timeout
            boundary = self._fetch_features_with_timeout(
                lambda: ox.geocode_to_gdf(self.region)
            )

            if boundary is None or boundary.empty:
                raise ValueError(f"Could not fetch boundary for {self.region}")

            # Extract the first polygon (in case multiple were returned)
            boundary = boundary.iloc[[0]]

            # Create a buffer around the boundary - properly handling projection
            # First project to a projected CRS for accurate buffering
            projected_boundary = boundary.to_crs("EPSG:3857")  # Web Mercator projection
            buffered_projected = projected_boundary.copy()
            buffered_projected.geometry = projected_boundary.buffer(
                self.buffer_distance
            )

            # Project back to original CRS
            buffered = buffered_projected.to_crs(boundary.crs)

            self.boundary = boundary
            self.buffered_boundary = buffered

            return boundary, buffered

        except Exception as e:
            self.log(f"Error fetching boundaries: {str(e)}", level="error")
            raise ValueError(f"Failed to obtain boundary for {self.region}: {str(e)}")

    def run(self):
        """
        Extract POIs and process them into a unified DataFrame.

        This method fetches POIs from OpenStreetMap for all amenity categories
        and combines them into a single DataFrame with consistent structure.

        Returns
        -------
        pandas.DataFrame
            DataFrame with POIs and their attributes.
        """
        # Get or fetch region boundaries
        try:
            boundary, buffered_boundary = self.get_boundaries()
        except ValueError as e:
            self.log(f"Could not get boundaries: {str(e)}", level="error")
            return pd.DataFrame()

        # Extract POIs for each category
        all_pois = []
        total_categories = len(self.amenity_tags)
        success_count = 0

        for i, (category, tags) in enumerate(self.amenity_tags.items(), 1):
            self.log(f"Extracting {category} POIs ({i}/{total_categories})...")

            try:
                # Extract POIs for this category
                pois_gdf = self._extract_category_pois(
                    category, tags, buffered_boundary
                )

                if pois_gdf is not None and len(pois_gdf) > 0:
                    # Convert to regular DataFrame with lat/lon
                    pois_df = self._gdf_to_dataframe(pois_gdf, category)
                    all_pois.append(pois_df)
                    success_count += 1

            except Exception as e:
                self.log(f"Error extracting {category} POIs: {str(e)}", level="error")
                # Continue with other categories

        # Combine all POIs into a unified dataset
        if not all_pois:
            self.log("No POIs extracted.", level="warning")
            return pd.DataFrame()

        # Merge all POI dataframes
        combined_df = pd.concat(all_pois, ignore_index=True)
        self.log(
            f"Extracted {len(combined_df)} POIs across {success_count}/{total_categories} categories.",
            level="info",
        )

        return combined_df

    def save(self, dataset, path=None, save_individual_categories=False):
        """
        Save extracted data to disk.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Dataset to save.
        path : str, optional
            Where to save the output files. Can be:
            - A directory path: Files will be saved in this directory using region name
            - A file path with extension: Dataset saved at this exact path, related files
              will use the same basename
            If None, uses "./data/{region_name}" as the directory.
        save_individual_categories : bool, optional
            Whether to save individual category files.

        Returns
        -------
        dict
            Dictionary with paths to all saved files.

        Raises
        ------
        ValueError
            If the dataset is empty or missing required columns.
        """
        # Input validation
        if dataset is None or dataset.empty:
            self.log("No data to save", level="error")
            return {}

        required_columns = ["latitude", "longitude", "category"]
        missing_columns = [
            col for col in required_columns if col not in dataset.columns
        ]
        if missing_columns:
            self.log(
                f"Dataset is missing required columns: {missing_columns}", level="error"
            )
            return {}

        # Determine directories and file naming
        if path is None:
            # Default path using region name
            base_dir = os.path.abspath(os.path.join("./data", self.place_name))
            basename = self.place_name
            is_file_path = False
        elif path.endswith(".csv") or path.endswith(".json"):
            # User specified exact file path
            base_dir = os.path.abspath(os.path.dirname(path))
            if not base_dir or base_dir == ".":
                base_dir = os.path.abspath(".")
            basename = os.path.splitext(os.path.basename(path))[0]
            is_file_path = True
        else:
            # User specified directory path
            base_dir = os.path.abspath(path)
            basename = self.place_name
            is_file_path = False

        # Create directory structure
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError as e:
            self.log(f"Failed to create directory {base_dir}: {str(e)}", level="error")
            return {}

        # Determine paths for all files
        if is_file_path:
            # User specified a file path
            dataset_path = path
            # Put related files in the same directory
            buffered_path = os.path.join(base_dir, f"{basename}_buffered.csv")
            boundary_path = os.path.join(base_dir, f"{basename}_boundary.geojson")
            buffered_boundary_path = os.path.join(
                base_dir, f"{basename}_buffered_boundary.geojson"
            )
        else:
            # Use conventional directory structure
            poti_dir = os.path.join(base_dir, "poti")
            geojson_dir = os.path.join(base_dir, "geojson")

            try:
                os.makedirs(poti_dir, exist_ok=True)
                os.makedirs(geojson_dir, exist_ok=True)
            except OSError as e:
                self.log(f"Failed to create subdirectories: {str(e)}", level="error")
                return {}

            dataset_path = os.path.join(poti_dir, f"{basename}_dataset.csv")
            buffered_path = os.path.join(poti_dir, f"{basename}_dataset_buffered.csv")
            boundary_path = os.path.join(geojson_dir, f"{basename}_boundary.geojson")
            buffered_boundary_path = os.path.join(
                geojson_dir, f"{basename}_buffered_boundary.geojson"
            )

        # Save files
        saved_files = {}

        # Save main dataset
        try:
            dataset.to_csv(dataset_path, index=False)
            self.log(f"Saved dataset to: {dataset_path}", level="info")
            saved_files["dataset"] = dataset_path
        except Exception as e:
            self.log(
                f"Failed to save dataset to {dataset_path}: {str(e)}", level="error"
            )

        # Save buffered dataset (for backward compatibility)
        try:
            dataset.to_csv(buffered_path, index=False)
            self.log(f"Saved buffered dataset to: {buffered_path}", level="info")
            saved_files["buffered_dataset"] = buffered_path
        except Exception as e:
            self.log(f"Failed to save buffered dataset: {str(e)}", level="warning")

        # Save boundaries if they exist
        if self.boundary is not None and self.buffered_boundary is not None:
            try:
                self.boundary.to_file(boundary_path, driver="GeoJSON")
                self.buffered_boundary.to_file(buffered_boundary_path, driver="GeoJSON")

                self.log(f"Saved boundary to: {boundary_path}", level="info")
                self.log(
                    f"Saved buffered boundary to: {buffered_boundary_path}",
                    level="info",
                )

                saved_files["boundary"] = boundary_path
                saved_files["buffered_boundary"] = buffered_boundary_path
            except Exception as e:
                self.log(f"Failed to save boundary files: {str(e)}", level="warning")

        # Save individual category files if requested
        if save_individual_categories:
            self.log("Saving individual category files...", level="info")
            category_files = {}

            # Group by category
            for category, group in dataset.groupby("category"):
                if is_file_path:
                    # Place category files next to the main file
                    category_csv = os.path.join(base_dir, f"{basename}_{category}.csv")
                    category_geojson = os.path.join(
                        base_dir, f"{basename}_{category}.geojson"
                    )
                else:
                    # Use conventional directory structure
                    category_csv = os.path.join(poti_dir, f"{basename}_{category}.csv")
                    category_geojson = os.path.join(
                        geojson_dir, f"{basename}_{category}.geojson"
                    )

                try:
                    # Save CSV
                    group.to_csv(category_csv, index=False)

                    # Convert back to GeoDataFrame for GeoJSON
                    geometry = [
                        Point(row.longitude, row.latitude)
                        for _, row in group.iterrows()
                    ]
                    gdf = gpd.GeoDataFrame(group, geometry=geometry, crs="EPSG:4326")

                    # Save GeoJSON
                    gdf.to_file(category_geojson, driver="GeoJSON")

                    category_files[category] = {
                        "csv": category_csv,
                        "geojson": category_geojson,
                    }

                    self.log(f"Saved {len(group)} {category} POIs", level="info")
                except Exception as e:
                    self.log(
                        f"Failed to save {category} files: {str(e)}", level="warning"
                    )

            if category_files:
                saved_files["categories"] = category_files

        return saved_files

    def _gdf_to_dataframe(self, gdf, category):
        """
        Convert a GeoDataFrame to a regular DataFrame with explicit lat/lon columns.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with point geometries
        category : str
            Category name for the POIs

        Returns
        -------
        pandas.DataFrame
            DataFrame with latitude and longitude columns
        """
        # Convert to DataFrame with explicit lat/lon columns
        df = pd.DataFrame(gdf)

        # Extract coordinates from geometry
        if "geometry" in df.columns:
            df["latitude"] = df.geometry.apply(lambda point: point.y if point else None)
            df["longitude"] = df.geometry.apply(
                lambda point: point.x if point else None
            )
            df.drop("geometry", axis=1, inplace=True)

        # Add category column
        df["category"] = category

        # Add default vulnerability influence (vi) value of 0.0
        df["vi"] = 0.0

        return df

    def _extract_category_pois(self, category, tags, buffered_boundary):
        """
        Extract POIs for a specific category within the buffered boundary.

        Parameters
        ----------
        category : str
            Category name
        tags : dict
            OSM tags for the category
        buffered_boundary : geopandas.GeoDataFrame
            Buffered boundary

        Returns
        -------
        geopandas.GeoDataFrame or None
            GeoDataFrame with POIs or None if no POIs found
        """
        try:
            # Get the polygon geometry from the buffered boundary
            polygon = buffered_boundary.iloc[0].geometry

            # Construct the tags query
            tags_list = []
            for key, value in tags.items():
                tags_list.append(f'"{key}"="{value}"')

            # tags_query = " or ".join(tags_list)

            # Fetch POIs with timeout
            self.log(f"Fetching {category} POIs...", level="info")
            pois = self._fetch_features_with_timeout(
                lambda: ox.features_from_polygon(polygon, tags=tags)
            )

            if pois is None or len(pois) == 0:
                self.log(f"No {category} POIs found.", level="warning")
                return None

            # Reset index to avoid issues with duplicate indices
            pois = pois.reset_index(drop=True)

            # Keep only point geometries
            pois = pois[pois.geometry.type == "Point"].copy()

            if len(pois) > 0:
                self.log(f"Found {len(pois)} {category} POIs", level="info")

                # Create custom ID field based on index to avoid duplicate ID issues
                # This replaces any existing 'id' column
                pois["id"] = [f"{category}_{i}" for i in range(len(pois))]

                # Handle essential columns
                essential_cols = ["geometry", "name", "osmid"]
                for col in essential_cols:
                    if col not in pois.columns and col != "geometry":
                        pois[col] = None

                # Keep only essential columns
                keep_cols = ["geometry", "name", "id", "osmid"]
                keep_cols = [col for col in keep_cols if col in pois.columns]

                # Subset the dataframe
                pois = pois[keep_cols].copy()

                return pois
            else:
                return None

        except Exception as e:
            self.log(f"Error extracting {category} POIs: {str(e)}", level="warning")
            return None

    def create_map(self, poi_df=None):
        """
        Create a Folium map visualizing the region and POIs.

        Parameters
        ----------
        poi_df : pandas.DataFrame, optional
            DataFrame with POIs to visualize

        Returns
        -------
        folium.Map or None
            Folium map object or None on error
        """
        try:
            # Get boundaries if not already fetched
            boundary, buffered_boundary = self.get_boundaries()

            # Get centroid for map center - project first for accurate calculation
            projected_boundary = boundary.to_crs("EPSG:3857")
            projected_centroid = projected_boundary.geometry.centroid
            # Project back to WGS84 for mapping
            centroid_wgs84 = gpd.GeoDataFrame(
                geometry=projected_centroid, crs="EPSG:3857"
            ).to_crs("EPSG:4326")
            centroid_point = centroid_wgs84.geometry.iloc[0]
            map_center = [centroid_point.y, centroid_point.x]

            # Create map centered on the region
            m = folium.Map(location=map_center, zoom_start=13)

            # Make sure boundary is in WGS84 for Folium
            boundary_wgs84 = (
                boundary.to_crs("EPSG:4326")
                if boundary.crs != "EPSG:4326"
                else boundary
            )
            buffered_wgs84 = (
                buffered_boundary.to_crs("EPSG:4326")
                if buffered_boundary.crs != "EPSG:4326"
                else buffered_boundary
            )

            # Add boundary
            folium.GeoJson(
                boundary_wgs84.__geo_interface__,
                name="Region Boundary",
                style_function=lambda x: {
                    "fillColor": "blue",
                    "color": "blue",
                    "weight": 2,
                    "fillOpacity": 0.1,
                },
            ).add_to(m)

            # Add buffered boundary
            folium.GeoJson(
                buffered_wgs84.__geo_interface__,
                name="Buffered Boundary",
                style_function=lambda x: {
                    "fillColor": "red",
                    "color": "red",
                    "weight": 1,
                    "fillOpacity": 0.05,
                },
            ).add_to(m)

            # Add POIs if provided
            if poi_df is not None and len(poi_df) > 0:
                # Create a feature group for POIs
                poi_group = folium.FeatureGroup(name="POIs")

                # Add each POI as a circle marker
                for _, poi in poi_df.iterrows():
                    if pd.notna(poi.latitude) and pd.notna(poi.longitude):
                        # Get name and category for popup
                        name = poi.get("name", "Unnamed")
                        category = poi.get("category", "Unknown")

                        # Create popup content
                        popup_content = f"<b>{name}</b><br>Type: {category}"

                        # Add marker
                        folium.CircleMarker(
                            location=[poi.latitude, poi.longitude],
                            radius=4,
                            color="green",
                            fill=True,
                            fill_color="green",
                            fill_opacity=0.7,
                            popup=popup_content,
                        ).add_to(poi_group)

                poi_group.add_to(m)

            # Add layer control
            folium.LayerControl().add_to(m)

            return m

        except Exception as e:
            self.log(f"Error creating map: {str(e)}", level="error")
            return None

    def _fetch_features_with_timeout(self, fetch_function):
        """
        Execute a fetch function with a timeout.

        Parameters
        ----------
        fetch_function : callable
            Function to execute with timeout

        Returns
        -------
        object or None
            Object returned by the fetch function or None on timeout/error
        """
        try:
            # Use the with_timeout decorator
            timeout_func = with_timeout(self.fetch_timeout)(fetch_function)
            result = timeout_func()
            return result
        except TimeoutException:
            self.log(
                f"Fetch operation timed out after {self.fetch_timeout} seconds",
                level="error",
            )
            return None
        except Exception as e:
            self.log(f"Error in fetch operation: {str(e)}", level="error")
            return None

    def clear_osm_cache(self):
        """
        Clear the OSM cache to ensure fresh data.

        Returns
        -------
        bool
            True if cache was cleared successfully, False otherwise
        """
        try:
            if hasattr(ox, "settings"):
                ox.settings.use_cache = False
                ox.settings.use_cache = True
                self.log("OSM cache cleared", level="info")
                return True
            else:
                self.log(
                    "Could not clear OSM cache (OSMNX API may have changed)",
                    level="warning",
                )
                return False
        except Exception as e:
            self.log(f"Error clearing OSM cache: {str(e)}", level="error")
            return False

    @classmethod
    def from_file(
        cls,
        file_path,
        region_name=None,
        buffer_distance=500,
        verbose=True,
    ):
        """
        Create an extractor and load data from an existing file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file with POI data
        region_name : str, optional
            Region name to use (or derived from filename)
        buffer_distance : int, optional
            Buffer distance in meters
        verbose : bool, optional
            Whether to print informational messages

        Returns
        -------
        tuple
            (DataExtractor instance, DataFrame with POI data)

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file cannot be read as a CSV or has invalid data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"POI data file not found: {file_path}")

        # Determine region name from file if not provided
        if region_name is None:
            # Try to extract from filename (e.g., "Porto_dataset.csv" -> "Porto")
            basename = os.path.basename(file_path)
            name_parts = basename.split("_")[0]
            region_name = f"{name_parts}, Unknown"

        # Create extractor
        extractor = cls(
            region=region_name,
            buffer_distance=buffer_distance,
            verbose=verbose,
        )

        # Load data
        try:
            extractor.log(f"Loading POI data from: {file_path}", level="info")
            poi_df = pd.read_csv(file_path)

            # Basic validation of loaded data
            if poi_df.empty:
                raise ValueError(f"File contains no data: {file_path}")

            required_cols = ["latitude", "longitude", "category"]
            missing = [col for col in required_cols if col not in poi_df.columns]
            if missing:
                extractor.log(
                    f"Warning: Missing columns in data file: {missing}", level="warning"
                )

            extractor.log(f"Loaded {len(poi_df)} POIs from file", level="info")
            return extractor, poi_df

        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {str(e)}")
