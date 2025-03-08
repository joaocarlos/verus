import os
import time

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd

from verus.utils.logger import Logger
from verus.utils.timer import TimeoutException, with_timeout


class DataExtractor(Logger):
    """
    Extract and process points of interest (POIs) from OpenStreetMap.

    This class fetches POIs from OpenStreetMap based on specified tags,
    processes them into a structured format, and saves them as GeoJSON
    files for further analysis.

    Attributes:
        region (str): The region name to extract data from.
        buffer_distance (float): Buffer distance in meters around the region.
        amenity_tags (dict): Dictionary of OSM tags to extract.

    Examples:
        >>> extractor = DataExtractor(region="Porto, Portugal")
        >>> extractor.extract_all_pois()
    """

    def __init__(
        self,
        region="Porto, Portugal",
        buffer_distance=500,
        amenity_tags=None,
        boundary_file=None,
        fetch_timeout=60,  # Add timeout parameter
        output_dir=None,  # Add output_dir parameter
        verbose=True,
    ):
        """
        Initialize the DataExtractor with input validation

        Args:
            region (str or None): Region to extract data from (e.g. "Porto, Portugal")
            buffer_distance (int): Buffer distance in meters
            amenity_tags (dict, optional): Custom amenity tags to extract
            boundary_file (str, optional): Path to a boundary shapefile or GeoJSON
            fetch_timeout (int): Timeout in seconds for fetching data from OSM
            verbose (bool): Whether to print informational messages
        """
        # Initialize the Logger
        super().__init__(verbose=verbose)

        # Check if at least one of region or boundary_file is provided
        if not region and not boundary_file:
            raise ValueError("Either region or boundary_file must be provided")

        if boundary_file and not os.path.exists(boundary_file):
            raise ValueError(f"Boundary file not found: {boundary_file}")

        if region and not isinstance(region, str):
            raise ValueError("Region must be a string")

        if not isinstance(buffer_distance, (int, float)) or buffer_distance <= 0:
            raise ValueError("Buffer distance must be a positive number")

        if not isinstance(fetch_timeout, int) or fetch_timeout <= 0:
            raise ValueError("Fetch timeout must be a positive integer")

        self.region = region
        self.buffer_distance = buffer_distance
        self.boundary_file = boundary_file
        self.fetch_timeout = fetch_timeout

        # Set place_name based on region or boundary file
        if region:
            self.place_name = region.split(",")[0].strip()
        elif boundary_file:
            self.place_name = os.path.splitext(os.path.basename(boundary_file))[0]

        # Setup output directories

        # Use the provided output_dir or default to current directory
        base_dir = output_dir if output_dir else os.path.curdir

        # Create absolute paths for all directories
        self.datasets_dir = os.path.abspath(os.path.join(base_dir, "poti"))
        self.geojson_dir = os.path.abspath(os.path.join(base_dir, "geojson"))

        # Create place-specific subdirectories
        if self.place_name:
            self.geojson_place_dir = os.path.join(self.geojson_dir, self.place_name)
        else:
            self.geojson_place_dir = self.geojson_dir

        # Create all necessary directories
        try:
            for directory in [
                self.datasets_dir,
                self.geojson_dir,
                self.geojson_place_dir,
            ]:
                os.makedirs(directory, exist_ok=True)
                self.log(f"Ensured directory exists: {directory}")
        except PermissionError as e:
            raise PermissionError(f"No permission to create directories: {str(e)}")
        except Exception as e:
            self.log(f"Warning: Issue creating directories: {str(e)}", "warning")

        # Default amenity tags if none provided
        self.amenity_tags = amenity_tags or {
            "school": {"amenity": "school"},
            "hospital": {"amenity": "hospital"},
            "university": {"amenity": "university"},
            "mall": {"shop": "mall"},
            "attraction": {"tourism": "attraction"},
            "station": {"public_transport": "station"},
            "bus_station": {"amenity": "bus_station"},
            "train_station": {"amenity": "train_station"},
            "metro_station": {"amenity": "metro_station"},
            "industrial": {"landuse": "industrial"},
        }

        # Initialize boundary variables
        self.boundary_polygon = None
        self.buffered_boundary = None

        # Configure OSM download settings using current OSMnx API
        # These settings help with potential lag issues
        try:
            # Create cache directory if it doesn't exist
            cache_folder = "./.osm_cache"
            os.makedirs(cache_folder, exist_ok=True)

            # Set up configuration based on OSMnx version
            ox.settings.use_cache = True
            ox.settings.cache_folder = cache_folder
            ox.settings.log_console = False
            ox.settings.timeout = self.fetch_timeout
            ox.settings.max_query_area_size = 50 * 1000 * 1000  # 50 sq km

            self.log("OSMnx settings configured successfully")
        except Exception as e:
            self.log(f"Warning: Could not configure OSMnx settings: {e}", "warning")

    def run(self, save_dataset=False):
        """
        Run the data extraction process with improved error handling

        Args:
            save_dataset (bool): Whether to save the dataset to a CSV file

        Returns:
            pd.DataFrame: DataFrame containing the extracted points of interest
        """
        try:
            # Get the boundary and buffered polygon
            _, buffered_polygon = self.get_boundaries()

            # Extract amenities with better error handling and timeouts
            gdfs = []
            total_categories = len(self.amenity_tags)
            processed = 0

            self.log(f"Extracting amenities for {total_categories} categories")

            for category, tags in self.amenity_tags.items():
                processed += 1
                self.log(
                    f"Processing category {processed}/{total_categories}: {category}"
                )

                gdf = self._fetch_features_with_timeout(
                    buffered_polygon, tags, category
                )

                if not gdf.empty:
                    gdf["category"] = category
                    gdfs.append(gdf)
                    self.log(f"Fetched {len(gdf)} '{category}' features.")
                else:
                    self.log(f"No features found for '{category}'.")

            if not gdfs:
                self.log("No data retrieved for any category.", "warning")
                return None

            # Geo-code the region
            try:
                self.log(f"Geocoding region: {self.region}")
                place_gdf = ox.geocode_to_gdf(self.region)
            except AttributeError:
                # For newer versions of OSMnx
                try:
                    place_gdf = ox.geocoder.geocode_to_gdf(self.region)
                except Exception as e:
                    self.log(f"Failed to geocode region '{self.region}': {e}", "error")
                    raise ValueError(f"Failed to geocode region '{self.region}': {e}")
            except Exception as e:
                self.log(f"Failed to geocode region '{self.region}': {e}", "error")
                raise ValueError(f"Failed to geocode region '{self.region}': {e}")

            # Project the geometry to a metric CRS for accurate buffering
            try:
                self.log("Creating buffered polygon")
                place_gdf = place_gdf.to_crs(epsg=3857)
                buffered_polygon = place_gdf.buffer(self.buffer_distance).union_all()
                buffered_polygon = (
                    gpd.GeoSeries([buffered_polygon], crs="EPSG:3857")
                    .to_crs(epsg=4326)
                    .iloc[0]
                )
            except Exception as e:
                self.log(f"Failed to create buffered polygon: {e}", "error")
                raise ValueError(f"Failed to create buffered polygon: {e}")

            # Save the boundary GeoJSON
            boundary_path = f"{self.geojson_dir}/{self.place_name}_boundaries.geojson"
            try:
                self.log(f"Saving boundary GeoJSON to {boundary_path}")
                boundary_geojson = place_gdf.to_json()
                if not os.path.exists(self.geojson_dir):
                    os.makedirs(self.geojson_dir, exist_ok=True)
                with open(boundary_path, "w") as f:
                    f.write(boundary_geojson)
            except Exception as e:
                self.log(f"Failed to save boundary GeoJSON: {e}", "warning")
                # Continue execution even if writing fails

            # Extract amenities
            gdfs = []
            self.log(f"Extracting amenities for {len(self.amenity_tags)} categories")
            for category, tags in self.amenity_tags.items():
                try:
                    gdf = ox.features_from_polygon(buffered_polygon, tags=tags)
                    if not gdf.empty:
                        gdf["category"] = category
                        gdfs.append(gdf)
                        self.log(f"Fetched {len(gdf)} '{category}' features.")
                    else:
                        self.log(f"No features found for '{category}'.")
                except Exception as e:
                    self.log(f"Error fetching '{category}': {e}", "warning")
                    # Continue with other categories

            if not gdfs:
                self.log("No data retrieved for any category.", "warning")
                return None

            try:
                self.log("Combining data from all categories")
                combined_gdf = pd.concat(gdfs, ignore_index=True)
            except Exception as e:
                self.log(f"Failed to concatenate geodataframes: {e}", "error")
                raise ValueError(f"Failed to concatenate geodataframes: {e}")

            try:
                self.log("Processing coordinate transformations")
                # Convert to a local projected CRS for accurate centroid calculation
                combined_gdf_proj = combined_gdf.to_crs(epsg=3763)
                combined_gdf_proj["centroid"] = combined_gdf_proj.geometry.centroid

                # Convert back to WGS84 for lat/lon coordinates
                combined_gdf = combined_gdf_proj.to_crs(epsg=4326)
                combined_gdf["centroid"] = combined_gdf_proj["centroid"].to_crs(
                    epsg=4326
                )
                combined_gdf["latitude"] = combined_gdf["centroid"].y
                combined_gdf["longitude"] = combined_gdf["centroid"].x
                combined_gdf.drop(columns=["centroid"], inplace=True)
            except Exception as e:
                self.log(f"Failed to process coordinate transformations: {e}", "error")
                raise ValueError(f"Failed to process coordinate transformations: {e}")

            # Ensure 'name' column exists
            if "name" not in combined_gdf.columns:
                combined_gdf["name"] = None

            # Select relevant columns and remove duplicates
            try:
                self.log("Processing final dataset")
                poti_df = combined_gdf[["latitude", "longitude", "category", "name"]]
                poti_df = poti_df.dropna(subset=["latitude", "longitude"])
                poti_df = poti_df.drop_duplicates(
                    subset=["latitude", "longitude"], keep="last"
                ).reset_index(drop=True)
                poti_df = poti_df.drop_duplicates(
                    subset=["name"], keep="last"
                ).reset_index(drop=True)
                poti_df = poti_df.dropna(subset=["name", "category"]).reset_index(
                    drop=True
                )
            except Exception as e:
                self.log(f"Failed to process dataframe: {e}", "error")
                raise ValueError(f"Failed to process dataframe: {e}")

            # Save to CSV if requested
            if save_dataset:
                self.save_dataset(poti_df)

            return poti_df

        except Exception as e:
            self.log(f"Error in extraction process: {e}", "error")
            return None

    def clear_osm_cache(self):
        """
        Clear the OSM cache to resolve potential issues with stale data

        Returns:
            bool: True if cache was cleared successfully, False otherwise
        """
        import shutil

        cache_dir = "./.osm_cache"
        try:
            if os.path.exists(cache_dir):
                self.log(f"Clearing OSM cache at {cache_dir}")
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                self.log("OSM cache cleared successfully", "success")
                return True
            else:
                self.log("No OSM cache directory found", "info")
                return False
        except Exception as e:
            self.log(f"Failed to clear OSM cache: {e}", "error")
            return False

    def _fetch_features_with_timeout(
        self, polygon, tags, category, max_retries=3, backoff_factor=2
    ):
        """
        Fetch features from OSM with timeout and retry mechanism.
        Will not retry if no features are found (empty result).

        Args:
            polygon: The polygon to extract features from
            tags: OSM tags to query
            category: Category name for logging
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor to increase wait time between retries

        Returns:
            GeoDataFrame: Features matching the tags (may be empty)
        """
        retry = 0
        last_error = None

        while retry < max_retries:
            try:
                start_time = time.time()

                # Apply timeout to the query function
                if os.name == "nt":  # Windows doesn't support signal.SIGALRM
                    # On Windows, rely on OSMnx's internal timeout
                    gdf = ox.features.features_from_polygon(polygon, tags=tags)
                else:
                    # On Unix-like systems, use our timeout decorator
                    @with_timeout(self.fetch_timeout)
                    def fetch_with_timeout(poly, t):
                        return ox.features.features_from_polygon(poly, tags=t)

                    gdf = fetch_with_timeout(polygon, tags)

                elapsed = time.time() - start_time

                # Check if the result is empty but valid (no matching features)
                if isinstance(gdf, gpd.GeoDataFrame):
                    if gdf.empty:
                        self.log(
                            f"No matching features found for '{category}' (in {elapsed:.2f}s)"
                        )
                    else:
                        self.log(
                            f"Fetched {len(gdf)} '{category}' features in {elapsed:.2f} seconds"
                        )
                    return gdf
                else:
                    # If we got something that's not a GeoDataFrame, treat as error
                    raise ValueError(f"Expected GeoDataFrame but got {type(gdf)}")

            except TimeoutException:
                retry += 1
                wait_time = backoff_factor**retry
                self.log(
                    f"Timeout fetching '{category}'. Retry {retry}/{max_retries} in {wait_time}s",
                    "warning",
                )
                time.sleep(wait_time)
                last_error = f"Timeout after {self.fetch_timeout} seconds"

            except Exception as e:
                # Check for specific error messages that indicate "no data" rather than a failure
                error_str = str(e).lower()

                # Expanded list of "no data" error patterns
                no_data_patterns = [
                    "no data",
                    "no features",
                    "nothing found",
                    "no matching features",
                    "check query location",
                    "no elements found",
                    "no objects found",
                    "no results",
                    "no amenities",
                    "empty result",
                ]

                # Check if any of these patterns are in the error message
                if any(pattern in error_str for pattern in no_data_patterns):
                    # This is not an error but a valid empty result
                    self.log(f"No matching features for '{category}': {e}")
                    return gpd.GeoDataFrame()  # Return empty GeoDataFrame immediately

                # Otherwise, it's a real error, so retry
                retry += 1
                wait_time = backoff_factor**retry
                self.log(
                    f"Error fetching '{category}': {e}. Retry {retry}/{max_retries} in {wait_time}s",
                    "warning",
                )
                time.sleep(wait_time)
                last_error = str(e)

        # If we get here, all retries failed
        self.log(f"All retries failed for '{category}': {last_error}", "error")
        return gpd.GeoDataFrame()

    def save_dataset(self, df, filename=None):
        """
        Save the dataset to a CSV file

        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str, optional): Custom filename. If None, uses default naming

        Returns:
            str: Path to the saved file
        """
        if df is None or df.empty:
            self.log("Cannot save empty dataset", "warning")
            return None

        try:
            # Determine filename
            if filename is None:
                filename = f"{self.place_name}_dataset_buffered.csv"

            # Ensure it has .csv extension
            if not filename.endswith(".csv"):
                filename += ".csv"

            # Create absolute path for the CSV file
            csv_path = os.path.join(self.datasets_dir, filename)

            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            # Save the file
            df.to_csv(csv_path, index=False)
            self.log(f"Saved {len(df)} points of interest to '{csv_path}'.", "success")
            return csv_path

        except Exception as e:
            self.log(f"Failed to save CSV file: {e}", "error")
            return None

    def get_boundaries(self):
        """
        Get the boundary polygon for the region.

        This function handles different sources of boundaries:
        - Region name (using osmnx geocoding)
        - Boundary file (shapefile, GeoJSON, etc.)

        Returns:
            tuple: (boundary_gdf, buffered_polygon) - GeoDataFrame with the region boundary
                and the buffered polygon geometry
        """
        try:
            boundary_gdf = None

            # Case 1: Use boundary file if provided
            if self.boundary_file:
                self.log(f"Loading boundary from file: {self.boundary_file}")
                try:
                    boundary_gdf = gpd.read_file(self.boundary_file)
                    if boundary_gdf.empty:
                        self.log(
                            f"Boundary file is empty: {self.boundary_file}", "error"
                        )
                        raise ValueError(
                            f"Boundary file is empty: {self.boundary_file}"
                        )

                    # Ensure the CRS is set correctly
                    if boundary_gdf.crs is None:
                        self.log("Boundary file has no CRS, assuming WGS84", "warning")
                        boundary_gdf.set_crs(epsg=4326, inplace=True)
                    elif boundary_gdf.crs != "EPSG:4326":
                        self.log(
                            f"Converting boundary CRS from {boundary_gdf.crs} to EPSG:4326"
                        )
                        boundary_gdf = boundary_gdf.to_crs(epsg=4326)
                except Exception as e:
                    self.log(f"Failed to load boundary file: {e}", "error")
                    raise ValueError(f"Failed to load boundary file: {e}")

            # Case 2: Use region name if no file is provided or if file loading failed
            if boundary_gdf is None and self.region:
                self.log(f"Geocoding region: {self.region}")
                try:
                    try:
                        # Try the newer OSMnx API first
                        boundary_gdf = ox.geocoder.geocode_to_gdf(self.region)
                    except AttributeError:
                        # Fall back to old API if needed
                        boundary_gdf = ox.geocode_to_gdf(self.region)
                except Exception as e:
                    self.log(f"Failed to geocode region '{self.region}': {e}", "error")
                    raise ValueError(f"Failed to geocode region '{self.region}': {e}")

            # If we still don't have a boundary, raise an error
            if boundary_gdf is None:
                raise ValueError(
                    "Could not obtain boundary from either file or region name"
                )

            # Create a buffered polygon for the boundary
            self.log(f"Creating buffered polygon with {self.buffer_distance}m distance")
            try:
                # Project to a metric CRS for accurate buffering
                boundary_gdf_proj = boundary_gdf.to_crs(epsg=3857)

                # Check if we have multiple geometries
                if len(boundary_gdf_proj) > 1:
                    # Dissolve all polygons into one
                    boundary_gdf_proj = boundary_gdf_proj.dissolve()

                # Buffer the geometry
                buffered_polygon = boundary_gdf_proj.buffer(
                    self.buffer_distance
                ).unary_union

                # Convert back to WGS84
                buffered_boundary = (
                    gpd.GeoSeries([buffered_polygon], crs="EPSG:3857")
                    .to_crs(epsg=4326)
                    .iloc[0]
                )

                # Save both to instance variables for reuse
                self.boundary_polygon = boundary_gdf
                self.buffered_boundary = buffered_boundary

                # Save the boundary GeoJSON
                boundary_path = os.path.join(
                    self.geojson_place_dir, f"{self.place_name}_boundaries.geojson"
                )
                try:
                    self.log(f"Saving boundary GeoJSON to {boundary_path}")
                    # Ensure parent directory exists
                    os.makedirs(os.path.dirname(boundary_path), exist_ok=True)

                    # Convert to GeoJSON and save
                    boundary_geojson = boundary_gdf.to_json()
                    with open(boundary_path, "w") as f:
                        f.write(boundary_geojson)

                    # Also save buffered boundary
                    buffered_path = os.path.join(
                        self.geojson_place_dir, f"{self.place_name}_buffered.geojson"
                    )
                    buffered_gdf = gpd.GeoDataFrame(
                        geometry=[buffered_boundary], crs="EPSG:4326"
                    )
                    buffered_gdf.to_file(buffered_path, driver="GeoJSON")
                    self.log(f"Saved buffered boundary to {buffered_path}")
                except Exception as e:
                    self.log(f"Failed to save boundary GeoJSON: {e}", "warning")
                    self.log(f"Attempted path was: {boundary_path}", "warning")

                return boundary_gdf, buffered_boundary

            except Exception as e:
                self.log(f"Failed to create buffered polygon: {e}", "error")
                raise ValueError(f"Failed to create buffered polygon: {e}")

        except Exception as e:
            self.log(f"Error getting boundaries: {e}", "error")
            raise e

    def get_boundary_path(self):
        """
        Get the path to the boundary GeoJSON file.

        Returns:
            str: Path to the boundary GeoJSON file
        """
        boundary_path = os.path.join(
            self.geojson_place_dir, f"{self.place_name}_boundaries.geojson"
        )

        # Check if the file exists
        if os.path.exists(boundary_path):
            return boundary_path

        # If the file doesn't exist, try to generate it
        try:
            self.get_boundaries()
            if os.path.exists(boundary_path):
                return boundary_path
        except Exception as e:
            self.log(f"Error generating boundary: {e}", "warning")

        return None

    def get_buffered_boundary_path(self):
        """
        Get the path to the buffered boundary GeoJSON file.

        Returns:
            str: Path to the buffered boundary GeoJSON file
        """
        buffered_path = os.path.join(
            self.geojson_place_dir, f"{self.place_name}_buffered.geojson"
        )

        # Check if the file exists
        if os.path.exists(buffered_path):
            return buffered_path

        # If the file doesn't exist, try to generate it
        try:
            self.get_boundaries()
            if os.path.exists(buffered_path):
                return buffered_path
        except Exception as e:
            self.log(f"Error generating buffered boundary: {e}", "warning")

        return None

    def create_map(self, poti_df=None):
        """Create a folium map with better error handling"""
        try:
            if poti_df is None:
                self.log("No dataframe provided, running extraction")
                poti_df = self.run()
                if poti_df is None or poti_df.empty:
                    self.log("No data available to create map", "warning")
                    return None
            elif poti_df.empty:
                self.log("Empty dataframe provided, cannot create map", "warning")
                return None

            # Check for required columns
            required_columns = ["latitude", "longitude", "name"]
            missing_columns = [
                col for col in required_columns if col not in poti_df.columns
            ]
            if missing_columns:
                self.log(
                    f"Dataframe missing required columns: {missing_columns}", "error"
                )
                raise ValueError(
                    f"Dataframe missing required columns: {missing_columns}"
                )

            # Create map centered on data points
            try:
                self.log("Creating map")
                m = folium.Map(
                    location=[poti_df["latitude"].mean(), poti_df["longitude"].mean()],
                    zoom_start=12,
                    tiles="cartodbpositron",
                )
            except Exception as e:
                self.log(f"Failed to create map: {e}", "error")
                raise ValueError(f"Failed to create map: {e}")

            # Add the boundary area to the map
            boundary_path = self.get_boundary_path()
            if boundary_path:
                try:
                    self.log("Adding boundary to map")
                    folium.GeoJson(
                        boundary_path,
                        name="boundary",
                        style_function=lambda feature: {
                            "color": "#808080",
                            "weight": 2,
                            "fillOpacity": 0,
                        },
                    ).add_to(m)
                except Exception as e:
                    self.log(f"Failed to add boundary to map: {e}", "warning")

            # Add the buffered boundary
            buffered_path = self.get_buffered_boundary_path()
            if buffered_path:
                try:
                    self.log("Adding buffered boundary to map")
                    folium.GeoJson(
                        buffered_path,
                        name="buffered",
                        style_function=lambda feature: {
                            "color": "#A52A2A",
                            "weight": 2,
                            "fillOpacity": 0,
                        },
                    ).add_to(m)
                except Exception as e:
                    self.log(f"Failed to add buffered boundary to map: {e}", "warning")

            # Add markers for each point
            self.log(f"Adding {len(poti_df)} markers to map")
            for i, row in poti_df.iterrows():
                try:
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=2,
                        stroke=False,
                        color="#A1AEB1",
                        fill=True,
                        fill_color="#616569",
                        fill_opacity=0.8,
                        popup=row.get("name", "No name"),
                    ).add_to(m)
                except Exception as e:
                    self.log(f"Failed to add marker for point {i}: {e}", "warning")

            return m

        except Exception as e:
            self.log(f"Error creating map: {e}", "error")
            return None
