"""
Hexagon grid generation module for the Verus project.

This module provides functionality to generate and manage hexagonal grids
for spatial analysis of Points of Interest (POIs).
"""

import math
import os

import folium
import geopandas as gpd
import numpy as np
import shapely
from osmnx import geocoder
from pyproj import Transformer
from shapely.geometry import Polygon

from verus.utils.logger import Logger


class HexagonGridGenerator(Logger):
    """
    Generate and manage hexagonal grids for spatial analysis.

    This class provides methods for creating hexagonal grids over geographic
    regions, assigning values to cells, and visualizing the results.
    Hexagonal grids are represented as GeoDataFrames and can be optionally
    saved to disk.

    Attributes:
        region (str): The region name for grid generation.
        edge_length (float): Length of each hexagon edge in meters.
        place_name (str): Simplified region name for file naming.

    Examples:
        >>> generator = HexagonGridGenerator(region="Porto, Portugal")
        >>> grid = generator.run()
        >>> generator.save_to_geojson(grid)
    """

    def __init__(
        self,
        region="Porto, Portugal",
        edge_length=250,
        verbose=True,
    ):
        """
        Initialize the HexagonGridGenerator.

        Parameters
        ----------
        region : str
            Region to generate hexagon grid for (e.g. "Porto, Portugal")
        edge_length : int or float
            Length of each hexagon edge in meters
        verbose : bool
            Whether to print log messages

        Raises
        ------
        ValueError
            If region is not a valid string or edge_length is not positive
        """
        # Initialize the Logger
        super().__init__(verbose=verbose)

        # Validate inputs
        if not isinstance(region, str) or not region.strip():
            raise ValueError("Region must be a non-empty string")

        if not isinstance(edge_length, (int, float)) or edge_length <= 0:
            raise ValueError("Edge length must be a positive number")

        self.region = region
        self.edge_length = edge_length
        self.place_name = region.split(",")[0].strip()

        self.log(
            f"Initialized grid generator for {self.region} with {edge_length}m edge length"
        )

    def generate_hex_grid(self, bbox, edge_length=None):
        """
        Generate a flat-topped hexagonal grid within a bounding box.

        Parameters
        ----------
        bbox : tuple
            Bounding box as (minx, miny, maxx, maxy) in EPSG:4326
        edge_length : float, optional
            Length of each hexagon edge in meters.
            If None, uses the instance's edge_length.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing the hexagonal grid

        Raises
        ------
        ValueError
            If grid generation fails
        """
        if edge_length is None:
            edge_length = self.edge_length

        self.log(f"Generating hexagonal grid with edge length {edge_length}m")

        try:
            # Define transformers for coordinate projection
            transformer_to_utm = Transformer.from_crs(
                "EPSG:4326", "EPSG:3857", always_xy=True
            )
            transformer_to_wgs84 = Transformer.from_crs(
                "EPSG:3857", "EPSG:4326", always_xy=True
            )

            # Project bounding box to planar CRS
            minx, miny, maxx, maxy = bbox
            minx_proj, miny_proj = transformer_to_utm.transform(minx, miny)
            maxx_proj, maxy_proj = transformer_to_utm.transform(maxx, maxy)

            # Flat-topped hexagon dimensions
            dx = math.sqrt(3) * edge_length  # Horizontal spacing
            dy = 3 / 2 * edge_length  # Vertical spacing

            hexagons = []
            x_start = minx_proj
            y = miny_proj
            row = 0

            # Generate hex grid
            while y < maxy_proj + dy:
                x = x_start + (row % 2) * (dx / 2)  # Offset alternate rows
                while x < maxx_proj + dx:
                    # Generate flat-topped hexagon centered at (x, y)
                    vertices = [
                        (
                            x + edge_length * math.cos(angle),
                            y + edge_length * math.sin(angle),
                        )
                        for angle in np.linspace(
                            math.pi / 6, 2 * math.pi + math.pi / 6, 7
                        )
                    ]
                    hexagon = Polygon(vertices)
                    hexagons.append(hexagon)
                    x += dx
                y += dy
                row += 1

            self.log(f"Generated {len(hexagons)} hexagons")

            # Reproject hexagons back to geographic CRS
            hexagons_wgs84 = [
                shapely.ops.transform(transformer_to_wgs84.transform, hex)
                for hex in hexagons
            ]

            # Create GeoDataFrame
            hex_grid = gpd.GeoDataFrame({"geometry": hexagons_wgs84}, crs="EPSG:4326")
            hex_grid["hex_id"] = [f"h_{i}" for i in range(len(hex_grid))]

            return hex_grid

        except Exception as e:
            self.log(f"Failed to generate hexagonal grid: {str(e)}", "error")
            raise ValueError(f"Failed to generate hexagonal grid: {str(e)}")

    def assign_random_values(self, hex_grid, seed=None, min_val=0, max_val=1):
        """
        Assign random values to the hexagonal grid.

        Parameters
        ----------
        hex_grid : GeoDataFrame
            Hexagonal grid
        seed : int, optional
            Random seed for reproducibility
        min_val : float
            Minimum value
        max_val : float
            Maximum value

        Returns
        -------
        GeoDataFrame
            Grid with random values assigned
        """
        try:
            if seed is not None:
                np.random.seed(seed)

            self.log(f"Assigning random values between {min_val} and {max_val}")
            hex_grid = hex_grid.copy()
            hex_grid["value"] = np.random.uniform(min_val, max_val, size=len(hex_grid))

            return hex_grid

        except Exception as e:
            self.log(f"Failed to assign random values: {str(e)}", "error")
            return hex_grid

    def assign_colors(self, hex_grid, color_scale=None):
        """
        Assign colors to hexagons based on their values.

        Parameters
        ----------
        hex_grid : GeoDataFrame
            Hexagonal grid with 'value' column
        color_scale : callable, optional
            Color mapping function

        Returns
        -------
        GeoDataFrame
            Grid with color values assigned
        """
        try:
            if "value" not in hex_grid.columns:
                self.log("No 'value' column found in grid", "warning")
                return hex_grid

            # Import here to avoid dependency if not using this function
            from branca.colormap import linear

            if color_scale is None:
                color_scale = linear.viridis.scale(
                    hex_grid["value"].min(), hex_grid["value"].max()
                )

            self.log("Assigning colors based on values")
            hex_grid = hex_grid.copy()
            hex_grid["color"] = hex_grid["value"].apply(color_scale)

            return hex_grid

        except Exception as e:
            self.log(f"Failed to assign colors: {str(e)}", "warning")
            # Return original grid without colors
            return hex_grid

    def clip_to_region(self, hex_grid, area_gdf=None):
        """
        Clip the hexagonal grid to the boundary of the region.

        Parameters
        ----------
        hex_grid : GeoDataFrame
            Hexagonal grid
        area_gdf : GeoDataFrame, optional
            Area to clip to. If None, uses the region.

        Returns
        -------
        GeoDataFrame
            Clipped hexagonal grid
        """
        try:
            if area_gdf is None:
                self.log(f"Geocoding region: {self.region}")
                area_gdf = geocoder.geocode_to_gdf(self.region)

            # Ensure both dataframes are in the same CRS
            hex_grid = hex_grid.set_crs("EPSG:4326")
            area_gdf = area_gdf.set_crs("EPSG:4326")

            self.log("Clipping grid to region boundaries")
            hex_grid_clipped = gpd.clip(hex_grid, area_gdf)
            self.log(
                f"Clipped from {len(hex_grid)} to {len(hex_grid_clipped)} hexagons"
            )

            return hex_grid_clipped

        except Exception as e:
            self.log(f"Failed to clip grid to region: {str(e)}", "error")
            return hex_grid  # Return original grid if clipping fails

    def save_to_geojson(self, gdf, path=None):
        """
        Save GeoDataFrame to a GeoJSON file.

        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame to save
        path : str, optional
            Where to save the output file. Can be:
            - A directory path: File will be saved as "{directory}/{place_name}_hex_grid.geojson"
            - A file path with extension: File saved at this exact path
            If None, uses "./data/geojson/{place_name}_hex_grid.geojson"

        Returns
        -------
        str or None
            Path to saved file or None if save failed
        """
        if gdf is None or len(gdf) == 0:
            self.log("Cannot save empty geodataframe", "warning")
            return None

        try:
            # Determine path
            if path is None:
                # Default path
                save_dir = os.path.abspath(os.path.join("./data", "geojson"))
                filename = f"{self.place_name}_hex_grid.geojson"
                output_file = os.path.join(save_dir, filename)
            elif path.endswith(".geojson") or path.endswith(".json"):
                # User specified exact file path
                output_file = os.path.abspath(path)
                save_dir = os.path.dirname(output_file)
            else:
                # User specified directory path
                save_dir = os.path.abspath(path)
                filename = f"{self.place_name}_hex_grid.geojson"
                output_file = os.path.join(save_dir, filename)

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Save the file
            gdf.to_file(output_file, driver="GeoJSON")
            self.log(f"Saved {len(gdf)} hexagons to '{output_file}'", "success")
            return output_file

        except Exception as e:
            self.log(f"Failed to save GeoJSON file: {str(e)}", "error")
            return None

    def create_map(self, hex_grid, area_gdf=None):
        """
        Create an interactive folium map with the hexagonal grid.

        Parameters
        ----------
        hex_grid : GeoDataFrame
            Hexagonal grid
        area_gdf : GeoDataFrame, optional
            Area boundary

        Returns
        -------
        folium.Map or None
            Interactive map object or None on error
        """
        try:
            if hex_grid is None or len(hex_grid) == 0:
                self.log("No data available to create map", "warning")
                return None

            # Get area if not provided
            if area_gdf is None:
                area_gdf = geocoder.geocode_to_gdf(self.region)

            # Alternative implementation for the centroid calculation
            # First convert to a projected CRS for accurate centroid calculation
            area_projected = area_gdf.to_crs(epsg=3857)
            # Get the geometry column
            geom_column = area_projected.geometry
            # Calculate centroid for each geometry in the column
            centroid_series = geom_column.centroid
            # Create a new GeoDataFrame with centroids
            centroid_gdf = gpd.GeoDataFrame(geometry=centroid_series, crs="EPSG:3857")
            # Project back to WGS84
            centroid_wgs84 = centroid_gdf.to_crs("EPSG:4326")
            # Get the first centroid (assuming there's only one area polygon)
            center_point = centroid_wgs84.geometry.iloc[0]
            # Extract coordinates
            center_lat = center_point.y
            center_lon = center_point.x

            self.log("Creating interactive map")
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12.5,
                tiles="cartodbpositron",
            )

            # Add hexagons to the map with styling based on values
            if "color" in hex_grid.columns and "value" in hex_grid.columns:
                # Hexagons with color values
                hexagons = folium.features.GeoJson(
                    hex_grid,
                    style_function=lambda feature: {
                        "fillColor": feature["properties"]["color"],
                        "color": "#000000",
                        "weight": 0.5,
                        "fillOpacity": 0.6,
                    },
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=["value"], aliases=["Value"], localize=True
                    ),
                )
                m.add_child(hexagons)
            else:
                # Basic hexagons without color
                hexagons = folium.features.GeoJson(
                    hex_grid,
                    style_function=lambda feature: {
                        "color": "#000000",
                        "weight": 0.5,
                        "fillOpacity": 0.4,
                        "fillColor": "#3186cc",
                    },
                )
                m.add_child(hexagons)

            # Add area boundary to the map
            boundary = folium.features.GeoJson(
                area_gdf,
                style_function=lambda feature: {
                    "color": "#808080",
                    "weight": 2,
                    "fillOpacity": 0,
                },
            )
            m.add_child(boundary)

            return m

        except Exception as e:
            self.log(f"Error creating map: {str(e)}", "error")
            return None

    def run(self, save_output=False, add_random_values=True, clip=True, path=None):
        """
        Run the hexagonal grid generation process.

        This method handles the complete workflow of:
        1. Getting the region boundary
        2. Generating the grid
        3. Optionally adding values and colors
        4. Optionally clipping to the region
        5. Optionally saving outputs

        Parameters
        ----------
        save_output : bool, optional
            Whether to save outputs to disk
        add_random_values : bool, optional
            Whether to add random values to hexagons
        clip : bool, optional
            Whether to clip the grid to region boundary
        path : str, optional
            Path for saving outputs, passed to save_to_geojson if save_output=True

        Returns
        -------
        GeoDataFrame
            Generated hexagonal grid
        """
        try:
            # Get the area GeoDataFrame
            self.log(f"Processing region: {self.region}")
            area_gdf = geocoder.geocode_to_gdf(self.region)

            # Get bounding box
            bounding_box = area_gdf.bounds.iloc[0]  # minx, miny, maxx, maxy

            # Generate the hexagonal grid
            hex_grid = self.generate_hex_grid(bounding_box, self.edge_length)

            # Add random values if requested
            if add_random_values:
                hex_grid = self.assign_random_values(hex_grid, seed=42)
                hex_grid = self.assign_colors(hex_grid)

            # Clip to region if requested
            if clip:
                hex_grid = self.clip_to_region(hex_grid, area_gdf)

            # Save outputs if requested
            if save_output:
                self.save_to_geojson(hex_grid, path=path)

                # Create and save map
                map_obj = self.create_map(hex_grid, area_gdf)
                if map_obj:
                    # Determine map path
                    if path is None:
                        maps_dir = os.path.abspath(os.path.join("./data", "maps"))
                    elif (
                        os.path.isfile(path)
                        or path.endswith(".geojson")
                        or path.endswith(".json")
                    ):
                        maps_dir = os.path.dirname(os.path.abspath(path))
                    else:
                        maps_dir = os.path.abspath(os.path.join(path, "maps"))

                    os.makedirs(maps_dir, exist_ok=True)
                    map_path = os.path.join(
                        maps_dir, f"{self.place_name}_hex_grid_map.html"
                    )
                    map_obj.save(map_path)
                    self.log(f"Saved map to {map_path}")

            return hex_grid

        except Exception as e:
            self.log(f"Error in hexagon grid generation: {str(e)}", "error")
            return None

    @classmethod
    def from_file(cls, file_path, verbose=True):
        """
        Create a generator and load grid from an existing file.

        Parameters
        ----------
        file_path : str
            Path to the GeoJSON file with grid data
        verbose : bool, optional
            Whether to print log messages

        Returns
        -------
        tuple
            (HexagonGridGenerator instance, GeoDataFrame with grid)

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file cannot be read as a GeoJSON
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Grid file not found: {file_path}")

        # Create a logger for initial messages before the class is instantiated
        temp_logger = Logger(name="HexagonGrid_Loader", verbose=verbose)

        try:
            # Load the grid
            temp_logger.log(f"Loading grid from: {file_path}")
            grid_gdf = gpd.read_file(file_path)

            if grid_gdf.empty:
                raise ValueError(f"File contains no valid geometry: {file_path}")

            # Try to extract region name from filename
            basename = os.path.basename(file_path)
            name_parts = basename.split("_")[0]
            region_name = f"{name_parts}, Unknown"

            # Create generator with default settings
            generator = cls(region=region_name, verbose=verbose)
            generator.log(f"Loaded {len(grid_gdf)} hexagons from file")

            # Ensure CRS is set
            if grid_gdf.crs is None:
                generator.log("Warning: CRS not defined, assuming EPSG:4326", "warning")
                grid_gdf = grid_gdf.set_crs("EPSG:4326")

            return generator, grid_gdf

        except Exception as e:
            raise ValueError(f"Error loading grid from {file_path}: {str(e)}")
