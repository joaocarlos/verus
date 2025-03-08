import math
import os

import folium
import geopandas as gpd
import numpy as np
import shapely
import shapely.geometry
from osmnx import geocoder
from pyproj import Transformer
from shapely.geometry import Polygon

from verus.utils.logger import Logger


class HexagonGridGenerator(Logger):
    def __init__(
        self,
        region="Porto, Portugal",
        edge_length=250,
        output_dir=None,  # Add output_dir parameter
        verbose=True,
    ):
        """
        Initialize the HexagonGridGenerator with input validation

        Args:
            region (str): Region to generate hexagon grid for (e.g. "Porto, Portugal")
            edge_length (int): Length of each hexagon edge in meters
            output_dir (str, optional): Base directory for output files
            verbose (bool): Whether to print informational messages
        """
        # Initialize the Logger
        super().__init__(verbose=verbose)

        if not isinstance(region, str) or not region.strip():
            raise ValueError("Region must be a non-empty string")

        if not isinstance(edge_length, (int, float)) or edge_length <= 0:
            raise ValueError("Edge length must be a positive number")

        self.region = region
        self.edge_length = edge_length
        self.place_name = region.split(",")[0].strip()

        # Setup output directories
        # Convert output_dir to absolute path to ensure consistency
        if output_dir:
            self.base_dir = os.path.abspath(output_dir)
        else:
            # Default to a data directory in the project root
            # Get the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to project root and then to data subdirectory
            self.base_dir = os.path.abspath(
                os.path.join(current_dir, "..", "..", "data")
            )

        self.log(f"Using base directory: {self.base_dir}")

        # Create absolute paths for all directories
        self.geojson_dir = os.path.join(self.base_dir, "geojson")
        self.maps_dir = os.path.join(self.base_dir, "maps")

        # Create place-specific subdirectories
        if self.place_name:
            self.geojson_place_dir = os.path.join(self.geojson_dir, self.place_name)
        else:
            self.geojson_place_dir = self.geojson_dir

        # Create all necessary directories
        try:
            for directory in [self.geojson_dir, self.geojson_place_dir, self.maps_dir]:
                os.makedirs(directory, exist_ok=True)
                self.log(f"Ensured directory exists: {directory}")
        except PermissionError as e:
            raise PermissionError(f"No permission to create directories: {str(e)}")
        except Exception as e:
            self.log(f"Warning: Issue creating directories: {str(e)}", "warning")

    def generate_hex_grid(self, bbox, edge_length=None):
        """
        Generates a flat-topped hexagonal grid within a bounding box.

        Args:
            bbox (tuple): Bounding box as (minx, miny, maxx, maxy) in EPSG:4326
            edge_length (float, optional): Length of each hexagon edge in meters.
                                          If None, uses the instance's edge_length.

        Returns:
            GeoDataFrame: GeoDataFrame containing the hexagonal grid
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

            return hex_grid

        except Exception as e:
            self.log(f"Failed to generate hexagonal grid: {e}", "error")
            raise ValueError(f"Failed to generate hexagonal grid: {e}")

    def assign_random_values(self, hex_grid, seed=None, min_val=0, max_val=1):
        """
        Assign random values to the hexagonal grid

        Args:
            hex_grid (GeoDataFrame): Hexagonal grid
            seed (int, optional): Random seed for reproducibility
            min_val (float): Minimum value
            max_val (float): Maximum value

        Returns:
            GeoDataFrame: Grid with random values assigned
        """
        try:
            if seed is not None:
                np.random.seed(seed)

            self.log(f"Assigning random values between {min_val} and {max_val}")
            hex_grid["value"] = np.random.uniform(min_val, max_val, size=len(hex_grid))

            return hex_grid

        except Exception as e:
            self.log(f"Failed to assign random values: {e}", "error")
            raise ValueError(f"Failed to assign random values: {e}")

    def assign_colors(self, hex_grid, color_scale=None):
        """
        Assign colors to hexagons based on their values

        Args:
            hex_grid (GeoDataFrame): Hexagonal grid with 'value' column
            color_scale (callable, optional): Color mapping function

        Returns:
            GeoDataFrame: Grid with color values assigned
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
            hex_grid["color"] = hex_grid["value"].apply(color_scale)

            return hex_grid

        except Exception as e:
            self.log(f"Failed to assign colors: {e}", "warning")
            # Return original grid without colors
            return hex_grid

    def clip_to_region(self, hex_grid, area_gdf=None):
        """
        Clip the hexagonal grid to the boundary of the region

        Args:
            hex_grid (GeoDataFrame): Hexagonal grid
            area_gdf (GeoDataFrame, optional): Area to clip to. If None, uses the region.

        Returns:
            GeoDataFrame: Clipped hexagonal grid
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
            self.log(f"Failed to clip grid to region: {e}", "error")
            return hex_grid  # Return original grid if clipping fails

    def save_to_geojson(self, gdf, filename=None, subfolder=None):
        """
        Save GeoDataFrame to a GeoJSON file

        Args:
            gdf (GeoDataFrame): GeoDataFrame to save
            filename (str, optional): Filename. If None, uses default naming.
            subfolder (str, optional): Subfolder within geojson_dir

        Returns:
            str: Path to saved file
        """
        if gdf is None or len(gdf) == 0:
            self.log("Cannot save empty geodataframe", "warning")
            return None

        try:
            # Determine filename
            if filename is None:
                filename = f"{self.place_name}_hex_grid.geojson"

            # Ensure it has .geojson extension
            if not filename.endswith(".geojson"):
                filename += ".geojson"

            # Determine the save directory
            if subfolder:
                # If subfolder is provided, create it under the place-specific directory
                save_dir = os.path.join(self.geojson_place_dir, subfolder)
            else:
                # Otherwise use the place-specific directory
                save_dir = self.geojson_place_dir

            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Create full output path
            output_file = os.path.join(save_dir, filename)

            # Save the file
            gdf.to_file(output_file, driver="GeoJSON")
            self.log(f"Saved {len(gdf)} hexagons to '{output_file}'", "success")
            return output_file

        except Exception as e:
            self.log(f"Failed to save GeoJSON file: {e}", "error")
            return None

    def create_map(self, hex_grid, area_gdf=None):
        """
        Create an interactive folium map with the hexagonal grid

        Args:
            hex_grid (GeoDataFrame): Hexagonal grid
            area_gdf (GeoDataFrame, optional): Area boundary

        Returns:
            folium.Map: Interactive map
        """
        try:
            if hex_grid is None or len(hex_grid) == 0:
                self.log("No data available to create map", "warning")
                return None

            # Get area if not provided
            if area_gdf is None:
                area_gdf = geocoder.geocode_to_gdf(self.region)

            # Get center of the map using proper projection for accurate centroid calculation
            # First convert to a projected CRS
            area_projected = area_gdf.to_crs(epsg=3857)
            # Calculate centroid in projected coordinates
            centroid_projected = area_projected.geometry.centroid
            # Convert back to WGS84
            centroid_wgs84 = centroid_projected.to_crs(epsg=4326)
            center_lat = centroid_wgs84.y.iloc[0]
            center_lon = centroid_wgs84.x.iloc[0]

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
            self.log(f"Error creating map: {e}", "error")
            return None

    def run(self, save_output=False, add_random_values=True, clip=True):
        """
        Run the hexagonal grid generation process

        Args:
            save_output (bool): Whether to save outputs to files
            add_random_values (bool): Whether to add random values to hexagons
            clip (bool): Whether to clip to region boundary

        Returns:
            GeoDataFrame: Generated hexagonal grid
        """
        try:
            # Get the area GeoDataFrame
            self.log(f"Processing region: {self.region}")
            area_gdf = geocoder.geocode_to_gdf(self.region)

            # Save boundary if requested
            if save_output:
                boundary_path = os.path.join(
                    self.geojson_place_dir, f"{self.place_name}_boundaries.geojson"
                )
                # Ensure directory exists
                os.makedirs(os.path.dirname(boundary_path), exist_ok=True)
                area_gdf.to_file(boundary_path, driver="GeoJSON")
                self.log(f"Saved boundary to {boundary_path}")

            # Get bounding box
            bounding_box = area_gdf.bounds.iloc[0]  # minx, miny, maxx, maxy

            # Generate the hexagonal grid
            hex_grid = self.generate_hex_grid(bounding_box, self.edge_length)

            # Add random values if requested
            if add_random_values:
                hex_grid = self.assign_random_values(hex_grid, seed=42)
                hex_grid = self.assign_colors(hex_grid)

            # Save raw grid if requested
            if save_output:
                self.save_to_geojson(
                    hex_grid, f"{self.place_name}_hex_grid_raw.geojson"
                )

            # Clip to region if requested
            if clip:
                hex_grid = self.clip_to_region(hex_grid, area_gdf)

                # Save clipped grid if requested
                if save_output:
                    self.save_to_geojson(
                        hex_grid, f"{self.place_name}_hex_grid_clipped.geojson"
                    )

            # Create and save map if requested
            if save_output:
                map_obj = self.create_map(hex_grid, area_gdf)
                if map_obj:
                    # Save map to the maps subdirectory using the pre-defined maps_dir
                    # This ensures we use the absolute path
                    map_path = os.path.join(
                        self.maps_dir, f"{self.place_name}_hex_grid_map.html"
                    )
                    map_obj.save(map_path)
                    self.log(f"Saved map to {map_path}")

            return hex_grid

        except Exception as e:
            self.log(f"Error in hexagon grid generation: {e}", "error")
            return None
