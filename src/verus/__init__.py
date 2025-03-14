"""
VERUS - Vulnerability Evaluation for Resilient Urban Systems

A comprehensive toolkit for urban vulnerability assessment through spatial analysis.

Core capabilities:
- Extract urban Points of Interest (POIs) from OpenStreetMap
- Apply time-based vulnerability indexing for different scenarios
- Identify meaningful POI clusters through advanced spatial clustering
- Calculate vulnerability using configurable distance metrics
- Apply spatial smoothing techniques to improve continuity
- Visualize vulnerability zones with interactive maps

Components:
- VERUS: Main vulnerability assessment class
- DataExtractor: POI extraction from OpenStreetMap
- TimeWindowGenerator: Time-based vulnerability indexing
- GeOPTICS: Density-based spatial clustering
- KMeansHaversine: Distance-aware K-means clustering
- HexagonGridGenerator: Spatial grid generation
"""

# Package metadata
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0.dev0"  # Default during development

__author__ = "João Carlos N. Bittencourt"
__team__ = "Laboratory of Emerging Smart Systems"
__email__ = "joaocarlos@ufrb.edu.br"

# Expose important classes at the top level
from .clustering import GeOPTICS, KMeansHaversine
from .data import DataExtractor
from .grid import HexagonGridGenerator
from .verus import VERUS

# Define public API
__all__ = [
    "VERUS",
    "DataExtractor",
    "GeOPTICS",
    "KMeansHaversine",
    "HexagonGridGenerator",
]
