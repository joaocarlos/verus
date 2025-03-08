"""
VERUS - Visualization and Extraction for Risk and Urban Safety

A package for spatial data extraction, clustering, and risk analysis.
"""

# Package metadata - defined ONCE for the whole project
__version__ = "0.1.0"  # Follow semantic versioning
__author__ = "João Carlos N. Bittencourt"
__team__ = "Laboratory of Emerging Smart Systems"
__email__ = "joaocarlos@ufrb.edu.br"

# You can also expose important classes at the top level if needed
from .clustering import GeOPTICS, KMeansHaversine
from .data import DataExtractor
from .grid import HexagonGridGenerator

# Define public API
__all__ = [
    "KMeansHaversine",
    "GeOPTICS",
    "DataExtractor",
    "HexagonGridGenerator",
]
