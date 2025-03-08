"""
verus.data package

This package contains modules for data extraction and processing.
It provides classes for fetching points of interest (POIs) from
OpenStreetMap and preparing datasets for analysis.
"""

# Import main classes to make them available directly from the package
from .extraction import DataExtractor
from .timewindow import TimeWindowGenerator

# Define what symbols are exported when using "from verus.data import *"
__all__ = ["DataExtractor", "TimeWindowGenerator"]

# Import version info from main package
try:
    from .. import __version__
except ImportError:
    __version__ = "unknown"  # Only used when imported standalone
