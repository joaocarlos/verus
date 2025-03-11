"""
verus.clustering package

This package contains modules and functions related to clustering algorithms.
It provides classes for OPTICS and K-means clustering with geospatial adaptations.
"""

# Import main classes to make them available directly from the package
from .kmeans import KMeansHaversine
from .optics import GeOPTICS

# Define what symbols are exported when using "from verus.clustering import *"
__all__ = ["KMeansHaversine", "GeOPTICS"]

# Import version info from main package
try:
    from .. import __version__
except ImportError:
    __version__ = "unknown"  # Only used when imported standalone
