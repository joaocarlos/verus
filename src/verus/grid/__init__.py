"""
verus.grid package

This package contains modules for spatial grid generation and manipulation.
It provides classes for creating hexagonal grids and other spatial tessellations
for geospatial analysis and data visualization.
"""

# Import main classes to make them available directly from the package
from .hexagon import HexagonGridGenerator

# Define what symbols are exported when using "from verus.grid import *"
__all__ = ["HexagonGridGenerator"]

# Import version info from main package
try:
    from .. import __version__
except ImportError:
    __version__ = "unknown"  # Only used when imported standalone
