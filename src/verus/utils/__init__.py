"""
verus.utils package

This package contains utility modules and helper functions used across the VERUS project.
It provides classes and functions for logging, timing operations, and other common tasks
that support the main functionality of the project.
"""

# Import main classes to make them available directly from the package
from .logger import Logger
from .timer import TimeoutException, Timer, with_timeout

# Define what symbols are exported when using "from verus.utils import *"
__all__ = ["Logger", "TimeoutException", "Timer", "with_timeout"]

# Import version info from main package
try:
    from .. import __version__
except ImportError:
    __version__ = "unknown"  # Only used when imported standalone
