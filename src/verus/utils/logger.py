"""
Logging utility for the Verus project.
This module provides consistent logging functionality across the project.
"""

import logging
import sys
from datetime import datetime

try:
    from colorama import Fore, Style, init

    # Initialize colorama for cross-platform color support
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("For colored output, install colorama: pip install colorama")


def log(message, level="info", verbose=True):
    """
    Print a formatted and colored log message.

    Args:
        message (str): Message to print
        level (str): Log level ('info', 'warning', 'error', 'success')
        verbose (bool): Whether to print informational messages
    """
    # Skip non-critical messages if verbose is False
    if not (verbose or level in ["warning", "error"]):
        return

    # Get current timestamp for the log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Define level prefixes and colors
    level_info = {
        "info": {
            "prefix": "INFO",
            "color": Fore.MAGENTA if COLORS_AVAILABLE else "",
        },
        "warning": {
            "prefix": "WARNING",
            "color": Fore.YELLOW if COLORS_AVAILABLE else "",
        },
        "error": {"prefix": "ERROR", "color": Fore.RED if COLORS_AVAILABLE else ""},
        "success": {
            "prefix": "SUCCESS",
            "color": Fore.GREEN if COLORS_AVAILABLE else "",
        },
    }

    # Get level properties, defaulting to info
    level_props = level_info.get(level, level_info["info"])
    prefix = level_props["prefix"]
    color = level_props["color"]

    # Format the message with colors if available
    if COLORS_AVAILABLE:
        print(f"{Fore.WHITE}{timestamp} {color}[{prefix}]{Style.RESET_ALL} {message}")
    else:
        print(f"{timestamp} [{prefix}] {message}")


class Logger:
    """Enhanced logger for Verus components with configurable output."""

    # Class-wide log levels dictionary shared by all instances
    LEVELS = {
        "debug": {"level": logging.DEBUG, "color": "\033[36m"},  # Cyan
        "info": {
            "level": logging.INFO,
            "color": "\033[35m",
        },  # Purple (changed from default)
        "success": {"level": logging.INFO, "color": "\033[32m"},  # Green
        "warning": {"level": logging.WARNING, "color": "\033[33m"},  # Yellow
        "error": {"level": logging.ERROR, "color": "\033[31m"},  # Red
        "critical": {"level": logging.CRITICAL, "color": "\033[35m"},  # Magenta
    }

    def __init__(self, name=None, verbose=True, log_file=None, level="info"):
        """
        Initialize the logger.

        Args:
            name (str, optional): Logger name. Defaults to caller's class name
            verbose (bool): Whether to print to stdout
            log_file (str, optional): Path to log file
            level (str): Minimum log level ('debug', 'info', 'warning', 'error', 'critical')
        """
        # Set a default name based on the caller if not provided
        if name is None:
            import inspect

            frame = inspect.stack()[1]
            caller_class = frame.frame.f_locals.get("self", None).__class__.__name__
            name = caller_class if caller_class else frame.function

        self.name = name
        self.verbose = verbose

        # Set up Python logger
        self.logger = logging.getLogger(f"verus.{name}")

        # Don't propagate to root logger to avoid duplicate logs
        self.logger.propagate = False

        # Remove all handlers if any exist
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Set level
        min_level = self.LEVELS.get(level.lower(), self.LEVELS["info"])["level"]
        self.logger.setLevel(min_level)

        # Add stdout handler if verbose
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(min_level)

            # Create a custom formatter that includes colors
            class ColorFormatter(logging.Formatter):
                def format(self, record):
                    # Map Python logging levels to our custom levels
                    level_name = record.levelname
                    custom_level = getattr(record, "custom_level", None)

                    # For custom levels like success, change the display level name
                    if custom_level == "success":
                        level_name = "SUCCESS"

                    # Get color for the level
                    color_key = custom_level if custom_level else level_name.lower()
                    color = Logger.LEVELS.get(color_key, Logger.LEVELS["info"])["color"]
                    reset = "\033[0m"

                    # Format with timestamp and colored level name
                    record.levelname_colored = f"{color}{level_name}{reset}"
                    return super().format(record)

            console_formatter = ColorFormatter(
                "%(asctime)s [%(levelname_colored)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if log_file provided
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(min_level)

            # Create a custom formatter without colors
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def log(self, message, level="info"):
        """
        Log a message at the specified level.

        Args:
            message (str): Message to log
            level (str): Log level ('debug', 'info', 'success', 'warning', 'error', 'critical')
        """
        level = level.lower()
        if level not in self.LEVELS:
            level = "info"  # Default to info for unknown levels

        # Get the proper logging method based on the base level
        base_level_name = "info" if level == "success" else level
        log_method = getattr(self.logger, base_level_name)

        # Create extra dict for custom level
        extra = {"custom_level": level}

        # Add color if verbose
        if self.verbose:
            # color = self.LEVELS[level]["color"]
            # reset = "\033[0m"
            # Only color the message if it's not already a custom level
            # (since we color the level name separately)
            formatted_message = message
        else:
            formatted_message = message

        # Log the message with the extra context
        log_method(formatted_message, extra=extra)
