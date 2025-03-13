"""
Timing utilities for the Verus project.
This module provides tools for timing code execution and implementing timeouts.
"""

import os
import signal
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from verus.utils.logger import Logger

# Type variables for generic function typing
T = TypeVar("T", bound=Callable[..., Any])


class TimeoutException(Exception):
    """Exception raised when a function execution times out."""

    pass


class Timer(Logger):
    """
    Context manager for timing code execution with logging capabilities.

    Example:
        with Timer("My operation"):
            time.sleep(1)
            # Will log "My operation took 1.00 seconds"
    """

    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """
        Initialize the timer with optional logging.

        Args:
            name (str, optional): Name of the operation being timed
            verbose (bool, optional): Whether to log the timing results. Defaults to True.
        """
        super().__init__(name=name or "Timer", verbose=verbose)
        self.operation_name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        """Start timing when entering the context."""
        self.start_time = time.time()
        if self.operation_name:
            self.log(f"Starting operation: {self.operation_name}", "info")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Log elapsed time when exiting the context."""
        self.elapsed = time.time() - self.start_time

        if self.operation_name:
            self.log(
                f"{self.operation_name} took {self.elapsed:.2f} seconds", "success"
            )
        else:
            self.log(f"Operation took {self.elapsed:.2f} seconds", "info")

    @property
    def duration(self) -> float:
        """
        Get the elapsed time in seconds.

        Returns:
            float: The elapsed time in seconds
        """
        return self.elapsed


class TimeoutManager(Logger):
    """
    Provides timeout functionality for function execution.

    This class contains methods for handling timeouts in function execution,
    useful for operations that might hang or take too long.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the TimeoutManager.

        Args:
            verbose (bool, optional): Whether to log timeout events. Defaults to True.
        """
        super().__init__(name="TimeoutManager", verbose=verbose)

    @staticmethod
    def timeout_handler(signum: int, frame: Any) -> None:
        """
        Handler for SIGALRM signal.

        Args:
            signum: Signal number
            frame: Current stack frame

        Raises:
            TimeoutException: Always raised to indicate timeout
        """
        raise TimeoutException("Function execution timed out")

    @classmethod
    def with_timeout(cls, seconds: int, verbose: bool = True) -> Callable[[T], T]:
        """
        Decorator to limit the execution time of a function.

        Args:
            seconds: Maximum number of seconds the function is allowed to run
            verbose: Whether to log timeout events

        Returns:
            Function decorator that raises TimeoutException if the function takes too long

        Example:

        .. code-block:: python

            @TimeoutManager.with_timeout(5)
            def slow_function():
                time.sleep(10)  # This will raise TimeoutException after 5 seconds
        """
        # Create a logger instance
        logger = cls(verbose=verbose)

        def decorator(func: T) -> T:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Windows doesn't support SIGALRM
                if os.name == "nt":
                    logger.log(
                        f"Timeout not supported on Windows, executing {func.__name__} without timeout",
                        "warning",
                    )
                    return func(*args, **kwargs)

                # On Unix systems, use SIGALRM for timeout
                original_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, cls.timeout_handler)

                try:
                    # Set alarm
                    signal.alarm(int(seconds))
                    logger.log(
                        f"Set timeout of {seconds} seconds for {func.__name__}", "debug"
                    )
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Clear the alarm
                    return result
                except TimeoutException:
                    logger.log(
                        f"Function {func.__name__} timed out after {seconds} seconds",
                        "error",
                    )
                    raise
                finally:
                    # Cancel alarm and restore original handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)

            return cast(T, wrapper)

        return decorator


# For backward compatibility and simpler usage
def with_timeout(seconds: int, verbose: bool = True) -> Callable[[T], T]:
    """
    Decorator to limit the execution time of a function.

    This is a convenience function that calls TimeoutManager.with_timeout.

    Args:
        seconds: Maximum number of seconds the function is allowed to run
        verbose: Whether to log timeout events

    Returns:
        Function decorator that raises TimeoutException if the function takes too long

    Example:

    .. code-block:: python

        @with_timeout(5)
        def slow_function():
            time.sleep(10)  # This will raise TimeoutException after 5 seconds
    """
    return TimeoutManager.with_timeout(seconds, verbose)


def time_function(func: Callable) -> Callable:
    """
    Decorator to time the execution of a function.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs its execution time

    Example:

    .. code-block:: python

        @time_function
        def my_function():
            time.sleep(1)
            # Will log "my_function took 1.00 seconds"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)

    return wrapper
