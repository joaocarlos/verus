import os
import signal
from functools import wraps


class TimeoutException(Exception):
    """Exception raised when a function execution times out."""

    pass


def timeout_handler(signum, frame):
    """Handler for SIGALRM signal."""
    raise TimeoutException("Function execution timed out")


def with_timeout(seconds):
    """
    Decorator to limit the execution time of a function.

    Args:
        seconds (int): Maximum number of seconds the function is allowed to run

    Returns:
        Function decorator that raises TimeoutException if the function takes too long

    Example:
        @with_timeout(5)
        def slow_function():
            time.sleep(10)  # This will raise TimeoutException after 5 seconds
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Windows doesn't support SIGALRM
            if os.name == "nt":
                return func(*args, **kwargs)

            # On Unix systems, use SIGALRM for timeout
            original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)

            try:
                # Set alarm
                signal.alarm(int(seconds))
                result = func(*args, **kwargs)
                return result
            finally:
                # Cancel alarm and restore original handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)

        return wrapper

    return decorator


class Timer:
    """
    Simple context manager for timing code execution.

    Example:
        with Timer("My operation"):
            time.sleep(1)
            # Will print "My operation took 1.00 seconds"
    """

    def __init__(self, name=None):
        """
        Initialize the timer.

        Args:
            name (str, optional): Name of the operation being timed. Defaults to None.
        """
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Start timing when entering the context."""
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Print elapsed time when exiting the context."""
        import time

        elapsed = time.time() - self.start_time

        if self.name:
            print(f"{self.name} took {elapsed:.2f} seconds")
        else:
            print(f"Operation took {elapsed:.2f} seconds")
